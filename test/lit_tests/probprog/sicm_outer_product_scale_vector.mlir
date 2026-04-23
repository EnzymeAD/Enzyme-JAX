// RUN: enzymexlamlir-opt --inline-mcmc-regions --sicm %s | FileCheck %s

// ============================================================================
// Test: Cholesky outer-product scale for VECTOR sigma_k (real hierarchical MVN)
// ============================================================================
//
// Unlike the scalar sigma test, this uses a vector sigma_k with K different
// values. The outer product is formed via two separate reshapes:
//   sigma_col = reshape(sigma_k, K, 1)  -> broadcast to KxK (row broadcast)
//   sigma_row = sigma_k (already 1xK)   -> broadcast to KxK (col broadcast)
//
// The two broadcasts have DIFFERENT source SSA values (one through a reshape,
// one direct), even though they derive from the same underlying vector.

module {
  func.func private @normal_vec(%rng : tensor<2xui64>, %mean : tensor<1x3xf64>, %std : tensor<1x3xf64>)
      -> (tensor<2xui64>, tensor<1x3xf64>) {
    %sample = arith.addf %mean, %std : tensor<1x3xf64>
    return %rng, %sample : tensor<2xui64>, tensor<1x3xf64>
  }

  func.func private @normal_vec_logpdf(%x : tensor<1x3xf64>, %mean : tensor<1x3xf64>, %std : tensor<1x3xf64>)
      -> tensor<f64> {
    %zero = arith.constant dense<0.0> : tensor<f64>
    %sum = stablehlo.reduce(%x init: %zero) applies stablehlo.add
        across dimensions = [0, 1] : (tensor<1x3xf64>, tensor<f64>) -> tensor<f64>
    %neg = arith.negf %sum : tensor<f64>
    return %neg : tensor<f64>
  }

  // CHECK-LABEL: func.func @test_vector_outer_product_scale
  // chol(Omega) should be hoisted before mcmc_region:
  // CHECK: stablehlo.cholesky
  // CHECK: enzyme.mcmc_region
  // No cholesky inside mcmc_region:
  // CHECK-NOT: stablehlo.cholesky
  // CHECK: enzyme.yield

  func.func @test_vector_outer_product_scale(%rng : tensor<2xui64>,
                                              %prior : tensor<1x3xf64>,
                                              %Omega : tensor<3x3xf64>,
                                              %eta : tensor<4x3xf64>,
                                              %trace : tensor<1x3xf64>)
      -> (tensor<1x3xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:8 = enzyme.mcmc @model_vector_outer_product(%rng, %prior, %Omega, %eta) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>]],
      all_addresses = [[#enzyme.symbol<1>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<1x3xf64>, tensor<3x3xf64>, tensor<4x3xf64>, tensor<1x3xf64>, tensor<f64>)
        -> (tensor<1x3xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<f64>, tensor<f64>, tensor<1x3xf64>)

    return %result#0, %result#1, %result#2 : tensor<1x3xf64>, tensor<1xi1>, tensor<2xui64>
  }

  func.func @model_vector_outer_product(%rng : tensor<2xui64>, %prior : tensor<1x3xf64>,
                                         %Omega : tensor<3x3xf64>, %eta : tensor<4x3xf64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<1x3xf64>
    %zero = arith.constant dense<0.0> : tensor<f64>

    // Sample log_sigma_k: vector of K=3 values
    %ls:2 = enzyme.sample @normal_vec(%rng, %prior, %std) {
      logpdf = @normal_vec_logpdf,
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<1x3xf64>, tensor<1x3xf64>) -> (tensor<2xui64>, tensor<1x3xf64>)

    // sigma_k = exp(log_sigma_k): vector of K values
    %sigma_k = stablehlo.exponential %ls#1 : tensor<1x3xf64>

    // Reshape for row broadcast: sigma_col = reshape(sigma_k, 3, 1)
    %sigma_col = stablehlo.reshape %sigma_k : (tensor<1x3xf64>) -> tensor<3x1xf64>

    // Row broadcast: row_bcast[i,j] = sigma_i
    %row_bcast = stablehlo.broadcast_in_dim %sigma_col, dims = [0, 1]
        : (tensor<3x1xf64>) -> tensor<3x3xf64>
    // Col broadcast: col_bcast[i,j] = sigma_j (sigma_k is already 1x3)
    %col_bcast = stablehlo.broadcast_in_dim %sigma_k, dims = [0, 1]
        : (tensor<1x3xf64>) -> tensor<3x3xf64>

    // D = outer_product(sigma, sigma) = sigma_i * sigma_j
    %D = stablehlo.multiply %row_bcast, %col_bcast : tensor<3x3xf64>

    // Sigma = D ⊙ Omega
    %Sigma = stablehlo.multiply %D, %Omega : tensor<3x3xf64>

    // L = cholesky(Sigma)
    %L = stablehlo.cholesky %Sigma : tensor<3x3xf64>

    // NCP transform: y_scaled = eta @ L
    %y_scaled = stablehlo.dot_general %eta, %L,
        contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
        : (tensor<4x3xf64>, tensor<3x3xf64>) -> tensor<4x3xf64>

    // Result = sum(y_scaled)
    %result = stablehlo.reduce(%y_scaled init: %zero) applies stablehlo.add
        across dimensions = [0, 1] : (tensor<4x3xf64>, tensor<f64>) -> tensor<f64>

    return %ls#0, %result : tensor<2xui64>, tensor<f64>
  }
}
