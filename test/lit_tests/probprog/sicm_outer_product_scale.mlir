// RUN: enzymexlamlir-opt --inline-mcmc-regions --sicm %s | FileCheck %s

// ============================================================================
// Test: Cholesky outer-product scale factorization for chol(D ⊙ Ω)
// ============================================================================
//
// Model (hierarchical MVN):
//   log_sigma ~ Normal(prior, 1.0)        <- sample site 1
//   sigma = exp(log_sigma)
//   D = sigma * sigma' (outer product)
//   Sigma = D ⊙ Omega (correlation → covariance)
//   L = cholesky(Sigma)
//   y_scaled = eta * L (NCP transform)
//   result = sum(y_scaled)
//
// Without SICM: cholesky(D ⊙ Ω) recomputed every iteration.
//
// With --sicm (outer product scale factorization):
//   chol(diag(σ) Ω diag(σ)) = chol(Ω) diag(σ)  [upper triangular, lower=false]
//   chol(Ω) is invariant and hoisted before mcmc_region.
//
// Result: O(n^3) cholesky per iteration -> O(n^2) diagonal scale per iteration.

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %std : tensor<f64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %sample = arith.addf %mean, %std : tensor<f64>
    return %rng, %sample : tensor<2xui64>, tensor<f64>
  }

  func.func private @normal_logpdf(%x : tensor<f64>, %mean : tensor<f64>, %std : tensor<f64>)
      -> tensor<f64> {
    %neg = arith.negf %x : tensor<f64>
    return %neg : tensor<f64>
  }

  // CHECK-LABEL: func.func @test_outer_product_scale
  // chol(Omega) hoisted before mcmc_region:
  // CHECK: stablehlo.cholesky
  // CHECK: enzyme.mcmc_region
  // No cholesky inside mcmc_region
  // CHECK-NOT: stablehlo.cholesky
  // CHECK: enzyme.yield

  func.func @test_outer_product_scale(%rng : tensor<2xui64>,
                                       %prior : tensor<f64>,
                                       %Omega : tensor<3x3xf64>,
                                       %eta : tensor<4x3xf64>,
                                       %trace : tensor<1x1xf64>)
      -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:8 = enzyme.mcmc @model_outer_product(%rng, %prior, %Omega, %eta) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>]],
      all_addresses = [[#enzyme.symbol<1>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<3x3xf64>, tensor<4x3xf64>, tensor<1x1xf64>, tensor<f64>)
        -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)

    return %result#0, %result#1, %result#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  func.func @model_outer_product(%rng : tensor<2xui64>, %prior : tensor<f64>,
                                  %Omega : tensor<3x3xf64>, %eta : tensor<4x3xf64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<f64>
    %zero = arith.constant dense<0.0> : tensor<f64>

    // Sample log_sigma (scalar sample representing 3 values via prior)
    %ls:2 = enzyme.sample @normal(%rng, %prior, %std) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    // sigma = exp(log_sigma), broadcast to tensor<1x3xf64>
    %sigma_scalar = stablehlo.exponential %ls#1 : tensor<f64>
    %sigma_bcast_1x3 = stablehlo.broadcast_in_dim %sigma_scalar, dims = []
        : (tensor<f64>) -> tensor<1x3xf64>

    // Outer product: D = sigma * sigma' via two broadcasts
    %row_bcast = stablehlo.broadcast_in_dim %sigma_bcast_1x3, dims = [1, 0]
        : (tensor<1x3xf64>) -> tensor<3x3xf64>
    %col_bcast = stablehlo.broadcast_in_dim %sigma_bcast_1x3, dims = [0, 1]
        : (tensor<1x3xf64>) -> tensor<3x3xf64>
    %D = stablehlo.multiply %row_bcast, %col_bcast : tensor<3x3xf64>

    // Sigma = D ⊙ Omega
    %Sigma = stablehlo.multiply %D, %Omega : tensor<3x3xf64>

    // L = cholesky(Sigma) — default lower=false (upper triangular)
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
