// RUN: enzymexlamlir-opt --inline-mcmc-regions --sicm %s | FileCheck %s

// ============================================================================
// Test: Eigendecomposition lift for full Cholesky solve (upper triangular)
// ============================================================================
//
// This test matches the actual IR produced by Reactant.jl's MVN logpdf:
//   C = cholesky(K, lower=false)    -- upper triangular
//   result = C \ b                  -- two triangular_solves
//
// Which produces:
//   %U = stablehlo.cholesky %K, lower = false
//   %v = triangular_solve(%U, %b)  {transpose_a = ADJOINT}      -- U^H \ b
//   %w = triangular_solve(%U, %v)  {transpose_a = NO_TRANSPOSE} -- U \ v
//
// The combined effect is w = Σ^{-1} b.
//
// With SICM (eigendecomposition lift):
//   syevd(A) -> Q, λ  (hoisted)
//   Σ^{-1} b = Q @ diag(1/(s*λ+t)) @ Q^T @ b
//
// CHECK-LABEL: func.func @test_full_solve_eigen_lift
// Eigendecomposition hoisted BEFORE mcmc_region
// CHECK: %[[Q:.*]], %[[LAMBDA:.*]], %[[INFO:.*]] = enzymexla.lapack.syevd
// CHECK-SAME: uplo = #enzymexla.uplo<L>
// Q^T hoisted
// CHECK: %[[QT:.*]] = stablehlo.transpose %[[Q]], dims = [1, 0]
// Q^T @ b hoisted
// CHECK: %[[QTB:.*]] = stablehlo.dot_general %[[QT]], %{{.*}}, contracting_dims = [1] x [0]
// No cholesky or triangular_solve anywhere
// CHECK-NOT: stablehlo.cholesky
// CHECK-NOT: stablehlo.triangular_solve
// CHECK: enzyme.mcmc_region
// Inside: eigenvalue scaling with 1/d (not 1/sqrt(d))
// CHECK: enzyme.sample_region
// CHECK: enzyme.sample_region
// Eigenvalue scaling: alpha^2 * lambda + sigma^2
// CHECK: stablehlo.multiply {{.*}} %[[LAMBDA]]
// CHECK: stablehlo.add
// Full inverse: 1/d (not sqrt)
// CHECK: stablehlo.divide
// CHECK-NOT: stablehlo.sqrt
// CHECK: stablehlo.multiply %[[Q]]
// CHECK: stablehlo.dot_general {{.*}} %[[QTB]]
// CHECK-NOT: stablehlo.cholesky
// CHECK-NOT: stablehlo.triangular_solve
// CHECK: enzyme.yield

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

  func.func @test_full_solve_eigen_lift(%rng : tensor<2xui64>,
                                         %prior_alpha : tensor<f64>,
                                         %prior_sigma : tensor<f64>,
                                         %base_cov : tensor<3x3xf64>,
                                         %b : tensor<3x1xf64>,
                                         %trace : tensor<1x2xf64>)
      -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:8 = enzyme.mcmc @model_full_solve(%rng, %prior_alpha, %prior_sigma, %base_cov, %b) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>], [#enzyme.symbol<2>]],
      all_addresses = [[#enzyme.symbol<1>], [#enzyme.symbol<2>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<3x3xf64>, tensor<3x1xf64>, tensor<1x2xf64>, tensor<f64>)
        -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>)

    return %result#0, %result#1, %result#2 : tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>
  }

  func.func @model_full_solve(%rng : tensor<2xui64>,
                                %prior_alpha : tensor<f64>,
                                %prior_sigma : tensor<f64>,
                                %base_cov : tensor<3x3xf64>,
                                %b : tensor<3x1xf64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<f64>
    %identity = stablehlo.constant dense<
      [[1.0, 0.0, 0.0],
       [0.0, 1.0, 0.0],
       [0.0, 0.0, 1.0]]> : tensor<3x3xf64>

    // Sample alpha
    %alpha:2 = enzyme.sample @normal(%rng, %prior_alpha, %std) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    // Sample sigma
    %sigma:2 = enzyme.sample @normal(%alpha#0, %prior_sigma, %std) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<2>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    // K = alpha^2 * base_cov + sigma^2 * I
    %alpha_sq = arith.mulf %alpha#1, %alpha#1 : tensor<f64>
    %alpha_bcast = stablehlo.broadcast_in_dim %alpha_sq, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %scaled_A = stablehlo.multiply %alpha_bcast, %base_cov : tensor<3x3xf64>

    %sigma_sq = arith.mulf %sigma#1, %sigma#1 : tensor<f64>
    %sigma_bcast = stablehlo.broadcast_in_dim %sigma_sq, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %scaled_I = stablehlo.multiply %sigma_bcast, %identity : tensor<3x3xf64>

    %K = stablehlo.add %scaled_A, %scaled_I : tensor<3x3xf64>

    // Upper Cholesky (matches Reactant default: lower=false)
    %U = stablehlo.cholesky %K, lower = false : tensor<3x3xf64>

    // Full solve: U^{-1}(U^{-H} b) = Σ^{-1} b
    %v = "stablehlo.triangular_solve"(%U, %b) <{
      left_side = true, lower = false,
      unit_diagonal = false,
      transpose_a = #stablehlo<transpose ADJOINT>
    }> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>

    %w = "stablehlo.triangular_solve"(%U, %v) <{
      left_side = true, lower = false,
      unit_diagonal = false,
      transpose_a = #stablehlo<transpose NO_TRANSPOSE>
    }> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>

    %c0 = arith.constant 0 : index
    %elem = tensor.extract %w[%c0, %c0] : tensor<3x1xf64>
    %result = tensor.from_elements %elem : tensor<f64>

    return %sigma#0, %result : tensor<2xui64>, tensor<f64>
  }
}
