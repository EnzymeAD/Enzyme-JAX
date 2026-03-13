// RUN: enzymexlamlir-opt --inline-mcmc-regions --sicm %s | FileCheck %s

// ============================================================================
// Test: Eigendecomposition lift for cholesky(s*A + t*I) where A is invariant
// ============================================================================
//
// Model (GP with fixed lengthscale / Linear Mixed Model pattern):
//   alpha ~ Normal(prior_alpha, 1.0)
//   sigma ~ Normal(prior_sigma, 1.0)
//   K = alpha^2 * base_cov + sigma^2 * I    <- two sample-dependent scalars
//   L = cholesky(K)
//   x = triangular_solve(L, b)
//
// Without SICM: cholesky is called every iteration (O(N^3)).
//
// With SICM (generalized eigendecomposition lift):
//   Phase 1 (rewrite):
//     syevd(base_cov) -> Q, lambda  (sample-invariant)
//     triangular_solve(L, b) -> Q @ diag(1/sqrt(alpha^2*lambda + sigma^2)) @ Q^T @ b
//   Phase 2 (hoist):
//     syevd, Q^T, and Q^T @ b are hoisted before mcmc_region
//
// Result: O(N^3) eigendecomposition computed once, per-iteration cost is O(N^2).

// CHECK-LABEL: func.func @test_scaled_eigen_lift
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
// Inside: scaled eigenvalue manipulation
// CHECK: enzyme.sample_region
// CHECK: enzyme.sample_region
// Eigenvalue scaling: alpha^2 * lambda + sigma^2
// CHECK: stablehlo.multiply {{.*}} %[[LAMBDA]]
// CHECK: stablehlo.add
// CHECK: stablehlo.sqrt
// CHECK: stablehlo.divide
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

  func.func @test_scaled_eigen_lift(%rng : tensor<2xui64>,
                                     %prior_alpha : tensor<f64>,
                                     %prior_sigma : tensor<f64>,
                                     %base_cov : tensor<3x3xf64>,
                                     %b : tensor<3x1xf64>,
                                     %trace : tensor<1x2xf64>)
      -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:8 = enzyme.mcmc @model_scaled_eigen(%rng, %prior_alpha, %prior_sigma, %base_cov, %b) given %trace
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

  func.func @model_scaled_eigen(%rng : tensor<2xui64>,
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

    // Cholesky
    %L = stablehlo.cholesky %K, lower = true : tensor<3x3xf64>

    // Triangular solve: L^{-1} @ b
    %x = "stablehlo.triangular_solve"(%L, %b) <{
      left_side = true, lower = true,
      unit_diagonal = false,
      transpose_a = #stablehlo<transpose NO_TRANSPOSE>
    }> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>

    %c0 = arith.constant 0 : index
    %elem = tensor.extract %x[%c0, %c0] : tensor<3x1xf64>
    %result = tensor.from_elements %elem : tensor<f64>

    return %sigma#0, %result : tensor<2xui64>, tensor<f64>
  }
}
