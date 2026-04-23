// RUN: enzymexlamlir-opt --inline-mcmc-regions --sicm %s | FileCheck %s

// ============================================================================
// Test: Eigendecomposition lift for cholesky(A + d*I) → syevd(A) + eigen math
// ============================================================================
//
// Model:
//   noise ~ Normal(prior_noise, 1.0)              <- sample site
//   shifted_cov = base_cov + broadcast(noise) * I  <- diagonal additive shift
//   L = cholesky(shifted_cov)                      <- expensive O(n^3)
//   x = triangular_solve(L, b)                     <- forward solve
//   result = x[0,0]
//
// Without SICM: cholesky(A + noise*I) is sample-dependent, stays inside loop.
//
// With --sicm (eigendecomposition lift):
//   Phase 1 (rewrite):
//     syevd(base_cov) → Q, lambda  (sample-invariant)
//     triangular_solve(L, b) → Q @ diag(1/sqrt(lambda+noise)) @ Q^T @ b
//   Phase 2 (hoist):
//     syevd(base_cov), transpose(Q), and Q^T @ b are hoisted before mcmc_region
//
// Result: expensive O(n^3) eigendecomposition computed once outside MCMC loop.
//         Per-iteration cost is O(n^2) matrix ops on eigenvalues.

// CHECK-LABEL: func.func @test_eigen_lift
// Eigendecomposition hoisted BEFORE mcmc_region
// CHECK: %[[Q:.*]], %[[LAMBDA:.*]], %[[INFO:.*]] = enzymexla.lapack.syevd
// CHECK-SAME: uplo = #enzymexla.uplo<L>
// Q^T hoisted
// CHECK: %[[QT:.*]] = stablehlo.transpose %[[Q]], dims = [1, 0]
// Q^T @ b hoisted (both Q^T and b are sample-invariant)
// CHECK: %[[QTB:.*]] = stablehlo.dot_general %[[QT]], %{{.*}}, contracting_dims = [1] x [0]
// No cholesky or triangular_solve anywhere
// CHECK-NOT: stablehlo.cholesky
// CHECK-NOT: stablehlo.triangular_solve
// CHECK: enzyme.mcmc_region
// Inside: eigenvalue manipulation, no cholesky/triangular_solve
// CHECK: enzyme.sample_region
// CHECK: stablehlo.add %[[LAMBDA]], %{{.*}} : tensor<3xf64>
// CHECK: stablehlo.sqrt {{.*}} : tensor<3xf64>
// CHECK: stablehlo.divide {{.*}} : tensor<3xf64>
// CHECK: stablehlo.broadcast_in_dim
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

  func.func @test_eigen_lift(%rng : tensor<2xui64>, %prior_noise : tensor<f64>,
                              %base_cov : tensor<3x3xf64>, %b : tensor<3x1xf64>,
                              %trace : tensor<1x1xf64>)
      -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:8 = enzyme.mcmc @model_eigen(%rng, %prior_noise, %base_cov, %b) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>]],
      all_addresses = [[#enzyme.symbol<1>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<3x3xf64>, tensor<3x1xf64>, tensor<1x1xf64>, tensor<f64>)
        -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)

    return %result#0, %result#1, %result#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  func.func @model_eigen(%rng : tensor<2xui64>, %prior_noise : tensor<f64>,
                           %base_cov : tensor<3x3xf64>, %b : tensor<3x1xf64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<f64>

    // Sample noise parameter
    %noise:2 = enzyme.sample @normal(%rng, %prior_noise, %std) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    // Build A + noise * I
    %identity = stablehlo.constant dense<
      [[1.0, 0.0, 0.0],
       [0.0, 1.0, 0.0],
       [0.0, 0.0, 1.0]]> : tensor<3x3xf64>
    %noise_bcast = stablehlo.broadcast_in_dim %noise#1, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %scaled_I = stablehlo.multiply %noise_bcast, %identity : tensor<3x3xf64>
    %shifted = stablehlo.add %base_cov, %scaled_I : tensor<3x3xf64>

    // Cholesky decomposition (expensive, sample-dependent)
    %L = stablehlo.cholesky %shifted, lower = true : tensor<3x3xf64>

    // Triangular solve: L^{-1} @ b
    %x = "stablehlo.triangular_solve"(%L, %b) <{
      left_side = true, lower = true,
      unit_diagonal = false,
      transpose_a = #stablehlo<transpose NO_TRANSPOSE>
    }> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>

    // Use first element as result
    %c0 = arith.constant 0 : index
    %elem = tensor.extract %x[%c0, %c0] : tensor<3x1xf64>
    %result = tensor.from_elements %elem : tensor<f64>

    return %noise#0, %result : tensor<2xui64>, tensor<f64>
  }
}
