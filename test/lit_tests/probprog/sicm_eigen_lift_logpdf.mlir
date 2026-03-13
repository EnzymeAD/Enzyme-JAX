// RUN: enzymexlamlir-opt --inline-mcmc-regions --sicm %s | FileCheck %s

// ============================================================================
// Test: Eigendecomposition lift for K = alpha^2*A + sigma^2*I
// ============================================================================
//
// The Cholesky + triangular_solve chain in the MVN logpdf is replaced by
// eigendecomposition of the base covariance A. SICM hoists the eigendecomp
// before the mcmc_region, then rewrites the logpdf to use eigenvalue scaling:
//   K^{-1} = Q * diag(1/(alpha^2*lambda + sigma^2)) * Q^T

// CHECK-LABEL: func.func @test_logpdf_eigen_lift
// Eigendecomposition hoisted before mcmc_region
// CHECK: %[[Q:.+]], %[[LAMBDA:.+]], %{{.*}} = enzymexla.lapack.syevd
// CHECK-NEXT: %[[QT:.+]] = stablehlo.transpose %[[Q]]
// CHECK: enzyme.mcmc_region
// Three sample_region ops (alpha, sigma, y)
// CHECK: enzyme.sample_region
// CHECK: enzyme.sample_region
// K = alpha^2 * base_cov + sigma^2 * I
// CHECK: stablehlo.add %{{.*}} : tensor<3x3xf64>
// CHECK: enzyme.sample_region
// Unified logpdf: eigenvalue-based solve replaces cholesky
// CHECK: ^bb0(%{{.*}}: tensor<f64>, %{{.*}}: tensor<f64>, %{{.*}}: tensor<3xf64>):
// Eigenvalue scaling: alpha^2 * lambda + sigma^2
// CHECK: stablehlo.multiply %{{.*}}, %[[LAMBDA]]
// CHECK: stablehlo.divide
// Eigenvector-based solve: Q^T @ x, then Q * diag(1/d) @ Q^T x
// CHECK: stablehlo.dot_general %[[QT]]
// CHECK: stablehlo.multiply %[[Q]]
// CHECK: stablehlo.dot_general
// No cholesky or triangular_solve in the logpdf
// CHECK-NOT: stablehlo.cholesky
// CHECK-NOT: stablehlo.triangular_solve
// CHECK: enzyme.yield
// CHECK: num_position_args = 3

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

  // MVN sampler and logpdf for y ~ MVN(0, K)
  func.func private @mvn_sampler(%rng : tensor<2xui64>, %K : tensor<3x3xf64>)
      -> (tensor<2xui64>, tensor<3xf64>) {
    %zeros = stablehlo.constant dense<0.0> : tensor<3xf64>
    return %rng, %zeros : tensor<2xui64>, tensor<3xf64>
  }

  func.func private @mvn_logpdf(%x : tensor<3xf64>, %K : tensor<3x3xf64>)
      -> tensor<f64> {
    // Cholesky (upper triangular, matching Reactant)
    %U = stablehlo.cholesky %K, lower = false : tensor<3x3xf64>

    // Full solve: U^{-1}(U^{-H} x)
    %x_col = stablehlo.reshape %x : (tensor<3xf64>) -> tensor<3x1xf64>
    %v = "stablehlo.triangular_solve"(%U, %x_col) <{
      left_side = true, lower = false,
      unit_diagonal = false,
      transpose_a = #stablehlo<transpose ADJOINT>
    }> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>

    %w = "stablehlo.triangular_solve"(%U, %v) <{
      left_side = true, lower = false,
      unit_diagonal = false,
      transpose_a = #stablehlo<transpose NO_TRANSPOSE>
    }> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>

    // Compute quadratic form: x^T @ K^{-1} @ x
    %w_flat = stablehlo.reshape %w : (tensor<3x1xf64>) -> tensor<3xf64>
    %prod = stablehlo.multiply %x, %w_flat : tensor<3xf64>
    %zero = stablehlo.constant dense<0.0> : tensor<f64>
    %quad = stablehlo.reduce(%prod init: %zero) across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
      reducer(%a: tensor<f64>, %b: tensor<f64>) {
        %s = stablehlo.add %a, %b : tensor<f64>
        stablehlo.return %s : tensor<f64>
      }

    %neg_half = stablehlo.constant dense<-5.000000e-01> : tensor<f64>
    %result = stablehlo.multiply %neg_half, %quad : tensor<f64>
    return %result : tensor<f64>
  }

  func.func @test_logpdf_eigen_lift(%rng : tensor<2xui64>,
                                     %prior_alpha : tensor<f64>,
                                     %prior_sigma : tensor<f64>,
                                     %base_cov : tensor<3x3xf64>,
                                     %trace : tensor<1x5xf64>)
      -> (tensor<1x5xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:8 = enzyme.mcmc @model_logpdf_eigen(%rng, %prior_alpha, %prior_sigma, %base_cov) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>], [#enzyme.symbol<2>], [#enzyme.symbol<3>]],
      all_addresses = [[#enzyme.symbol<1>], [#enzyme.symbol<2>], [#enzyme.symbol<3>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<3x3xf64>, tensor<1x5xf64>, tensor<f64>)
        -> (tensor<1x5xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x5xf64>, tensor<1x5xf64>, tensor<f64>, tensor<f64>, tensor<1x5xf64>)

    return %result#0, %result#1, %result#2 : tensor<1x5xf64>, tensor<1xi1>, tensor<2xui64>
  }

  func.func @model_logpdf_eigen(%rng : tensor<2xui64>,
                                  %prior_alpha : tensor<f64>,
                                  %prior_sigma : tensor<f64>,
                                  %base_cov : tensor<3x3xf64>)
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

    // Sample y ~ MVN(0, K) with logpdf that contains cholesky + solve chain
    %y:2 = enzyme.sample @mvn_sampler(%sigma#0, %K) {
      logpdf = @mvn_logpdf,
      symbol = #enzyme.symbol<3>
    } : (tensor<2xui64>, tensor<3x3xf64>) -> (tensor<2xui64>, tensor<3xf64>)

    %c0 = arith.constant 0 : index
    %elem = tensor.extract %y#1[%c0] : tensor<3xf64>
    %result = tensor.from_elements %elem : tensor<f64>

    return %y#0, %result : tensor<2xui64>, tensor<f64>
  }
}
