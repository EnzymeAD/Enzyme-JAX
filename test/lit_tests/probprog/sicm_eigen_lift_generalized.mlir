// RUN: enzymexlamlir-opt --inline-mcmc-regions --sicm %s | FileCheck %s

// ============================================================================
// Test: Generalized eigendecomposition lift for cholesky(s*A + t*B)
// ============================================================================
//
// Model (phylogenetic regression):
//   sigma_p ~ Normal(prior, 1.0)       <- sample site 1
//   sigma_e ~ Normal(prior, 1.0)       <- sample site 2
//   K = sigma_p^2 * C_phylo + sigma_e^2 * C_env
//   L = cholesky(K)
//   full_solve: L\(L^T\b) = K^{-1}b
//   diag_L: diag(L) for log-determinant
//   result = full_solve[0,0] + sum(log(diag_L))
//
// Without SICM: cholesky(s*A + t*B) recomputed every iteration.
//
// With --sicm (generalized eigendecomposition lift):
//   Phase 1 (rewrite):
//     L_B = chol(C_env)                          (invariant)
//     C_w = L_B^{-1} * C_phylo * L_B^{-T}       (whitened, invariant)
//     Q, lambda = syevd(C_w)                     (invariant)
//     Full solve: K^{-1}b = L_B^{-T} * Q * diag(1/(s*lambda+t)) * Q^T * L_B^{-1} * b
//     Diagonal: diag(chol(K)) -> diag(L_B) .* sqrt(s*lambda+t)
//   Phase 2 (hoist):
//     L_B, C_w, syevd, Q^T, L_B^{-1}*b, diag(L_B) hoisted before mcmc_region
//
// Result: O(n^3) cholesky per iteration -> O(n^2) eigen-based solve per iteration.

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

  // CHECK-LABEL: func.func @test_generalized_eigen_lift
  // Whitening chain hoisted before mcmc_region:
  //   1. chol(C_env)
  //   2. L_B^{-1} * C_phylo * L_B^{-T}
  //   3. syevd(C_w)
  //   4. diag(L_B) extraction
  // CHECK: %[[LB:.+]] = stablehlo.cholesky %{{.*}}, lower = true : tensor<3x3xf64>
  // CHECK: stablehlo.triangular_solve
  // CHECK: %[[Q:.+]], %[[LAMBDA:.+]], %{{.*}} = enzymexla.lapack.syevd
  // CHECK: enzyme.mcmc_region
  // No cholesky inside mcmc_region
  // CHECK-NOT: stablehlo.cholesky
  // CHECK: enzyme.yield

  func.func @test_generalized_eigen_lift(%rng : tensor<2xui64>,
                                          %prior : tensor<f64>,
                                          %C_phylo : tensor<3x3xf64>,
                                          %C_env : tensor<3x3xf64>,
                                          %b : tensor<3x1xf64>,
                                          %trace : tensor<1x2xf64>)
      -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:8 = enzyme.mcmc @model_gen_eigen(%rng, %prior, %C_phylo, %C_env, %b) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>], [#enzyme.symbol<2>]],
      all_addresses = [[#enzyme.symbol<1>], [#enzyme.symbol<2>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<3x3xf64>, tensor<3x3xf64>, tensor<3x1xf64>, tensor<1x2xf64>, tensor<f64>)
        -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>)

    return %result#0, %result#1, %result#2 : tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>
  }

  func.func @model_gen_eigen(%rng : tensor<2xui64>, %prior : tensor<f64>,
                               %C_phylo : tensor<3x3xf64>, %C_env : tensor<3x3xf64>,
                               %b : tensor<3x1xf64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<f64>
    %zero = arith.constant dense<0.0> : tensor<f64>

    // Identity matrix for diagonal extraction
    %I = stablehlo.constant dense<[[1.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 0.0, 1.0]]> : tensor<3x3xf64>

    // Sample sigma_p
    %sp:2 = enzyme.sample @normal(%rng, %prior, %std) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    // Sample sigma_e
    %se:2 = enzyme.sample @normal(%sp#0, %prior, %std) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<2>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    // K = sigma_p^2 * C_phylo + sigma_e^2 * C_env
    %sp_sq = stablehlo.multiply %sp#1, %sp#1 : tensor<f64>
    %sp_bcast = stablehlo.broadcast_in_dim %sp_sq, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %sA = stablehlo.multiply %sp_bcast, %C_phylo : tensor<3x3xf64>

    %se_sq = stablehlo.multiply %se#1, %se#1 : tensor<f64>
    %se_bcast = stablehlo.broadcast_in_dim %se_sq, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %tB = stablehlo.multiply %se_bcast, %C_env : tensor<3x3xf64>

    %K = stablehlo.add %sA, %tB : tensor<3x3xf64>

    // Cholesky decomposition (expensive, sample-dependent)
    %L = stablehlo.cholesky %K, lower = true : tensor<3x3xf64>

    // Full solve chain: L^{-1}(L^{-T} b) = K^{-1} b
    %inner = "stablehlo.triangular_solve"(%L, %b) <{
      left_side = true, lower = true,
      unit_diagonal = false,
      transpose_a = #stablehlo<transpose ADJOINT>
    }> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>

    %outer = "stablehlo.triangular_solve"(%L, %inner) <{
      left_side = true, lower = true,
      unit_diagonal = false,
      transpose_a = #stablehlo<transpose NO_TRANSPOSE>
    }> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>

    // Diagonal extraction: diag(L) via dot_general(L, I)
    %diag_L = "stablehlo.dot_general"(%L, %I) <{
      dot_dimension_numbers = #stablehlo.dot<
        lhs_batching_dimensions = [1],
        rhs_batching_dimensions = [1],
        lhs_contracting_dimensions = [0],
        rhs_contracting_dimensions = [0]
      >
    }> : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3xf64>

    // Log-determinant: sum(log(diag(L)))
    %log_diag = stablehlo.log %diag_L : tensor<3xf64>
    %log_det = stablehlo.reduce(%log_diag init: %zero) applies stablehlo.add across dimensions = [0]
        : (tensor<3xf64>, tensor<f64>) -> tensor<f64>

    // Use first element of solve + log_det as result
    %c0 = arith.constant 0 : index
    %elem = tensor.extract %outer[%c0, %c0] : tensor<3x1xf64>
    %solve_val = tensor.from_elements %elem : tensor<f64>
    %result = stablehlo.add %solve_val, %log_det : tensor<f64>

    return %se#0, %result : tensor<2xui64>, tensor<f64>
  }
}
