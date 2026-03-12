// RUN: enzymexlamlir-opt --inline-mcmc-regions --sicm %s | FileCheck %s

// ============================================================================
// Test: Unselected sample sites are sample-invariant for SICM
// ============================================================================
//
// Model (hierarchical MVN with sampled correlation matrix):
//   Omega ~ LKJ(...)                         <- sample site 0, NOT selected
//   log_sigma ~ Normal(prior, 1.0)           <- sample site 1, SELECTED
//   sigma = exp(log_sigma)
//   D = sigma * sigma' (outer product)
//   Sigma = D ⊙ Omega
//   L = cholesky(Sigma)
//   result = sum(L)
//
// The selection only contains symbol<1> (log_sigma). Omega (symbol<0>) is
// sampled but NOT selected — its value comes from the trace and is fixed
// during MCMC. The analysis should treat Omega as sample-invariant, allowing
// CholeskyOuterProductScale to fire: chol(D ⊙ Ω) → diag(σ) * chol(Ω)
// where chol(Ω) is hoisted before the mcmc_region.

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

  // Stub sampler and logpdf for the correlation matrix (Omega)
  func.func private @lkj_sampler(%rng : tensor<2xui64>, %eta : tensor<f64>)
      -> (tensor<2xui64>, tensor<3x3xf64>) {
    %identity = arith.constant dense<[[1.0, 0.0, 0.0],
                                       [0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0]]> : tensor<3x3xf64>
    return %rng, %identity : tensor<2xui64>, tensor<3x3xf64>
  }

  func.func private @lkj_logpdf(%Omega : tensor<3x3xf64>, %eta : tensor<f64>)
      -> tensor<f64> {
    %zero = arith.constant dense<0.0> : tensor<f64>
    return %zero : tensor<f64>
  }

  // CHECK-LABEL: func.func @test_unselected_sample_invariant
  // chol(Omega) hoisted before mcmc_region because Omega is unselected:
  // CHECK: stablehlo.cholesky
  // CHECK: enzyme.mcmc_region
  // No cholesky inside mcmc_region
  // CHECK-NOT: stablehlo.cholesky
  // CHECK: enzyme.yield

  func.func @test_unselected_sample_invariant(
      %rng : tensor<2xui64>,
      %prior : tensor<f64>,
      %lkj_eta : tensor<f64>,
      %trace : tensor<1x4xf64>)
      -> (tensor<1x4xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    // selection only has symbol<1> (log_sigma), NOT symbol<0> (Omega)
    %result:8 = enzyme.mcmc @model_with_unselected_omega(%rng, %prior, %lkj_eta) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>]],
      all_addresses = [[#enzyme.symbol<0>], [#enzyme.symbol<1>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x4xf64>, tensor<f64>)
        -> (tensor<1x4xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x4xf64>, tensor<1x4xf64>, tensor<f64>, tensor<f64>, tensor<1x4xf64>)

    return %result#0, %result#1, %result#2 : tensor<1x4xf64>, tensor<1xi1>, tensor<2xui64>
  }

  func.func @model_with_unselected_omega(
      %rng : tensor<2xui64>, %prior : tensor<f64>, %lkj_eta : tensor<f64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<f64>
    %zero = arith.constant dense<0.0> : tensor<f64>

    // Sample Omega ~ LKJ (NOT selected for MCMC)
    %omega:2 = enzyme.sample @lkj_sampler(%rng, %lkj_eta) {
      logpdf = @lkj_logpdf,
      symbol = #enzyme.symbol<0>
    } : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<3x3xf64>)

    // Sample log_sigma ~ Normal (SELECTED for MCMC)
    %ls:2 = enzyme.sample @normal(%omega#0, %prior, %std) {
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

    // Sigma = D ⊙ Omega (Omega is from unselected sample — should be invariant)
    %Sigma = stablehlo.multiply %D, %omega#1 : tensor<3x3xf64>

    // L = cholesky(Sigma) — should be factored into diag(sigma) * chol(Omega)
    %L = stablehlo.cholesky %Sigma : tensor<3x3xf64>

    // Result = sum(L)
    %result = stablehlo.reduce(%L init: %zero) applies stablehlo.add
        across dimensions = [0, 1] : (tensor<3x3xf64>, tensor<f64>) -> tensor<f64>

    return %ls#0, %result : tensor<2xui64>, tensor<f64>
  }
}
