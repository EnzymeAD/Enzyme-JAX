// RUN: enzymexlamlir-opt --inline-mcmc-regions --sicm %s | FileCheck %s

// ============================================================================
// Test: Scale-Family MVN — Cholesky factored via SICM
// ============================================================================
//
// Model (scale_family_mvn):
//   tau ~ HalfNormal(1)
//   K = tau^2 * R_data         <- computed in model body
//   y ~ MVN(0, K)              <- Cholesky done inside MVN logpdf
//
// SICM decomposes cholesky(tau^2 * R) into sqrt(tau^2) * cholesky(R),
// hoists cholesky(R) and log(cholesky(R)) before the mcmc_region.

module {
  func.func private @halfnormal_sampler(%rng : tensor<2xui64>, %std : tensor<f64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %sample = math.absf %std : tensor<f64>
    return %rng, %sample : tensor<2xui64>, tensor<f64>
  }

  func.func private @halfnormal_logpdf(%x : tensor<f64>, %std : tensor<f64>)
      -> tensor<f64> {
    %neg = arith.negf %x : tensor<f64>
    return %neg : tensor<f64>
  }

  // MVN sampler (stub — just returns zeros)
  func.func private @mvn_sampler(%rng : tensor<2xui64>, %cov : tensor<3x3xf64>)
      -> (tensor<2xui64>, tensor<3xf64>) {
    %zero = arith.constant dense<0.0> : tensor<3xf64>
    return %rng, %zero : tensor<2xui64>, tensor<3xf64>
  }

  // MVN logpdf — contains Cholesky (the expensive op)
  // logpdf(x, cov) = -0.5 * (x' * cov^-1 * x + log|cov| + n*log(2*pi))
  // Simplified: just does cholesky + triangular_solve + log(diag(L))
  func.func private @mvn_logpdf(%x : tensor<3xf64>, %cov : tensor<3x3xf64>)
      -> tensor<f64> {
    // L = cholesky(cov) — O(N^3), the expensive operation
    %L = stablehlo.cholesky %cov, lower = true : tensor<3x3xf64>

    // Solve: L \ x
    %x_col = stablehlo.reshape %x : (tensor<3xf64>) -> tensor<3x1xf64>
    %solved = "stablehlo.triangular_solve"(%L, %x_col) <{
      left_side = true, lower = true,
      unit_diagonal = false,
      transpose_a = #stablehlo<transpose NO_TRANSPOSE>
    }> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>

    // log(diag(L)) for log determinant — extract [0,0] element
    %log_L = stablehlo.log %L : tensor<3x3xf64>
    %sliced = stablehlo.slice %log_L [0:1, 0:1] : (tensor<3x3xf64>) -> tensor<1x1xf64>
    %result = stablehlo.reshape %sliced : (tensor<1x1xf64>) -> tensor<f64>
    return %result : tensor<f64>
  }

  // ===== Test: Scale-family MVN (Cholesky inside logpdf) =====
  //
  // After inline-mcmc-regions + SICM partial inlining:
  //   1. Phase 0: Cholesky (parameter-only) hoisted from logpdf body to model body
  //   2. Phase 1: CholeskyScaleFactorizationHLO decomposes cholesky(tau^2 * R)
  //              into sqrt(tau^2) * cholesky(R)
  //   3. Phase 1: LogMultiplyDistributionHLO decomposes log(sqrt(tau^2) * chol(R))
  //              into log(sqrt(tau^2)) + log(chol(R))
  //   4. Phase 2: cholesky(R) and log(cholesky(R)) hoisted before mcmc_region

  // CHECK-LABEL: func.func @test_scale_family_mvn
  // Cholesky and log(cholesky) hoisted before mcmc_region
  // CHECK: %[[CHOL:.+]] = stablehlo.cholesky %{{.*}}, lower = true
  // CHECK-NEXT: %[[LOG_CHOL:.+]] = stablehlo.log %[[CHOL]]
  // CHECK: enzyme.mcmc_region
  // CHECK: enzyme.sample_region
  // CHECK: enzyme.sample_region
  // Unified logpdf with tau (position) and y (non-position, observation)
  // CHECK: ^bb0(%[[TAU:.+]]: tensor<f64>, %{{.*}}: tensor<3xf64>):
  // halfnormal logpdf of tau
  // CHECK-NEXT: arith.negf %[[TAU]]
  // log-det: log(sqrt(tau^2)) + hoisted log(chol(R))
  // CHECK: math.sqrt
  // CHECK-NEXT: stablehlo.log
  // CHECK-NEXT: stablehlo.broadcast_in_dim
  // CHECK-NEXT: stablehlo.add %{{.*}}, %[[LOG_CHOL]]
  // CHECK-NEXT: stablehlo.slice
  // CHECK-NEXT: stablehlo.reshape
  // Sum halfnormal + MVN logpdf contributions
  // CHECK-NEXT: arith.addf
  // CHECK-NEXT: enzyme.yield
  // CHECK: num_position_args = 1

  func.func @test_scale_family_mvn(%rng : tensor<2xui64>,
                                    %R_data : tensor<3x3xf64>,
                                    %trace : tensor<1x4xf64>)
      -> (tensor<1x4xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:8 = enzyme.mcmc @model_scale_family_mvn(%rng, %R_data) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>]],
      all_addresses = [[#enzyme.symbol<1>, #enzyme.symbol<2>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x4xf64>, tensor<f64>)
        -> (tensor<1x4xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x4xf64>, tensor<1x4xf64>, tensor<f64>, tensor<f64>, tensor<1x4xf64>)

    return %result#0, %result#1, %result#2 : tensor<1x4xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // Model body: sample tau, compute K = tau^2 * R_data, observe y ~ MVN(0, K)
  func.func @model_scale_family_mvn(%rng : tensor<2xui64>,
                                     %R_data : tensor<3x3xf64>)
      -> (tensor<2xui64>, tensor<3xf64>) {
    %std = arith.constant dense<1.0> : tensor<f64>

    // Sample tau ~ HalfNormal(1)
    %tau:2 = enzyme.sample @halfnormal_sampler(%rng, %std) {
      logpdf = @halfnormal_logpdf,
      symbol = #enzyme.symbol<1>,
      support = #enzyme.support<POSITIVE>
    } : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    // K = tau^2 * R_data  (scale * invariant, visible in model body)
    %tau_sq = stablehlo.multiply %tau#1, %tau#1 : tensor<f64>
    %tau_sq_mat = stablehlo.broadcast_in_dim %tau_sq, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %K = stablehlo.multiply %tau_sq_mat, %R_data : tensor<3x3xf64>

    // y ~ MVN(0, K) — Cholesky is INSIDE mvn_logpdf, not here
    %y:2 = enzyme.sample @mvn_sampler(%tau#0, %K) {
      logpdf = @mvn_logpdf,
      symbol = #enzyme.symbol<2>
    } : (tensor<2xui64>, tensor<3x3xf64>) -> (tensor<2xui64>, tensor<3xf64>)

    return %y#0, %y#1 : tensor<2xui64>, tensor<3xf64>
  }

  // ===== Test: Same model but with NCP (Cholesky in model body) =====
  //
  // Alternative formulation: compute Cholesky in the model body,
  // then use it for observation logpdf. SICM CAN hoist this.

  // CHECK-LABEL: func.func @test_scale_family_mvn_ncp
  // Cholesky on R_data hoisted before mcmc_region
  // CHECK: %[[CHOL:.+]] = stablehlo.cholesky %{{.*}}, lower = true
  // CHECK: enzyme.mcmc_region
  // CHECK: enzyme.sample_region
  // Decomposed: sqrt(tau^2) * hoisted chol(R)
  // CHECK: %[[TAU_SQ:.+]] = stablehlo.multiply %{{.*}}, %{{.*}} : tensor<f64>
  // CHECK-NEXT: %[[SQRT:.+]] = math.sqrt %[[TAU_SQ]] : tensor<f64>
  // CHECK-NEXT: %[[SQRT_MAT:.+]] = stablehlo.broadcast_in_dim %[[SQRT]], dims = [] : (tensor<f64>) -> tensor<3x3xf64>
  // CHECK-NEXT: %[[SCALED_L:.+]] = stablehlo.multiply %[[SQRT_MAT]], %[[CHOL]] : tensor<3x3xf64>
  // CHECK-NEXT: stablehlo.slice %[[SCALED_L]] [0:1, 0:1]
  // CHECK-NEXT: stablehlo.reshape
  // Logpdf: just halfnormal negation (single position arg)
  // CHECK: ^bb0(%[[T:.+]]: tensor<f64>):
  // CHECK-NEXT: arith.negf %[[T]]
  // CHECK-NEXT: enzyme.yield
  // No cholesky inside mcmc_region
  // CHECK-NOT: stablehlo.cholesky
  // CHECK: num_position_args = 1

  func.func @test_scale_family_mvn_ncp(%rng : tensor<2xui64>,
                                        %R_data : tensor<3x3xf64>,
                                        %trace : tensor<1x1xf64>)
      -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:8 = enzyme.mcmc @model_scale_family_mvn_ncp(%rng, %R_data) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>]],
      all_addresses = [[#enzyme.symbol<1>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x1xf64>, tensor<f64>)
        -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)

    return %result#0, %result#1, %result#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // NCP model: Cholesky is done directly in the model body
  func.func @model_scale_family_mvn_ncp(%rng : tensor<2xui64>,
                                         %R_data : tensor<3x3xf64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<f64>

    // Sample tau ~ HalfNormal(1)
    %tau:2 = enzyme.sample @halfnormal_sampler(%rng, %std) {
      logpdf = @halfnormal_logpdf,
      symbol = #enzyme.symbol<1>,
      support = #enzyme.support<POSITIVE>
    } : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    // Cholesky directly in model body: cholesky(tau^2 * R_data)
    %tau_sq = stablehlo.multiply %tau#1, %tau#1 : tensor<f64>
    %tau_sq_mat = stablehlo.broadcast_in_dim %tau_sq, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %scaled_cov = stablehlo.multiply %tau_sq_mat, %R_data : tensor<3x3xf64>
    %L = stablehlo.cholesky %scaled_cov, lower = true : tensor<3x3xf64>

    // Extract [0,0] element as result (simplified)
    %sliced = stablehlo.slice %L [0:1, 0:1] : (tensor<3x3xf64>) -> tensor<1x1xf64>
    %result = stablehlo.reshape %sliced : (tensor<1x1xf64>) -> tensor<f64>

    return %tau#0, %result : tensor<2xui64>, tensor<f64>
  }

  // ===== Test: Submodel with Cholesky (full inlining) =====
  //
  // Model:
  //   result = submodel(rng, R_data)
  //     where submodel internally does:
  //       L = cholesky(R_data)     <- expensive, depends only on R_data (invariant)
  //       z ~ Normal(0, 1)         <- sample
  //       y = L[0,0] * z           <- uses both L and sample
  //       return rng, y
  //
  // After inline-mcmc-regions + SICM Phase 0 (full submodel inlining):
  //   1. Submodel boundary dissolved — all ops moved into mcmc_region body
  //   2. Inner sample gets composite symbol: symbol<1, 2> (outer=1, inner=2)
  //   3. Addresses flattened: [[<1>, <2>]] -> [[<1, 2>]]
  //   4. Phase 2: cholesky hoisted before mcmc_region (only depends on block arg)

  // CHECK-LABEL: func.func @test_submodel_cholesky
  // Cholesky and derived ops hoisted before mcmc_region
  // CHECK: %[[CHOL:.+]] = stablehlo.cholesky %{{.*}}, lower = true
  // CHECK-NEXT: %[[SLICE:.+]] = stablehlo.slice %[[CHOL]] [0:1, 0:1]
  // CHECK-NEXT: %[[SCALAR:.+]] = stablehlo.reshape %[[SLICE]]
  // CHECK: enzyme.mcmc_region
  // Inner sample with composite symbol (submodel dissolved)
  // CHECK: enzyme.sample @halfnormal_sampler
  // CHECK-SAME: symbol = #enzyme.symbol<1, 2>
  // Multiply uses hoisted scalar
  // CHECK: stablehlo.multiply %[[SCALAR]], %{{.*}} : tensor<f64>
  // CHECK-NEXT: enzyme.yield
  // Logpdf: constant zero (halfnormal has no logpdf contribution here)
  // CHECK: logpdf
  // CHECK: enzyme.yield
  // Addresses flattened to composite symbols
  // CHECK: all_addresses = {{\[}}[#enzyme.symbol<1, 2>]]
  // CHECK-SAME: selection = {{\[}}[#enzyme.symbol<1, 2>]]

  func.func @test_submodel_cholesky(%rng : tensor<2xui64>,
                                     %R_data : tensor<3x3xf64>,
                                     %trace : tensor<1x1xf64>)
      -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:8 = enzyme.mcmc @model_submodel_chol(%rng, %R_data) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>, #enzyme.symbol<2>]],
      all_addresses = [[#enzyme.symbol<1>, #enzyme.symbol<2>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x1xf64>, tensor<f64>)
        -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)

    return %result#0, %result#1, %result#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // Outer model: just calls the submodel
  func.func @model_submodel_chol(%rng : tensor<2xui64>,
                                  %R_data : tensor<3x3xf64>)
      -> (tensor<2xui64>, tensor<f64>) {
    // Call submodel (no logpdf — this is a submodel call)
    %sub:2 = enzyme.sample @chol_submodel(%rng, %R_data) {
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<3x3xf64>) -> (tensor<2xui64>, tensor<f64>)

    return %sub#0, %sub#1 : tensor<2xui64>, tensor<f64>
  }

  // Submodel: Cholesky on invariant input + sample
  func.func @chol_submodel(%rng : tensor<2xui64>,
                            %cov : tensor<3x3xf64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<f64>

    // L = cholesky(cov) — expensive, invariant w.r.t. inner sample
    %L = stablehlo.cholesky %cov, lower = true : tensor<3x3xf64>

    // z ~ Normal(0, 1)
    %z:2 = enzyme.sample @halfnormal_sampler(%rng, %std) {
      logpdf = @halfnormal_logpdf,
      symbol = #enzyme.symbol<2>,
      support = #enzyme.support<POSITIVE>
    } : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    // y = L[0,0] * z (simplified usage of L with sample value)
    %L_slice = stablehlo.slice %L [0:1, 0:1] : (tensor<3x3xf64>) -> tensor<1x1xf64>
    %L_scalar = stablehlo.reshape %L_slice : (tensor<1x1xf64>) -> tensor<f64>
    %y = stablehlo.multiply %L_scalar, %z#1 : tensor<f64>

    return %z#0, %y : tensor<2xui64>, tensor<f64>
  }
}
