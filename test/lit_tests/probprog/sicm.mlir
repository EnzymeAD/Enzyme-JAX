// RUN: enzymexlamlir-opt --inline-mcmc-regions --sicm %s | FileCheck %s

// ============================================================================
// Test: Hierarchical model with Cholesky scale factorization + SICM hoisting
// ============================================================================
//
// Hierarchical model:
//   scale ~ Normal(prior_scale, 1.0)                  <- sample site
//   scaled_cov = broadcast(scale) * base_cov          <- element-wise scale
//   L = cholesky(scaled_cov)                          <- expensive, sample-dependent
//   result = L[0,0]                                   <- use L
//
// Without SICM: cholesky(scale * base_cov) is sample-dependent, stays inside loop.
//
// With --sicm (fixpoint rewrite + hoist):
//   Phase 1 (rewrite): cholesky(scale * base_cov) -> sqrt(scale) * cholesky(base_cov)
//   Phase 2 (hoist):   cholesky(base_cov) is sample-invariant -> hoist before mcmc_region
//
// Result: expensive cholesky is computed once outside the MCMC loop.

// CHECK-LABEL: func.func @test_cholesky_hierarchical
// CHECK-SAME: %[[RNG:.*]]: tensor<2xui64>, %[[PRIOR:.*]]: tensor<f64>, %[[COV:.*]]: tensor<2x2xf64>, %[[TRACE:.*]]: tensor<1x1xf64>
// Cholesky on base_cov is hoisted BEFORE mcmc_region (exposed by rewrite)
// CHECK-DAG: %[[ONE:.*]] = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK-DAG: %[[IDX:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[STEP:.*]] = arith.constant dense<1.000000e-01> : tensor<f64>
// CHECK: %[[CHOL:.*]] = stablehlo.cholesky %[[COV]], lower = true : tensor<2x2xf64>
// No more cholesky before or inside mcmc_region
// CHECK-NOT: stablehlo.cholesky
// CHECK: enzyme.mcmc_region(%[[RNG]], %[[PRIOR]], %[[COV]]) given %[[TRACE]] step_size = %[[STEP]]
// Sample the scale parameter
// CHECK: enzyme.sample_region(%{{.*}}, %{{.*}}, %[[ONE]])
// CHECK: symbol = #enzyme.symbol<1>
// After sampling: sqrt(scale), broadcast, multiply with hoisted cholesky
// CHECK-NEXT: %[[SQRT:.*]] = math.sqrt %{{.*}} : tensor<f64>
// CHECK-NEXT: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[SQRT]], dims = [] : (tensor<f64>) -> tensor<2x2xf64>
// CHECK-NEXT: %[[SCALED:.*]] = stablehlo.multiply %[[BCAST]], %[[CHOL]] : tensor<2x2xf64>
// Extract result from the scaled cholesky factor
// CHECK-NEXT: %[[DIAG:.*]] = tensor.extract %[[SCALED]][%[[IDX]], %[[IDX]]] : tensor<2x2xf64>
// CHECK-NEXT: %[[RES:.*]] = tensor.from_elements %[[DIAG]] : tensor<f64>
// CHECK-NEXT: enzyme.yield %{{.*}}, %[[RES]] : tensor<2xui64>, tensor<f64>
// Verify no cholesky anywhere after mcmc_region
// CHECK-NOT: stablehlo.cholesky
// CHECK: return

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

  // ===== Test 1: Cholesky scale factorization (existing) =====

  func.func @test_cholesky_hierarchical(%rng : tensor<2xui64>, %prior_scale : tensor<f64>,
                                         %base_cov : tensor<2x2xf64>, %trace : tensor<1x1xf64>)
      -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:3 = enzyme.mcmc @model_cholesky_hier(%rng, %prior_scale, %base_cov) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>]],
      all_addresses = [[#enzyme.symbol<1>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<2x2xf64>, tensor<1x1xf64>, tensor<f64>)
        -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>)

    return %result#0, %result#1, %result#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  func.func @model_cholesky_hier(%rng : tensor<2xui64>, %prior_scale : tensor<f64>,
                                  %base_cov : tensor<2x2xf64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<f64>

    %scale:2 = enzyme.sample @normal(%rng, %prior_scale, %std) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    %scale_mat = stablehlo.broadcast_in_dim %scale#1, dims = [] : (tensor<f64>) -> tensor<2x2xf64>
    %scaled_cov = stablehlo.multiply %scale_mat, %base_cov : tensor<2x2xf64>

    %L = stablehlo.cholesky %scaled_cov, lower = true : tensor<2x2xf64>

    %c0 = arith.constant 0 : index
    %diag = tensor.extract %L[%c0, %c0] : tensor<2x2xf64>
    %result = tensor.from_elements %diag : tensor<f64>

    return %scale#0, %result : tensor<2xui64>, tensor<f64>
  }

  // ===== Test 2: Dot general linearity =====
  // Model: sample scale, compute (scale * W) @ x where W, x are data
  // SICM rewrites: dot_general(broadcast(scale) * W, x) -> broadcast(scale) * dot_general(W, x)
  // Then hoists dot_general(W, x) outside mcmc_region.

  // CHECK-LABEL: func.func @test_dot_general_linearity
  // CHECK-SAME: %[[RNG:.*]]: tensor<2xui64>, %[[PRIOR:.*]]: tensor<f64>, %[[W:.*]]: tensor<3x2xf64>, %[[X:.*]]: tensor<2x1xf64>, %[[TRACE:.*]]: tensor<1x1xf64>
  // dot_general(W, x) is hoisted BEFORE mcmc_region
  // CHECK: %[[DOT:.*]] = stablehlo.dot_general %[[W]], %[[X]], contracting_dims = [1] x [0] : (tensor<3x2xf64>, tensor<2x1xf64>) -> tensor<3x1xf64>
  // CHECK-NOT: stablehlo.dot_general
  // CHECK: enzyme.mcmc_region
  // Inside the region: only scalar broadcast + multiply remains
  // CHECK: enzyme.sample_region
  // CHECK: stablehlo.broadcast_in_dim
  // CHECK-NEXT: %[[SCALED_DOT:.*]] = stablehlo.multiply %{{.*}}, %[[DOT]] : tensor<3x1xf64>
  // No dot_general inside mcmc_region
  // CHECK-NOT: stablehlo.dot_general
  // CHECK: enzyme.yield

  func.func @test_dot_general_linearity(%rng : tensor<2xui64>, %prior : tensor<f64>,
                                         %W : tensor<3x2xf64>, %x : tensor<2x1xf64>,
                                         %trace : tensor<1x1xf64>)
      -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:3 = enzyme.mcmc @model_dot_general(%rng, %prior, %W, %x) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>]],
      all_addresses = [[#enzyme.symbol<1>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<3x2xf64>, tensor<2x1xf64>, tensor<1x1xf64>, tensor<f64>)
        -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>)

    return %result#0, %result#1, %result#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // Model: sample scale, then (broadcast(scale) * W) @ x
  func.func @model_dot_general(%rng : tensor<2xui64>, %prior : tensor<f64>,
                                %W : tensor<3x2xf64>, %x : tensor<2x1xf64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<f64>

    %scale:2 = enzyme.sample @normal(%rng, %prior, %std) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    // scale * W (broadcast scalar to 3x2 matrix)
    %scale_mat = stablehlo.broadcast_in_dim %scale#1, dims = [] : (tensor<f64>) -> tensor<3x2xf64>
    %scaled_W = stablehlo.multiply %scale_mat, %W : tensor<3x2xf64>

    // (scale * W) @ x — expensive matmul with sample-dependent operand
    %y = stablehlo.dot_general %scaled_W, %x, contracting_dims = [1] x [0]
        : (tensor<3x2xf64>, tensor<2x1xf64>) -> tensor<3x1xf64>

    // Use first element as result
    %c0 = arith.constant 0 : index
    %elem = tensor.extract %y[%c0, %c0] : tensor<3x1xf64>
    %result = tensor.from_elements %elem : tensor<f64>

    return %scale#0, %result : tensor<2xui64>, tensor<f64>
  }

  // ===== Test 3: Triangular solve scale factorization =====
  // Model: sample scale, compute triangular_solve(broadcast(scale) * L, b) where L, b are data
  // SICM rewrites: triangular_solve(s*L, b) -> (1/s) * triangular_solve(L, b)
  // Then hoists triangular_solve(L, b) outside mcmc_region.

  // CHECK-LABEL: func.func @test_triangular_solve_scale
  // CHECK-SAME: %[[RNG:.*]]: tensor<2xui64>, %[[PRIOR:.*]]: tensor<f64>, %[[L:.*]]: tensor<3x3xf64>, %[[B:.*]]: tensor<3x1xf64>, %[[TRACE:.*]]: tensor<1x1xf64>
  // triangular_solve(L, b) is hoisted BEFORE mcmc_region
  // CHECK: %[[SOLVE:.*]] = "stablehlo.triangular_solve"(%[[L]], %[[B]]) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
  // CHECK-NOT: stablehlo.triangular_solve
  // CHECK: enzyme.mcmc_region
  // Inside the region: only (1/scale) * hoisted_result
  // CHECK: enzyme.sample_region
  // CHECK: stablehlo.divide
  // CHECK: stablehlo.broadcast_in_dim
  // CHECK-NEXT: %[[SCALED_SOLVE:.*]] = stablehlo.multiply %{{.*}}, %[[SOLVE]] : tensor<3x1xf64>
  // CHECK-NOT: stablehlo.triangular_solve
  // CHECK: enzyme.yield

  func.func @test_triangular_solve_scale(%rng : tensor<2xui64>, %prior : tensor<f64>,
                                          %L : tensor<3x3xf64>, %b : tensor<3x1xf64>,
                                          %trace : tensor<1x1xf64>)
      -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:3 = enzyme.mcmc @model_triangular_solve(%rng, %prior, %L, %b) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>]],
      all_addresses = [[#enzyme.symbol<1>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<3x3xf64>, tensor<3x1xf64>, tensor<1x1xf64>, tensor<f64>)
        -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>)

    return %result#0, %result#1, %result#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // Model: sample scale, then triangular_solve(broadcast(scale) * L, b)
  func.func @model_triangular_solve(%rng : tensor<2xui64>, %prior : tensor<f64>,
                                     %L : tensor<3x3xf64>, %b : tensor<3x1xf64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<f64>

    %scale:2 = enzyme.sample @normal(%rng, %prior, %std) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    // scale * L (broadcast scalar to 3x3 matrix)
    %scale_mat = stablehlo.broadcast_in_dim %scale#1, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %scaled_L = stablehlo.multiply %scale_mat, %L : tensor<3x3xf64>

    // triangular_solve(scale * L, b) — expensive with sample-dependent L
    %x = "stablehlo.triangular_solve"(%scaled_L, %b) <{
      left_side = true, lower = true,
      unit_diagonal = false,
      transpose_a = #stablehlo<transpose NO_TRANSPOSE>
    }> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>

    // Use first element as result
    %c0 = arith.constant 0 : index
    %elem = tensor.extract %x[%c0, %c0] : tensor<3x1xf64>
    %result = tensor.from_elements %elem : tensor<f64>

    return %scale#0, %result : tensor<2xui64>, tensor<f64>
  }

  // ===== Test 4: Log-multiply distribution =====
  // Model: sample scale, compute log(broadcast(scale) * diag_values) where diag_values is data
  // SICM rewrites: log(s * A) -> log(s) + log(A)
  // Then hoists log(A) outside mcmc_region.

  // CHECK-LABEL: func.func @test_log_multiply_dist
  // CHECK-SAME: %[[RNG:.*]]: tensor<2xui64>, %[[PRIOR:.*]]: tensor<f64>, %[[DIAG:.*]]: tensor<3xf64>, %[[TRACE:.*]]: tensor<1x1xf64>
  // log(diag_values) is hoisted BEFORE mcmc_region
  // CHECK: %[[LOG_DIAG:.*]] = stablehlo.log %[[DIAG]] : tensor<3xf64>
  // CHECK-NOT: stablehlo.log %[[DIAG]]
  // CHECK: enzyme.mcmc_region
  // Inside the region: log(scale) + hoisted log(diag)
  // CHECK: enzyme.sample_region
  // CHECK: stablehlo.log %{{.*}} : tensor<f64>
  // CHECK: stablehlo.broadcast_in_dim
  // CHECK-NEXT: %[[SUM:.*]] = stablehlo.add %{{.*}}, %[[LOG_DIAG]] : tensor<3xf64>
  // CHECK-NOT: stablehlo.log %[[DIAG]]
  // CHECK: enzyme.yield

  func.func @test_log_multiply_dist(%rng : tensor<2xui64>, %prior : tensor<f64>,
                                     %diag_values : tensor<3xf64>,
                                     %trace : tensor<1x1xf64>)
      -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:3 = enzyme.mcmc @model_log_multiply(%rng, %prior, %diag_values) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>]],
      all_addresses = [[#enzyme.symbol<1>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<3xf64>, tensor<1x1xf64>, tensor<f64>)
        -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>)

    return %result#0, %result#1, %result#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // Model: sample scale, then log(broadcast(scale) * diag_values)
  func.func @model_log_multiply(%rng : tensor<2xui64>, %prior : tensor<f64>,
                                 %diag_values : tensor<3xf64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<f64>

    %scale:2 = enzyme.sample @normal(%rng, %prior, %std) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    // scale * diag_values (broadcast scalar to vector)
    %scale_vec = stablehlo.broadcast_in_dim %scale#1, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %scaled_diag = stablehlo.multiply %scale_vec, %diag_values : tensor<3xf64>

    // log(scale * diag_values) — would be expensive if diag_values is large
    %log_result = stablehlo.log %scaled_diag : tensor<3xf64>

    // Use first element as result
    %c0 = arith.constant 0 : index
    %elem = tensor.extract %log_result[%c0] : tensor<3xf64>
    %result = tensor.from_elements %elem : tensor<f64>

    return %scale#0, %result : tensor<2xui64>, tensor<f64>
  }

  // ===== Test 5: Cholesky + Log composition chain (2 fixpoint iterations) =====
  // Model: sample scale, compute log(cholesky(scale * A)) where A is data (2x2 matrix)
  // This requires 2 fixpoint iterations of SICM:
  //   Iter 1 rewrite: cholesky(s*A) -> sqrt(s)*cholesky(A)
  //   Iter 1 hoist:   cholesky(A) hoisted
  //   Iter 2 rewrite: log(sqrt(s)*chol(A)) -> log(sqrt(s)) + log(chol(A))
  //   Iter 2 hoist:   log(chol(A)) hoisted (chol(A) is now a region arg, so log(chol(A)) is invariant)

  // CHECK-LABEL: func.func @test_cholesky_log_composition
  // CHECK-SAME: %[[RNG:.*]]: tensor<2xui64>, %[[PRIOR:.*]]: tensor<f64>, %[[COV:.*]]: tensor<2x2xf64>, %[[TRACE:.*]]: tensor<1x1xf64>
  // Both cholesky(COV) and log(cholesky(COV)) should be hoisted
  // CHECK: %[[CHOL:.*]] = stablehlo.cholesky %[[COV]], lower = true : tensor<2x2xf64>
  // CHECK: %[[LOG_CHOL:.*]] = stablehlo.log %[[CHOL]] : tensor<2x2xf64>
  // CHECK-NOT: stablehlo.cholesky
  // CHECK: enzyme.mcmc_region
  // Inside: sqrt(scale), log(sqrt(scale)), broadcast, add — no cholesky or log of invariant
  // CHECK: enzyme.sample_region
  // CHECK: math.sqrt
  // CHECK: stablehlo.log
  // CHECK: stablehlo.broadcast_in_dim
  // CHECK-NEXT: stablehlo.add %{{.*}}, %[[LOG_CHOL]] : tensor<2x2xf64>
  // CHECK-NOT: stablehlo.cholesky
  // CHECK: enzyme.yield

  func.func @test_cholesky_log_composition(%rng : tensor<2xui64>, %prior : tensor<f64>,
                                            %base_cov : tensor<2x2xf64>,
                                            %trace : tensor<1x1xf64>)
      -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:3 = enzyme.mcmc @model_cholesky_log(%rng, %prior, %base_cov) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>]],
      all_addresses = [[#enzyme.symbol<1>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<2x2xf64>, tensor<1x1xf64>, tensor<f64>)
        -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>)

    return %result#0, %result#1, %result#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // Model: sample scale, then log(cholesky(scale * A))
  // After cholesky rewrite: log(sqrt(s) * cholesky(A))
  // After log rewrite: log(sqrt(s)) + log(cholesky(A))
  func.func @model_cholesky_log(%rng : tensor<2xui64>, %prior : tensor<f64>,
                                 %base_cov : tensor<2x2xf64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<f64>

    %scale:2 = enzyme.sample @normal(%rng, %prior, %std) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    // scale * base_cov
    %scale_mat = stablehlo.broadcast_in_dim %scale#1, dims = [] : (tensor<f64>) -> tensor<2x2xf64>
    %scaled_cov = stablehlo.multiply %scale_mat, %base_cov : tensor<2x2xf64>

    // cholesky(scale * base_cov)
    %L = stablehlo.cholesky %scaled_cov, lower = true : tensor<2x2xf64>

    // log(L) — applied element-wise to the whole matrix
    // After cholesky rewrite, L = sqrt(s) * cholesky(A), so
    // log(L) = log(sqrt(s) * cholesky(A)) which the log pattern decomposes
    %log_L = stablehlo.log %L : tensor<2x2xf64>

    // Use [0,0] element as result
    %c0 = arith.constant 0 : index
    %elem = tensor.extract %log_L[%c0, %c0] : tensor<2x2xf64>
    %result = tensor.from_elements %elem : tensor<f64>

    return %scale#0, %result : tensor<2xui64>, tensor<f64>
  }
}
