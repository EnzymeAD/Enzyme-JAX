// RUN: enzymexlamlir-opt --sicm %s | FileCheck %s
//
// Real-world IR pattern from scale_family_mvn benchmark (post inline-mcmc-regions).
//
// Model:
//   tau ~ HalfNormal(1)
//   K = tau^2 * R_data
//   y ~ MVN(0, K)      (Cholesky inside sample_region sampler + logpdf bodies)
//
// SICM hoists cholesky(R), diag(L), log(diag(L)) before mcmc_region.
// The unified logpdf uses tri_solve with hoisted chol(R), scaled by 1/sqrt(tau).
// Net effect: O(N^3) Cholesky computed once, O(1) scalar ops per NUTS iteration.

// CHECK-LABEL: func.func @scale_family_mvn
// Hoisted before mcmc_region: cholesky(R), diag(L), log(diag(L))
// CHECK: %[[CHOL:.+]] = stablehlo.cholesky
// CHECK-NEXT: %[[DIAG:.+]] = stablehlo.dot_general %[[CHOL]]
// CHECK-NEXT: %[[LOG_DIAG:.+]] = stablehlo.log %[[DIAG]]
// CHECK: enzyme.mcmc_region
// CHECK: enzyme.sample_region
// K = tau^2 * R_data
// CHECK: stablehlo.multiply %{{.*}} : tensor<3x3xf64>
// CHECK: enzyme.sample_region
// Unified logpdf: factored solve using hoisted chol(R)
// CHECK: ^bb0(%{{.*}}: tensor<1xf64>, %{{.*}}: tensor<3xf64>):
// sqrt(tau^2) for scaling
// CHECK: math.sqrt
// tri_solve with hoisted chol(R), scaled by 1/sqrt(tau)
// CHECK: stablehlo.triangular_solve
// CHECK: stablehlo.divide
// CHECK: stablehlo.triangular_solve
// log-det: log(sqrt(tau)) + hoisted log(diag(L))
// CHECK: stablehlo.add %{{.*}}, %[[LOG_DIAG]]
// No cholesky in logpdf body
// CHECK-NOT: stablehlo.cholesky
// CHECK: enzyme.yield
// CHECK: num_position_args = 1

module {
  func.func @scale_family_mvn(
      %rng : tensor<2xui64>,
      %R_data : tensor<3x3xf64>,
      %trace : tensor<1x4xf64>,
      %step_size : tensor<f64>,
      %inv_mass : tensor<1x1xf64>)
      -> (tensor<1x1xf64>, tensor<i1>, tensor<2xui64>) {

    %cst = arith.constant dense<0.000000e+00> : tensor<f64>
    %cst_0 = arith.constant dense<-5.000000e-01> : tensor<f64>
    %cst_1 = arith.constant dense<2.756815599614018> : tensor<f64>
    %cst_2 = arith.constant dense<[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]> : tensor<3x3xf64>

    %new_trace, %accepted, %output_rng,
    %final_pos, %final_grad, %final_U, %final_ss, %final_im
      = enzyme.mcmc_region(%rng, %R_data) given %trace
          inverse_mass_matrix = %inv_mass step_size = %step_size {
    ^bb0(%arg5: tensor<2xui64>, %arg6: tensor<3x3xf64>):

      // === Sample tau ~ HalfNormal(1) ===
      %tau:2 = enzyme.sample_region(%arg5) sampler {
      ^bb0(%srng: tensor<2xui64>):
        %output_state, %output = stablehlo.rng_bit_generator %srng, algorithm = DEFAULT
            : (tensor<2xui64>) -> (tensor<2xui64>, tensor<1xui64>)
        %c_srl = arith.constant dense<11> : tensor<1xui64>
        %c_or  = arith.constant dense<4607182418800017408> : tensor<1xui64>
        %c_one = arith.constant dense<1.0> : tensor<1xf64>
        %c_two = arith.constant dense<2.0> : tensor<1xf64>
        %c_neg = arith.constant dense<-1.0> : tensor<1xf64>
        %c_lo  = arith.constant dense<-0.9999999> : tensor<1xf64>
        %c_hi  = arith.constant dense<0.9999999> : tensor<1xf64>
        %c_sqrt2 = arith.constant dense<1.4142135623730951> : tensor<1xf64>
        %s1 = stablehlo.shift_right_logical %output, %c_srl : tensor<1xui64>
        %s2 = stablehlo.or %s1, %c_or : tensor<1xui64>
        %s3 = stablehlo.bitcast_convert %s2 : (tensor<1xui64>) -> tensor<1xf64>
        %s4 = stablehlo.subtract %s3, %c_one : tensor<1xf64>
        %s5 = stablehlo.multiply %s4, %c_two : tensor<1xf64>
        %s6 = stablehlo.add %s5, %c_neg : tensor<1xf64>
        %s7 = stablehlo.clamp %c_lo, %s6, %c_hi : tensor<1xf64>
        %s8 = chlo.erf_inv %s7 : tensor<1xf64> -> tensor<1xf64>
        %s9 = stablehlo.multiply %s8, %c_sqrt2 : tensor<1xf64>
        %s10 = stablehlo.abs %s9 : tensor<1xf64>
        enzyme.yield %output_state, %s10 : tensor<2xui64>, tensor<1xf64>
      } logpdf {
      ^bb0(%lp_x: tensor<1xf64>):
        %xr = stablehlo.reshape %lp_x : (tensor<1xf64>) -> tensor<f64>
        %xsq = stablehlo.multiply %xr, %xr : tensor<f64>
        %neg = stablehlo.multiply %xsq, %cst_0 : tensor<f64>
        enzyme.yield %neg : tensor<f64>
      } {symbol = #enzyme.symbol<1>, support = #enzyme.support<POSITIVE>}
        : (tensor<2xui64>) -> (tensor<2xui64>, tensor<1xf64>)

      // === K = tau^2 * R_data ===
      %tau_mat = stablehlo.reshape %tau#1 : (tensor<1xf64>) -> tensor<1x1xf64>
      %tau_sq = stablehlo.multiply %tau_mat, %tau_mat : tensor<1x1xf64>
      %tau_sq_bc = stablehlo.broadcast_in_dim %tau_sq, dims = [0, 1]
          : (tensor<1x1xf64>) -> tensor<3x3xf64>
      %K = stablehlo.multiply %tau_sq_bc, %arg6 : tensor<3x3xf64>

      // === y ~ MVN(0, K) — Cholesky inside sampler + logpdf bodies ===
      %y:2 = enzyme.sample_region(%tau#0, %K) sampler {
      ^bb0(%srng2: tensor<2xui64>, %scov: tensor<3x3xf64>):
        %output_state2, %output2 = stablehlo.rng_bit_generator %srng2, algorithm = DEFAULT
            : (tensor<2xui64>) -> (tensor<2xui64>, tensor<3xui64>)
        %c_srl2 = arith.constant dense<11> : tensor<3xui64>
        %c_or2  = arith.constant dense<4607182418800017408> : tensor<3xui64>
        %c_one2 = arith.constant dense<1.0> : tensor<3xf64>
        %c_two2 = arith.constant dense<2.0> : tensor<3xf64>
        %c_neg2 = arith.constant dense<-1.0> : tensor<3xf64>
        %c_lo2  = arith.constant dense<-0.9999999> : tensor<3xf64>
        %c_hi2  = arith.constant dense<0.9999999> : tensor<3xf64>
        %c_sqrt2b = arith.constant dense<1.4142135623730951> : tensor<3xf64>
        %u1 = stablehlo.shift_right_logical %output2, %c_srl2 : tensor<3xui64>
        %u2 = stablehlo.or %u1, %c_or2 : tensor<3xui64>
        %u3 = stablehlo.bitcast_convert %u2 : (tensor<3xui64>) -> tensor<3xf64>
        %u4 = stablehlo.subtract %u3, %c_one2 : tensor<3xf64>
        %u5 = stablehlo.multiply %u4, %c_two2 : tensor<3xf64>
        %u6 = stablehlo.add %u5, %c_neg2 : tensor<3xf64>
        %u7 = stablehlo.clamp %c_lo2, %u6, %c_hi2 : tensor<3xf64>
        %u8 = chlo.erf_inv %u7 : tensor<3xf64> -> tensor<3xf64>
        // Cholesky in sampler body (not hoisted by SICM — only logpdf is hoisted)
        %L_sampler = stablehlo.cholesky %scov : tensor<3x3xf64>
        %z = stablehlo.dot_general %L_sampler, %u8,
            contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT]
            : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
        %scaled = stablehlo.multiply %c_sqrt2b, %z : tensor<3xf64>
        enzyme.yield %output_state2, %scaled : tensor<2xui64>, tensor<3xf64>
      } logpdf {
      ^bb0(%lp_obs: tensor<3xf64>, %lp_cov: tensor<3x3xf64>):
        // Cholesky in logpdf body — THIS is what SICM Phase 0 hoists
        %L = stablehlo.cholesky %lp_cov : tensor<3x3xf64>
        %obs_col = stablehlo.reshape %lp_obs : (tensor<3xf64>) -> tensor<3x1xf64>
        %s1b = "stablehlo.triangular_solve"(%L, %obs_col) <{
            left_side = true, lower = false,
            unit_diagonal = false,
            transpose_a = #stablehlo<transpose ADJOINT>
        }> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
        %s2b = "stablehlo.triangular_solve"(%L, %s1b) <{
            left_side = true, lower = false,
            unit_diagonal = false,
            transpose_a = #stablehlo<transpose NO_TRANSPOSE>
        }> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
        // diag(L) via dot with identity for log-det
        %diag_L = stablehlo.dot_general %L, %cst_2,
            batching_dims = [1] x [1], contracting_dims = [0] x [0]
            : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3xf64>
        %log_diag = stablehlo.log %diag_L : tensor<3xf64>
        %sum_log = stablehlo.reduce(%log_diag init: %cst)
            applies stablehlo.add across dimensions = [0]
            : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
        %half_log_det = stablehlo.multiply %cst_0, %sum_log : tensor<f64>
        %solved_flat = stablehlo.reshape %s2b : (tensor<3x1xf64>) -> tensor<3xf64>
        %quad = stablehlo.dot_general %lp_obs, %solved_flat,
            contracting_dims = [0] x [0]
            : (tensor<3xf64>, tensor<3xf64>) -> tensor<f64>
        %sum1 = stablehlo.add %quad, %half_log_det : tensor<f64>
        %sum2 = stablehlo.add %sum1, %cst_1 : tensor<f64>
        %result = stablehlo.multiply %cst_0, %sum2 : tensor<f64>
        enzyme.yield %result : tensor<f64>
      } {symbol = #enzyme.symbol<2>, support = #enzyme.support<REAL>}
        : (tensor<2xui64>, tensor<3x3xf64>) -> (tensor<2xui64>, tensor<3xf64>)

      enzyme.yield %y#0, %y#1 : tensor<2xui64>, tensor<3xf64>
    } attributes {
      all_addresses = [[#enzyme.symbol<1>], [#enzyme.symbol<2>]],
      fn = "model_scale_family_mvn",
      nuts_config = #enzyme.nuts_config<,
          max_delta_energy = 1.000000e+03 : f64,
          adapt_step_size = false, adapt_mass_matrix = false>,
      selection = [[#enzyme.symbol<1>]]
    } : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x4xf64>,
         tensor<1x1xf64>, tensor<f64>)
      -> (tensor<1x1xf64>, tensor<i1>, tensor<2xui64>,
          tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>,
          tensor<f64>, tensor<1x1xf64>)

    return %new_trace, %accepted, %output_rng
        : tensor<1x1xf64>, tensor<i1>, tensor<2xui64>
  }
}
