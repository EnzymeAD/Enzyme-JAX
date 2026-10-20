// RUN: enzymexlamlir-opt %s --sicm | FileCheck %s
// RUN: enzymexlamlir-opt %s --sicm="schedule=%s" | FileCheck %s
// RUN: enzymexlamlir-opt %s --sicm="schedule=%s schedule-entry-point=builtin_rules_without_cholesky_scale" | FileCheck %s --check-prefix=NORULE
// RUN: enzymexlamlir-opt %s --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s --check-prefix=INTERP

// The PDL patterns in the schedule are generated from user_rule_pdl.pdll.

module attributes {transform.with_named_sequence} {
  func.func private @halfnormal_sampler(%rng : tensor<2xui64>, %std : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %sample = stablehlo.abs %std : tensor<f64>
    return %rng, %sample : tensor<2xui64>, tensor<f64>
  }
  func.func private @halfnormal_logpdf(%x : tensor<f64>, %std : tensor<f64>) -> tensor<f64> {
    %neg = stablehlo.negate %x : tensor<f64>
    return %neg : tensor<f64>
  }

  // CHECK-LABEL: func.func @user_rule_cholesky_scale
  // CHECK-SAME: (%{{[^:]+}}: tensor<2xui64>, %[[R:[^:]+]]: tensor<3x3xf64>
  // CHECK: %[[CHOL:.+]] = stablehlo.cholesky %[[R]], lower = true : tensor<3x3xf64>
  // CHECK-NEXT: impulse.infer_region @model
  // CHECK: %[[TAU:.+]]:2 = impulse.sample_region
  // CHECK: %[[TAUSQ:.+]] = stablehlo.multiply %[[TAU]]#1, %[[TAU]]#1{{.*}} : tensor<f64>
  // CHECK-NEXT: %[[SQRT:.+]] = stablehlo.sqrt %[[TAUSQ]] : tensor<f64>
  // CHECK-NEXT: %[[SQRTB:.+]] = stablehlo.broadcast_in_dim %[[SQRT]], dims = [] : (tensor<f64>) -> tensor<3x3xf64>
  // CHECK-NEXT: %[[L:.+]] = stablehlo.multiply %[[SQRTB]], %[[CHOL]] : tensor<3x3xf64>
  // CHECK-NEXT: impulse.yield %[[TAU]]#0, %[[L]] : tensor<2xui64>, tensor<3x3xf64>
  // NORULE-LABEL: func.func @user_rule_cholesky_scale
  // NORULE-NOT: stablehlo.cholesky
  // NORULE: %[[TAU:.+]]:2 = impulse.sample_region
  // NORULE: %[[TAUSQ:.+]] = stablehlo.multiply %[[TAU]]#1, %[[TAU]]#1 : tensor<f64>
  // NORULE-NEXT: %[[B:.+]] = stablehlo.broadcast_in_dim %[[TAUSQ]], dims = [] : (tensor<f64>) -> tensor<3x3xf64>
  // NORULE-NEXT: %[[K:.+]] = stablehlo.multiply %[[B]], %{{.+}} : tensor<3x3xf64>
  // NORULE-NEXT: %[[L:.+]] = stablehlo.cholesky %[[K]], lower = true : tensor<3x3xf64>
  // NORULE-NEXT: impulse.yield %[[TAU]]#0, %[[L]] : tensor<2xui64>, tensor<3x3xf64>
  // INTERP-LABEL: func.func @user_rule_cholesky_scale
  // INTERP: ^bb0(%{{[^:]+}}: tensor<2xui64>, %[[R:[^:]+]]: tensor<3x3xf64>):
  // INTERP: %[[TAU:.+]]:2 = impulse.sample_region
  // INTERP: %[[TAUSQ:.+]] = stablehlo.multiply %[[TAU]]#1, %[[TAU]]#1{{.*}} : tensor<f64>
  // INTERP-NEXT: %[[SQRT:.+]] = stablehlo.sqrt %[[TAUSQ]] : tensor<f64>
  // INTERP-NEXT: %[[CHOL:.+]] = stablehlo.cholesky %[[R]], lower = true : tensor<3x3xf64>
  // INTERP-NEXT: %[[SQRTB:.+]] = stablehlo.broadcast_in_dim %[[SQRT]], dims = [] : (tensor<f64>) -> tensor<3x3xf64>
  // INTERP-NEXT: %[[L:.+]] = stablehlo.multiply %[[SQRTB]], %[[CHOL]] : tensor<3x3xf64>
  // INTERP-NEXT: impulse.yield %[[TAU]]#0, %[[L]] : tensor<2xui64>, tensor<3x3xf64>
  func.func @user_rule_cholesky_scale(%rng : tensor<2xui64>, %R : tensor<3x3xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %R) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_R: tensor<3x3xf64>):
      %tau:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.abs %s : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%x: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %x : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<1>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<POSITIVE>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %tausq = stablehlo.multiply %tau#1, %tau#1 : tensor<f64>
      %bcast = stablehlo.broadcast_in_dim %tausq, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %K = stablehlo.multiply %bcast, %arg_R : tensor<3x3xf64>
      %L = stablehlo.cholesky %K, lower = true : tensor<3x3xf64>
      impulse.yield %tau#0, %L : tensor<2xui64>, tensor<3x3xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]} : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @user_rule_commuted_upper
  // CHECK-SAME: (%{{[^:]+}}: tensor<2xui64>, %[[R:[^:]+]]: tensor<3x3xf64>
  // CHECK: %[[CHOL:.+]] = stablehlo.cholesky %[[R]] : tensor<3x3xf64>
  // CHECK-NEXT: impulse.infer_region @model
  // CHECK: %[[TAU:.+]]:2 = impulse.sample_region
  // CHECK: %[[TAUSQ:.+]] = stablehlo.multiply %[[TAU]]#1, %[[TAU]]#1{{.*}} : tensor<f64>
  // CHECK-NEXT: %[[SQRT:.+]] = stablehlo.sqrt %[[TAUSQ]] : tensor<f64>
  // CHECK-NEXT: %[[SQRTB:.+]] = stablehlo.broadcast_in_dim %[[SQRT]], dims = [] : (tensor<f64>) -> tensor<3x3xf64>
  // CHECK-NEXT: %[[U:.+]] = stablehlo.multiply %[[SQRTB]], %[[CHOL]] : tensor<3x3xf64>
  // CHECK-NEXT: impulse.yield %[[TAU]]#0, %[[U]] : tensor<2xui64>, tensor<3x3xf64>
  // NORULE-LABEL: func.func @user_rule_commuted_upper
  // NORULE-NOT: stablehlo.cholesky
  // NORULE: %[[K:.+]] = stablehlo.multiply %{{.+}}, %{{.+}} : tensor<3x3xf64>
  // NORULE-NEXT: %[[U:.+]] = stablehlo.cholesky %[[K]] : tensor<3x3xf64>
  // NORULE-NEXT: impulse.yield %{{.+}}#0, %[[U]] : tensor<2xui64>, tensor<3x3xf64>
  func.func @user_rule_commuted_upper(%rng : tensor<2xui64>, %R : tensor<3x3xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %R) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_R: tensor<3x3xf64>):
      %tau:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.abs %s : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%x: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %x : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<1>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<POSITIVE>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %tausq = stablehlo.multiply %tau#1, %tau#1 : tensor<f64>
      %bcast = stablehlo.broadcast_in_dim %tausq, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %K = stablehlo.multiply %arg_R, %bcast : tensor<3x3xf64>
      %U = stablehlo.cholesky %K : tensor<3x3xf64>
      impulse.yield %tau#0, %U : tensor<2xui64>, tensor<3x3xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]} : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @user_rule_then_builtin_triangular_solve
  // CHECK-SAME: (%{{[^:]+}}: tensor<2xui64>, %[[R:[^:]+]]: tensor<3x3xf64>, %[[B:[^:]+]]: tensor<3x1xf64>
  // CHECK: %[[ONE:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f64>
  // CHECK: %[[CHOL:.+]] = stablehlo.cholesky %[[R]], lower = true : tensor<3x3xf64>
  // CHECK-NEXT: %[[SOLVE:.+]] = "stablehlo.triangular_solve"(%[[CHOL]], %[[B]]) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
  // CHECK-NEXT: impulse.infer_region @model
  // CHECK: %[[TAU:.+]]:2 = impulse.sample_region
  // CHECK: %[[TAUSQ:.+]] = stablehlo.multiply %[[TAU]]#1, %[[TAU]]#1{{.*}} : tensor<f64>
  // CHECK-NEXT: %[[SQRT:.+]] = stablehlo.sqrt %[[TAUSQ]] : tensor<f64>
  // CHECK-NEXT: %[[INV:.+]] = stablehlo.divide %[[ONE]], %[[SQRT]] : tensor<f64>
  // CHECK-NEXT: %[[INVB:.+]] = stablehlo.broadcast_in_dim %[[INV]], dims = [] : (tensor<f64>) -> tensor<3x1xf64>
  // CHECK-NEXT: %[[Y:.+]] = stablehlo.multiply %[[INVB]], %[[SOLVE]] : tensor<3x1xf64>
  // CHECK-NEXT: impulse.yield %[[TAU]]#0, %[[Y]] : tensor<2xui64>, tensor<3x1xf64>
  // NORULE-LABEL: func.func @user_rule_then_builtin_triangular_solve
  // NORULE-NOT: stablehlo.cholesky
  // NORULE: %[[K:.+]] = stablehlo.multiply %{{.+}}, %{{.+}} : tensor<3x3xf64>
  // NORULE-NEXT: %[[L:.+]] = stablehlo.cholesky %[[K]], lower = true : tensor<3x3xf64>
  // NORULE-NEXT: %[[Y:.+]] = "stablehlo.triangular_solve"(%[[L]], %{{.+}})
  // NORULE-NEXT: impulse.yield %{{.+}}#0, %[[Y]] : tensor<2xui64>, tensor<3x1xf64>
  func.func @user_rule_then_builtin_triangular_solve(%rng : tensor<2xui64>, %R : tensor<3x3xf64>, %b : tensor<3x1xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %R, %b) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_R: tensor<3x3xf64>, %arg_b: tensor<3x1xf64>):
      %tau:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.abs %s : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%x: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %x : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<1>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<POSITIVE>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %tausq = stablehlo.multiply %tau#1, %tau#1 : tensor<f64>
      %bcast = stablehlo.broadcast_in_dim %tausq, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %K = stablehlo.multiply %bcast, %arg_R : tensor<3x3xf64>
      %L = stablehlo.cholesky %K, lower = true : tensor<3x3xf64>
      %y = "stablehlo.triangular_solve"(%L, %arg_b) {left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
      impulse.yield %tau#0, %y : tensor<2xui64>, tensor<3x1xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]} : (tensor<2xui64>, tensor<3x3xf64>, tensor<3x1xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @user_rule_dependent_matrix_kept
  // CHECK-NOT: stablehlo.cholesky
  // CHECK: ^bb0(%{{[^:]+}}: tensor<2xui64>, %[[R:[^:]+]]: tensor<3x3xf64>):
  // CHECK: %[[TAU:.+]]:2 = impulse.sample_region
  // CHECK: %[[TAUSQ:.+]] = stablehlo.multiply %[[TAU]]#1, %[[TAU]]#1{{.*}} : tensor<f64>
  // CHECK-NEXT: %[[SB:.+]] = stablehlo.broadcast_in_dim %[[TAUSQ]], dims = [] : (tensor<f64>) -> tensor<3x3xf64>
  // CHECK-NEXT: %[[TB:.+]] = stablehlo.broadcast_in_dim %[[TAU]]#1, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
  // CHECK-NEXT: %[[A:.+]] = stablehlo.add %[[R]], %[[TB]] : tensor<3x3xf64>
  // CHECK-NEXT: %[[K:.+]] = stablehlo.multiply %[[SB]], %[[A]] : tensor<3x3xf64>
  // CHECK-NEXT: %[[L:.+]] = stablehlo.cholesky %[[K]], lower = true : tensor<3x3xf64>
  // CHECK-NEXT: impulse.yield %[[TAU]]#0, %[[L]] : tensor<2xui64>, tensor<3x3xf64>
  func.func @user_rule_dependent_matrix_kept(%rng : tensor<2xui64>, %R : tensor<3x3xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %R) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_R: tensor<3x3xf64>):
      %tau:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.abs %s : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%x: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %x : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<1>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<POSITIVE>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %tausq = stablehlo.multiply %tau#1, %tau#1 : tensor<f64>
      %bcast = stablehlo.broadcast_in_dim %tausq, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %shift = stablehlo.broadcast_in_dim %tau#1, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %A = stablehlo.add %arg_R, %shift : tensor<3x3xf64>
      %K = stablehlo.multiply %bcast, %A : tensor<3x3xf64>
      %L = stablehlo.cholesky %K, lower = true : tensor<3x3xf64>
      impulse.yield %tau#0, %L : tensor<2xui64>, tensor<3x3xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]} : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @user_rule_unproven_sign_kept
  // CHECK-NOT: stablehlo.cholesky
  // CHECK: ^bb0(%{{[^:]+}}: tensor<2xui64>, %[[R:[^:]+]]: tensor<3x3xf64>):
  // CHECK: %[[TAU:.+]]:2 = impulse.sample_region
  // CHECK: %[[SB:.+]] = stablehlo.broadcast_in_dim %[[TAU]]#1, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
  // CHECK-NEXT: %[[K:.+]] = stablehlo.multiply %[[SB]], %[[R]] : tensor<3x3xf64>
  // CHECK-NEXT: %[[L:.+]] = stablehlo.cholesky %[[K]], lower = true : tensor<3x3xf64>
  // CHECK-NEXT: impulse.yield %[[TAU]]#0, %[[L]] : tensor<2xui64>, tensor<3x3xf64>
  func.func @user_rule_unproven_sign_kept(%rng : tensor<2xui64>, %R : tensor<3x3xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %R) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_R: tensor<3x3xf64>):
      %tau:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.abs %s : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%x: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %x : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<1>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<REAL>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %bcast = stablehlo.broadcast_in_dim %tau#1, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %K = stablehlo.multiply %bcast, %arg_R : tensor<3x3xf64>
      %L = stablehlo.cholesky %K, lower = true : tensor<3x3xf64>
      impulse.yield %tau#0, %L : tensor<2xui64>, tensor<3x3xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]} : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // The built-in rules, with CholeskyScale replaced by the user rule.
  transform.named_sequence @__transform_main(%payload: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %payload : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %funcs {
      transform.apply_patterns.impulse.sicm_cholesky_outer_product_scale
      transform.apply_patterns.impulse.sicm_cholesky_eigen_lift
      transform.apply_patterns.impulse.sicm_dot_general_scale
      transform.apply_patterns.impulse.sicm_dot_general_shift
      transform.apply_patterns.impulse.sicm_triangular_solve_scale
      transform.apply_patterns.impulse.sicm_log_multiply
      transform.apply_patterns.impulse.sicm_dot_absorb_diag_mul
      transform.apply_patterns.impulse.sicm_dot_absorb_fft
      transform.apply_patterns.impulse.sicm_dot_absorb_transpose
      transform.apply_patterns.impulse.sicm_dot_absorb_scatter
      transform.apply_patterns.impulse.sicm_power_scale
      transform.apply_patterns.impulse.sicm_sqrt_scale
      transform.apply_patterns.impulse.sicm_rsqrt_scale
      transform.apply_patterns.impulse.sicm_abs_scale
      transform.apply_patterns.impulse.sicm_negate
      transform.apply_patterns.impulse.sicm_exp_add
      transform.apply_patterns.impulse.sicm_divide_scale
      transform.apply_patterns.impulse.sicm_gaussian_sufficient_stats
      transform.apply_patterns.impulse.sicm_pdl {
        pdl.pattern @UserCholeskyScale : benefit(3) {
          %0 = type
          %1 = type
          %2 = operand : %0
          %3 = operand
          %4 = attribute = array<i64>
          %5 = operation "stablehlo.broadcast_in_dim"(%2 : !pdl.value)  {"broadcast_dimensions" = %4} -> (%1 : !pdl.type)
          %6 = result 0 of %5
          %7 = operation "stablehlo.multiply"(%6, %3 : !pdl.value, !pdl.value)  -> (%1 : !pdl.type)
          %8 = result 0 of %7
          %9 = attribute
          %10 = operation "stablehlo.cholesky"(%8 : !pdl.value)  {"lower" = %9} -> (%1 : !pdl.type)
          apply_native_constraint "SampleDependent"(%2 : !pdl.value)
          apply_native_constraint "SampleInvariant"(%3 : !pdl.value)
          apply_native_constraint "Nonnegative"(%2 : !pdl.value)
          rewrite %10 {
            %11 = operation "stablehlo.sqrt"(%2 : !pdl.value)  -> (%0 : !pdl.type)
            %12 = operation "stablehlo.cholesky"(%3 : !pdl.value)  {"lower" = %9} -> (%1 : !pdl.type)
            %13 = result 0 of %11
            %14 = operation "stablehlo.broadcast_in_dim"(%13 : !pdl.value)  {"broadcast_dimensions" = %4} -> (%1 : !pdl.type)
            %15 = result 0 of %14
            %16 = result 0 of %12
            %17 = operation "stablehlo.multiply"(%15, %16 : !pdl.value, !pdl.value)  -> (%1 : !pdl.type)
            replace %10 with %17
          }
        }
        pdl.pattern @UserCholeskyScaleCommuted : benefit(3) {
          %0 = type
          %1 = type
          %2 = operand : %0
          %3 = operand
          %4 = attribute = array<i64>
          %5 = operation "stablehlo.broadcast_in_dim"(%2 : !pdl.value)  {"broadcast_dimensions" = %4} -> (%1 : !pdl.type)
          %6 = result 0 of %5
          %7 = operation "stablehlo.multiply"(%3, %6 : !pdl.value, !pdl.value)  -> (%1 : !pdl.type)
          %8 = result 0 of %7
          %9 = attribute
          %10 = operation "stablehlo.cholesky"(%8 : !pdl.value)  {"lower" = %9} -> (%1 : !pdl.type)
          apply_native_constraint "SampleDependent"(%2 : !pdl.value)
          apply_native_constraint "SampleInvariant"(%3 : !pdl.value)
          apply_native_constraint "Nonnegative"(%2 : !pdl.value)
          rewrite %10 {
            %11 = operation "stablehlo.sqrt"(%2 : !pdl.value)  -> (%0 : !pdl.type)
            %12 = operation "stablehlo.cholesky"(%3 : !pdl.value)  {"lower" = %9} -> (%1 : !pdl.type)
            %13 = result 0 of %11
            %14 = operation "stablehlo.broadcast_in_dim"(%13 : !pdl.value)  {"broadcast_dimensions" = %4} -> (%1 : !pdl.type)
            %15 = result 0 of %14
            %16 = result 0 of %12
            %17 = operation "stablehlo.multiply"(%15, %16 : !pdl.value, !pdl.value)  -> (%1 : !pdl.type)
            replace %10 with %17
          }
        }
        pdl.pattern @UserCholeskyScaleDefaultLower : benefit(2) {
          %0 = type
          %1 = type
          %2 = operand : %0
          %3 = operand
          %4 = attribute = array<i64>
          %5 = operation "stablehlo.broadcast_in_dim"(%2 : !pdl.value)  {"broadcast_dimensions" = %4} -> (%1 : !pdl.type)
          %6 = result 0 of %5
          %7 = operation "stablehlo.multiply"(%6, %3 : !pdl.value, !pdl.value)  -> (%1 : !pdl.type)
          %8 = result 0 of %7
          %9 = operation "stablehlo.cholesky"(%8 : !pdl.value)  -> (%1 : !pdl.type)
          apply_native_constraint "SampleDependent"(%2 : !pdl.value)
          apply_native_constraint "SampleInvariant"(%3 : !pdl.value)
          apply_native_constraint "Nonnegative"(%2 : !pdl.value)
          rewrite %9 {
            %10 = operation "stablehlo.sqrt"(%2 : !pdl.value)  -> (%0 : !pdl.type)
            %11 = operation "stablehlo.cholesky"(%3 : !pdl.value)  -> (%1 : !pdl.type)
            %12 = result 0 of %10
            %13 = operation "stablehlo.broadcast_in_dim"(%12 : !pdl.value)  {"broadcast_dimensions" = %4} -> (%1 : !pdl.type)
            %14 = result 0 of %13
            %15 = result 0 of %11
            %16 = operation "stablehlo.multiply"(%14, %15 : !pdl.value, !pdl.value)  -> (%1 : !pdl.type)
            replace %9 with %16
          }
        }
        pdl.pattern @UserCholeskyScaleCommutedDefaultLower : benefit(2) {
          %0 = type
          %1 = type
          %2 = operand : %0
          %3 = operand
          %4 = attribute = array<i64>
          %5 = operation "stablehlo.broadcast_in_dim"(%2 : !pdl.value)  {"broadcast_dimensions" = %4} -> (%1 : !pdl.type)
          %6 = result 0 of %5
          %7 = operation "stablehlo.multiply"(%3, %6 : !pdl.value, !pdl.value)  -> (%1 : !pdl.type)
          %8 = result 0 of %7
          %9 = operation "stablehlo.cholesky"(%8 : !pdl.value)  -> (%1 : !pdl.type)
          apply_native_constraint "SampleDependent"(%2 : !pdl.value)
          apply_native_constraint "SampleInvariant"(%3 : !pdl.value)
          apply_native_constraint "Nonnegative"(%2 : !pdl.value)
          rewrite %9 {
            %10 = operation "stablehlo.sqrt"(%2 : !pdl.value)  -> (%0 : !pdl.type)
            %11 = operation "stablehlo.cholesky"(%3 : !pdl.value)  -> (%1 : !pdl.type)
            %12 = result 0 of %10
            %13 = operation "stablehlo.broadcast_in_dim"(%12 : !pdl.value)  {"broadcast_dimensions" = %4} -> (%1 : !pdl.type)
            %14 = result 0 of %13
            %15 = result 0 of %11
            %16 = operation "stablehlo.multiply"(%14, %15 : !pdl.value, !pdl.value)  -> (%1 : !pdl.type)
            replace %9 with %16
          }
        }
      }
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @builtin_rules_without_cholesky_scale(%payload: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %payload : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %funcs {
      transform.apply_patterns.impulse.sicm_cholesky_outer_product_scale
      transform.apply_patterns.impulse.sicm_cholesky_eigen_lift
      transform.apply_patterns.impulse.sicm_dot_general_scale
      transform.apply_patterns.impulse.sicm_dot_general_shift
      transform.apply_patterns.impulse.sicm_triangular_solve_scale
      transform.apply_patterns.impulse.sicm_log_multiply
      transform.apply_patterns.impulse.sicm_dot_absorb_diag_mul
      transform.apply_patterns.impulse.sicm_dot_absorb_fft
      transform.apply_patterns.impulse.sicm_dot_absorb_transpose
      transform.apply_patterns.impulse.sicm_dot_absorb_scatter
      transform.apply_patterns.impulse.sicm_power_scale
      transform.apply_patterns.impulse.sicm_sqrt_scale
      transform.apply_patterns.impulse.sicm_rsqrt_scale
      transform.apply_patterns.impulse.sicm_abs_scale
      transform.apply_patterns.impulse.sicm_negate
      transform.apply_patterns.impulse.sicm_exp_add
      transform.apply_patterns.impulse.sicm_divide_scale
      transform.apply_patterns.impulse.sicm_gaussian_sufficient_stats
    } : !transform.any_op
    transform.yield
  }
}
