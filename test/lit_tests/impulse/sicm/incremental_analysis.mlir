// RUN: enzymexlamlir-opt %s --sicm="schedule=%s" | FileCheck %s
// RUN: enzymexlamlir-opt %s --sicm="schedule=%s verify-incremental-analysis=true" 2>/dev/null | FileCheck %s
// RUN: enzymexlamlir-opt %s --sicm="schedule=%s schedule-entry-point=without_user_rule" | FileCheck %s --check-prefix=NORULE
// RUN: enzymexlamlir-opt %s --sicm="schedule=%s max-iterations=1" | FileCheck %s --check-prefix=ONE
// RUN: enzymexlamlir-opt %s --sicm="schedule=%s verify-incremental-analysis=true" -o /dev/null 2>&1 | FileCheck %s --check-prefix=VERIFY --implicit-check-not=UNSOUND

// CHECK-LABEL: func.func @rule_keeps_label
// CHECK-SAME: (%{{[^:]+}}: tensor<2xui64>, %[[R:[^:]+]]: tensor<3x3xf64>
// CHECK: %[[CHOL:.+]] = stablehlo.cholesky %[[R]], lower = true : tensor<3x3xf64>
// CHECK-NEXT: impulse.infer_region @model
// CHECK: %[[TAU:.+]]:2 = impulse.sample_region
// CHECK: %[[TAUSQ:.+]] = stablehlo.multiply %[[TAU]]#1, %[[TAU]]#1{{.*}} : tensor<f64>
// CHECK-NEXT: %[[SQRT:.+]] = stablehlo.sqrt %[[TAUSQ]] : tensor<f64>
// CHECK-NEXT: %[[SQRTB:.+]] = stablehlo.broadcast_in_dim %[[SQRT]], dims = [] : (tensor<f64>) -> tensor<3x3xf64>
// CHECK-NEXT: %[[L:.+]] = stablehlo.multiply %[[SQRTB]], %[[CHOL]] : tensor<3x3xf64>
// CHECK-NEXT: impulse.yield %[[TAU]]#0, %[[L]] : tensor<2xui64>, tensor<3x3xf64>

// CHECK-LABEL: func.func @rule_adds_dependence
// CHECK-NOT: stablehlo.exponential
// CHECK-NOT: stablehlo.log
// CHECK: impulse.infer_region @model
// CHECK-NEXT: ^bb0(%{{[^:]+}}: tensor<2xui64>, %[[C:[^:]+]]: tensor<3xf64>):
// CHECK: %[[TAU:.+]]:2 = impulse.sample_region
// CHECK: %[[TB:.+]] = stablehlo.broadcast_in_dim %[[TAU]]#1, dims = [] : (tensor<f64>) -> tensor<3xf64>
// CHECK-NEXT: %[[M:.+]] = stablehlo.multiply %[[C]], %[[TB]] : tensor<3xf64>
// CHECK-NEXT: %[[D:.+]] = stablehlo.subtract %[[M]], %[[M]] : tensor<3xf64>
// CHECK-NEXT: %[[NEG:.+]] = stablehlo.subtract %[[D]], %[[C]] : tensor<3xf64>
// CHECK-NEXT: %[[E:.+]] = stablehlo.exponential %[[NEG]] : tensor<3xf64>
// CHECK-NEXT: %[[W:.+]] = stablehlo.log %[[E]] : tensor<3xf64>
// CHECK-NEXT: %[[Y:.+]] = stablehlo.add %[[W]], %[[M]] : tensor<3xf64>
// CHECK-NEXT: impulse.yield %[[TAU]]#0, %[[Y]] : tensor<2xui64>, tensor<3xf64>

// NORULE-LABEL: func.func @rule_adds_dependence
// NORULE-SAME: (%{{[^:]+}}: tensor<2xui64>, %[[C:[^:]+]]: tensor<3xf64>
// NORULE: %[[NEG:.+]] = stablehlo.negate %[[C]] : tensor<3xf64>
// NORULE-NEXT: %[[E:.+]] = stablehlo.exponential %[[NEG]] : tensor<3xf64>
// NORULE-NEXT: %[[W:.+]] = stablehlo.log %[[E]] : tensor<3xf64>
// NORULE-NEXT: impulse.infer_region @model
// NORULE: %[[TB:.+]] = stablehlo.broadcast_in_dim %{{.+}}#1, dims = [] : (tensor<f64>) -> tensor<3xf64>
// NORULE-NEXT: %[[M:.+]] = stablehlo.multiply %{{.+}}, %[[TB]] : tensor<3xf64>
// NORULE-NEXT: %[[Y:.+]] = stablehlo.add %[[W]], %[[M]] : tensor<3xf64>
// NORULE-NEXT: impulse.yield %{{.+}}#0, %[[Y]] : tensor<2xui64>, tensor<3xf64>

// CHECK-LABEL: func.func @fold_removes_dependence
// CHECK-SAME: (%{{[^:]+}}: tensor<2xui64>, %[[R:[^:]+]]: tensor<3x3xi64>
// CHECK: %[[ZERO:.+]] = stablehlo.constant dense<0> : tensor<3x3xi64>
// CHECK: %[[A:.+]] = stablehlo.add %[[R]], %[[ZERO]] : tensor<3x3xi64>
// CHECK-NEXT: %[[DOT:.+]] = stablehlo.dot_general %[[A]], %[[ZERO]], contracting_dims = [1] x [0] : (tensor<3x3xi64>, tensor<3x3xi64>) -> tensor<3x3xi64>
// CHECK-NEXT: impulse.infer_region @model
// CHECK: %[[TAU:.+]]:2 = impulse.sample_region
// CHECK: %[[TI:.+]] = stablehlo.convert %[[TAU]]#1 : (tensor<f64>) -> tensor<i64>
// CHECK-NEXT: %[[TB:.+]] = stablehlo.broadcast_in_dim %[[TI]], dims = [] : (tensor<i64>) -> tensor<3x3xi64>
// CHECK-NEXT: %[[D:.+]] = stablehlo.multiply %[[TB]], %[[DOT]] : tensor<3x3xi64>
// CHECK-NEXT: impulse.yield %[[TAU]]#0, %[[D]] : tensor<2xui64>, tensor<3x3xi64>

// ONE-LABEL: func.func @fold_removes_dependence
// ONE-SAME: (%{{[^:]+}}: tensor<2xui64>, %[[R:[^:]+]]: tensor<3x3xi64>
// ONE: %[[ZERO:.+]] = stablehlo.constant dense<0> : tensor<3x3xi64>
// ONE: %[[A:.+]] = stablehlo.add %[[R]], %[[ZERO]] : tensor<3x3xi64>
// ONE-NEXT: impulse.infer_region @model
// ONE: %[[TAU:.+]]:2 = impulse.sample_region
// ONE: %[[TI:.+]] = stablehlo.convert %[[TAU]]#1 : (tensor<f64>) -> tensor<i64>
// ONE-NEXT: %[[DOT:.+]] = stablehlo.dot_general %[[A]], %[[ZERO]], contracting_dims = [1] x [0] : (tensor<3x3xi64>, tensor<3x3xi64>) -> tensor<3x3xi64>
// ONE-NEXT: %[[TB:.+]] = stablehlo.broadcast_in_dim %[[TI]], dims = [] : (tensor<i64>) -> tensor<3x3xi64>
// ONE-NEXT: %[[D:.+]] = stablehlo.multiply %[[TB]], %[[DOT]] : tensor<3x3xi64>
// ONE-NEXT: impulse.yield %[[TAU]]#0, %[[D]] : tensor<2xui64>, tensor<3x3xi64>

// @rule_keeps_label
// VERIFY: sicm-verify-incremental: sampler round 0: {{[0-9]+}} checks, 0 unsound, stale 0 at end (max 0); replacements 1 same label, 0 add, 0 remove; 1 rule applications, 0 change a label;
// VERIFY-NEXT: sicm-verify-incremental: sampler round 1: 0 checks, 0 unsound, stale 0 at end (max 0); replacements 0 same label, 0 add, 0 remove; 0 rule applications, 0 change a label;
// VERIFY-NEXT: sicm-verify-incremental: logpdf round 0: 0 checks, 0 unsound, stale 0 at end (max 0);
// @rule_adds_dependence
// VERIFY-NEXT: sicm-verify-incremental: replacement adds a dependence: 'stablehlo.negate' by <pattern rooted at stablehlo.negate>
// VERIFY-NEXT: sicm-verify-incremental: rule <pattern rooted at stablehlo.negate> changes the label of 2 surviving value(s)
// VERIFY-DAG: sicm-verify-incremental:   now dependent: result 0 of %{{.+}} = stablehlo.exponential %{{.+}} : tensor<3xf64>
// VERIFY-DAG: sicm-verify-incremental:   now dependent: result 0 of %{{.+}} = stablehlo.log %{{.+}} : tensor<3xf64>
// VERIFY: sicm-verify-incremental: sampler round 0: {{[0-9]+}} checks, 0 unsound, stale 0 at end (max {{[0-9]+}}); replacements 0 same label, 1 add, 0 remove; 1 rule applications, 1 change a label; 0 rebuilt; work {{[0-9]+}} raised, 3 use visits,
// VERIFY-NEXT: sicm-verify-incremental: sampler round 1: 0 checks, 0 unsound, stale 0 at end (max 0); replacements 0 same label, 0 add, 0 remove; 0 rule applications, 0 change a label;
// VERIFY-NEXT: sicm-verify-incremental: logpdf round 0: 0 checks, 0 unsound, stale 0 at end (max 0);
// @fold_removes_dependence
// VERIFY-NEXT: sicm-verify-incremental: replacement removes a dependence: 'arith.muli' by folding
// VERIFY-NEXT: sicm-verify-incremental: sampler round 0: {{[0-9]+}} checks, 0 unsound, stale 1 at end (max 1); replacements 0 same label, 0 add, 1 remove; 0 rule applications, 0 change a label;
// VERIFY-NEXT: sicm-verify-incremental: sampler round 1: {{[0-9]+}} checks, 0 unsound, stale 0 at end (max 0); replacements 1 same label, 0 add, 0 remove; 1 rule applications, 0 change a label;
// VERIFY-NEXT: sicm-verify-incremental: sampler round 2: 0 checks, 0 unsound, stale 0 at end (max 0); replacements 0 same label, 0 add, 0 remove; 0 rule applications, 0 change a label;
// VERIFY-NEXT: sicm-verify-incremental: logpdf round 0: 0 checks, 0 unsound, stale 0 at end (max 0);

module attributes {transform.with_named_sequence} {
  func.func private @halfnormal_sampler(%rng : tensor<2xui64>, %std : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %sample = stablehlo.abs %std : tensor<f64>
    return %rng, %sample : tensor<2xui64>, tensor<f64>
  }
  func.func private @halfnormal_logpdf(%x : tensor<f64>, %std : tensor<f64>) -> tensor<f64> {
    %neg = stablehlo.negate %x : tensor<f64>
    return %neg : tensor<f64>
  }

  func.func @rule_keeps_label(%rng : tensor<2xui64>, %R : tensor<3x3xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
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

  func.func @rule_adds_dependence(%rng : tensor<2xui64>, %c : tensor<3xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %c) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_c: tensor<3xf64>):
      %tau:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.abs %s : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%x: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %x : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<1>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<POSITIVE>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %tb = stablehlo.broadcast_in_dim %tau#1, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %m = stablehlo.multiply %arg_c, %tb : tensor<3xf64>
      %n = stablehlo.negate %arg_c : tensor<3xf64>
      %e = stablehlo.exponential %n : tensor<3xf64>
      %w = stablehlo.log %e : tensor<3xf64>
      %y = stablehlo.add %w, %m : tensor<3xf64>
      impulse.yield %tau#0, %y : tensor<2xui64>, tensor<3xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]} : (tensor<2xui64>, tensor<3xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  func.func @fold_removes_dependence(%rng : tensor<2xui64>, %R : tensor<3x3xi64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %R) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_R: tensor<3x3xi64>):
      %tau:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.abs %s : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%x: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %x : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<1>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<REAL>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %ti = stablehlo.convert %tau#1 : (tensor<f64>) -> tensor<i64>
      %tb = stablehlo.broadcast_in_dim %ti, dims = [] : (tensor<i64>) -> tensor<3x3xi64>
      %x = stablehlo.add %arg_R, %tb : tensor<3x3xi64>
      %zero = stablehlo.constant dense<0> : tensor<3x3xi64>
      %z = arith.muli %x, %zero : tensor<3x3xi64>
      %A = stablehlo.add %arg_R, %z : tensor<3x3xi64>
      %SA = stablehlo.multiply %tb, %A : tensor<3x3xi64>
      %d = stablehlo.dot_general %SA, %z, contracting_dims = [1] x [0] : (tensor<3x3xi64>, tensor<3x3xi64>) -> tensor<3x3xi64>
      impulse.yield %tau#0, %d : tensor<2xui64>, tensor<3x3xi64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]} : (tensor<2xui64>, tensor<3x3xi64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // NegateThroughSample: -c = (c * t - c * t) - c, c invariant, t dependent.
  transform.named_sequence @__transform_main(%payload: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %payload : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %funcs {
      transform.apply_patterns.impulse.sicm_cholesky_scale
      transform.apply_patterns.impulse.sicm_dot_general_scale
      transform.apply_patterns.impulse.sicm_pdl {
        pdl.pattern @NegateThroughSample : benefit(1) {
          %0 = type
          %1 = operand : %0
          %2 = operand
          %3 = operation "stablehlo.negate"(%1 : !pdl.value) -> (%0 : !pdl.type)
          %4 = operation "stablehlo.multiply"(%1, %2 : !pdl.value, !pdl.value) -> (%0 : !pdl.type)
          %5 = result 0 of %4
          apply_native_constraint "SampleInvariant"(%1 : !pdl.value)
          apply_native_constraint "SampleDependent"(%2 : !pdl.value)
          rewrite %3 {
            %6 = operation "stablehlo.subtract"(%5, %5 : !pdl.value, !pdl.value) -> (%0 : !pdl.type)
            %7 = result 0 of %6
            %8 = operation "stablehlo.subtract"(%7, %1 : !pdl.value, !pdl.value) -> (%0 : !pdl.type)
            replace %3 with %8
          }
        }
      }
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @without_user_rule(%payload: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %payload : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %funcs {
      transform.apply_patterns.impulse.sicm_cholesky_scale
      transform.apply_patterns.impulse.sicm_dot_general_scale
    } : !transform.any_op
    transform.yield
  }
}
