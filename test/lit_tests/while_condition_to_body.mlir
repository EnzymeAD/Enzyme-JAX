// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s --check-prefixes=CHECK,ONCE
// RUN: enzymexlamlir-opt --enzyme-hlo-opt --enzyme-hlo-opt %s | FileCheck %s --check-prefixes=CHECK,TWICE

// Before: test i < limit in the condition, then increment i in the body.
// After: test 0 < limit once before entering, and return the next comparison
// from the body alongside i + 1. The condition only returns the carried bool.
// This lets a GPU backend fuse the comparison with the increment and avoids
// a separate condition kernel. A nonpositive limit still executes zero times.
// CHECK-LABEL: func.func @dynamic_limit
// CHECK-DAG: %[[ZERO:.*]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-DAG: %[[ONE:.*]] = stablehlo.constant dense<1> : tensor<i32>
// ONCE: %[[INIT:.*]] = stablehlo.compare LT, %[[ZERO]], %arg0
// A subsequent optimizer invocation may put the constant on the right, but
// must not add another predicate argument to the loop.
// TWICE: %[[INIT:.*]] = stablehlo.compare GT, %arg0, %[[ZERO]]
// CHECK: %[[RESULT:.*]]:3 = stablehlo.while(%[[I:.*]] = %[[ZERO]], %[[X:.*]] = %arg1, %[[P:.*]] = %[[INIT]])
// CHECK-NEXT: cond {
// CHECK-NEXT: stablehlo.return %[[P]] : tensor<i1>
// CHECK-NEXT: } do {
// CHECK: %[[NEXT:.*]] = stablehlo.add %[[I]], %[[ONE]]
// CHECK: %[[SQUARE:.*]] = stablehlo.multiply %[[X]], %[[X]]
// CHECK: %[[TEST:.*]] = stablehlo.compare LT, %[[NEXT]], %arg0
// CHECK: stablehlo.return %[[NEXT]], %[[SQUARE]], %[[TEST]]
// CHECK: return %[[RESULT]]#1
func.func @dynamic_limit(%limit: tensor<i32>, %input: tensor<8xf32>) -> tensor<8xf32> {
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %one = stablehlo.constant dense<1> : tensor<i32>
  %result:2 = stablehlo.while(%i = %zero, %values = %input) : tensor<i32>, tensor<8xf32>
  cond {
    %test = stablehlo.compare LT, %i, %limit, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %test : tensor<i1>
  } do {
    %next = stablehlo.add %i, %one : tensor<i32>
    %square = stablehlo.multiply %values, %values : tensor<8xf32>
    stablehlo.return %next, %square : tensor<i32>, tensor<8xf32>
  }
  return %result#1 : tensor<8xf32>
}

// The initial comparison must use the actual initial counter, and the next
// comparison must use the actual updated counter. Do not assume start=0 or
// step=1. The data result must still be returned to the caller.
// CHECK-LABEL: func.func @dynamic_start_and_step_two
// CHECK: %[[INIT:.*]] = stablehlo.compare LT, %arg0, %arg1
// CHECK: %[[RESULT:.*]]:3 = stablehlo.while(%[[I:.*]] = %arg0, %[[X:.*]] = %arg2, %[[P:.*]] = %[[INIT]])
// CHECK-NEXT: cond {
// CHECK-NEXT: stablehlo.return %[[P]]
// CHECK-NEXT: } do {
// CHECK: %[[NEXT:.*]] = stablehlo.add %[[I]],
// CHECK: %[[TEST:.*]] = stablehlo.compare LT, %[[NEXT]], %arg1
// CHECK: stablehlo.return %[[NEXT]], %{{.*}}, %[[TEST]]
// CHECK: return %[[RESULT]]#1
func.func @dynamic_start_and_step_two(%start: tensor<i32>, %limit: tensor<i32>, %input: tensor<8xf32>) -> tensor<8xf32> {
  %two = stablehlo.constant dense<2> : tensor<i32>
  %result:2 = stablehlo.while(%i = %start, %values = %input) : tensor<i32>, tensor<8xf32>
  cond {
    %test = stablehlo.compare LT, %i, %limit, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %test : tensor<i1>
  } do {
    %next = stablehlo.add %i, %two : tensor<i32>
    %square = stablehlo.multiply %values, %values : tensor<8xf32>
    stablehlo.return %next, %square : tensor<i32>, tensor<8xf32>
  }
  return %result#1 : tensor<8xf32>
}

// Preserve unsigned comparison semantics, including values whose signed
// interpretation is negative. The comparison is cloned, not reconstructed.
// CHECK-LABEL: func.func @unsigned_limit
// CHECK: stablehlo.compare {{(LT|GT)}}, {{.*}}, UNSIGNED
// CHECK: stablehlo.while
// CHECK-NEXT: cond {
// CHECK-NEXT: stablehlo.return %{{.*}} : tensor<i1>
// CHECK-NEXT: } do {
// CHECK: stablehlo.compare {{(LT|GT)}}, {{.*}}, UNSIGNED
func.func @unsigned_limit(%start: tensor<i32>, %limit: tensor<i32>, %input: tensor<8xf32>) -> tensor<8xf32> {
  %one = stablehlo.constant dense<1> : tensor<i32>
  %result:2 = stablehlo.while(%i = %start, %values = %input) : tensor<i32>, tensor<8xf32>
  cond {
    %test = stablehlo.compare GT, %limit, %i, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %test : tensor<i1>
  } do {
    %next = stablehlo.add %i, %one : tensor<i32>
    %square = stablehlo.multiply %values, %values : tensor<8xf32>
    stablehlo.return %next, %square : tensor<i32>, tensor<8xf32>
  }
  return %result#1 : tensor<8xf32>
}

// Static trip counts should remain recognizable to downstream unrolling.
// CHECK-LABEL: func.func @static_limit
// CHECK: :2 = stablehlo.while
// CHECK-NEXT: cond {
// CHECK-NEXT: %[[TEST:.*]] = stablehlo.compare
// CHECK-NEXT: stablehlo.return %[[TEST]]
func.func @static_limit(%input: tensor<8xf32>) -> tensor<8xf32> {
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %one = stablehlo.constant dense<1> : tensor<i32>
  %three = stablehlo.constant dense<3> : tensor<i32>
  %result:2 = stablehlo.while(%i = %zero, %values = %input) : tensor<i32>, tensor<8xf32>
  cond {
    %test = stablehlo.compare LT, %i, %three, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %test : tensor<i1>
  } do {
    %next = stablehlo.add %i, %one : tensor<i32>
    %square = stablehlo.multiply %values, %values : tensor<8xf32>
    stablehlo.return %next, %square : tensor<i32>, tensor<8xf32>
  }
  return %result#1 : tensor<8xf32>
}

// An effect in the condition is outside the scope of this lowering. Keep it
// in the condition region, including its evaluation when the loop exits.
// CHECK-LABEL: func.func @effectful_condition
// CHECK: :2 = stablehlo.while
// CHECK-NEXT: cond {
// CHECK: stablehlo.custom_call @observe
// CHECK: %[[TEST:.*]] = stablehlo.compare
// CHECK: stablehlo.return %[[TEST]]
func.func @effectful_condition(%limit: tensor<i32>, %input: tensor<8xf32>) -> tensor<8xf32> {
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %one = stablehlo.constant dense<1> : tensor<i32>
  %result:2 = stablehlo.while(%i = %zero, %values = %input) : tensor<i32>, tensor<8xf32>
  cond {
    stablehlo.custom_call @observe(%i) {has_side_effect = true} : (tensor<i32>) -> ()
    %test = stablehlo.compare LT, %i, %limit, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %test : tensor<i1>
  } do {
    %next = stablehlo.add %i, %one : tensor<i32>
    %square = stablehlo.multiply %values, %values : tensor<8xf32>
    stablehlo.return %next, %square : tensor<i32>, tensor<8xf32>
  }
  return %result#1 : tensor<8xf32>
}

// An explicit result-sharding contract needs a corresponding entry for the
// added predicate. Leave such loops alone until that contract can be updated.
// CHECK-LABEL: func.func @sharded_loop
// CHECK: :2 = stablehlo.while
// CHECK-SAME: mhlo.sharding
// CHECK-NEXT: cond {
// CHECK-NEXT: %[[TEST:.*]] = stablehlo.compare
// CHECK-NEXT: stablehlo.return %[[TEST]]
func.func @sharded_loop(%limit: tensor<i32>, %input: tensor<8xf32>) -> tensor<8xf32> {
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %one = stablehlo.constant dense<1> : tensor<i32>
  %result:2 = "stablehlo.while"(%zero, %input) ({
  ^bb0(%i: tensor<i32>, %values: tensor<8xf32>):
    %test = stablehlo.compare LT, %i, %limit, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %test : tensor<i1>
  }, {
  ^bb0(%i: tensor<i32>, %values: tensor<8xf32>):
    %next = stablehlo.add %i, %one : tensor<i32>
    %square = stablehlo.multiply %values, %values : tensor<8xf32>
    stablehlo.return %next, %square : tensor<i32>, tensor<8xf32>
  }) {mhlo.sharding = "{{replicated}, {replicated}}"} : (tensor<i32>, tensor<8xf32>) -> (tensor<i32>, tensor<8xf32>)
  return %result#1 : tensor<8xf32>
}

// Checkpointing must still recognize the original induction comparison when
// EnzymeHLOOpt runs before differentiation.
// CHECK-LABEL: func.func @checkpointed_loop
// CHECK: :2 = stablehlo.while
// CHECK-NEXT: cond {
// CHECK-NEXT: %[[TEST:.*]] = stablehlo.compare
// CHECK-NEXT: stablehlo.return %[[TEST]]
func.func @checkpointed_loop(%limit: tensor<i32>, %input: tensor<8xf32>) -> tensor<8xf32> {
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %one = stablehlo.constant dense<1> : tensor<i32>
  %result:2 = stablehlo.while(%i = %zero, %values = %input) : tensor<i32>, tensor<8xf32> attributes {enzyme.enable_checkpointing = true, enzyme.binomial_checkpointing, enzyme.checkpoint_period = 3 : i64}
  cond {
    %test = stablehlo.compare LT, %i, %limit, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %test : tensor<i1>
  } do {
    %next = stablehlo.add %i, %one : tensor<i32>
    %square = stablehlo.multiply %values, %values : tensor<8xf32>
    stablehlo.return %next, %square : tensor<i32>, tensor<8xf32>
  }
  return %result#1 : tensor<8xf32>
}

// Generated checkpoint segments may be differentiated again. Preserve their
// counted-loop form as well.
// CHECK-LABEL: func.func @checkpoint_segment
// CHECK: :2 = stablehlo.while
// CHECK-NEXT: cond {
// CHECK-NEXT: %[[TEST:.*]] = stablehlo.compare
// CHECK-NEXT: stablehlo.return %[[TEST]]
func.func @checkpoint_segment(%limit: tensor<i32>, %input: tensor<8xf32>) -> tensor<8xf32> {
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %one = stablehlo.constant dense<1> : tensor<i32>
  %result:2 = stablehlo.while(%i = %zero, %values = %input) : tensor<i32>, tensor<8xf32> attributes {enzymexla.checkpoint_segment}
  cond {
    %test = stablehlo.compare LT, %i, %limit, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %test : tensor<i1>
  } do {
    %next = stablehlo.add %i, %one : tensor<i32>
    %square = stablehlo.multiply %values, %values : tensor<8xf32>
    stablehlo.return %next, %square : tensor<i32>, tensor<8xf32>
  }
  return %result#1 : tensor<8xf32>
}
