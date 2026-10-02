// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=slice_if" --transform-interpreter --enzyme-hlo-remove-transform --verify-each | FileCheck %s

// An unsliced result is used before the slice being pushed into the if.
// The replacement if must stay before that early use.
// CHECK-LABEL: func.func @early_unsliced_result
// CHECK: %[[IF:.*]]:2 = "stablehlo.if"
// CHECK: stablehlo.slice
// CHECK: stablehlo.slice
// CHECK: %[[EARLY:.*]] = stablehlo.add %[[IF]]#1, %arg2
// CHECK: %[[VALUE:.*]] = stablehlo.reshape %[[IF]]#0
// CHECK: stablehlo.add %[[EARLY]], %[[VALUE]]
func.func @early_unsliced_result(%pred: tensor<i1>, %x: tensor<2xf64>, %s: tensor<f64>) -> tensor<f64> {
  %r:2 = "stablehlo.if"(%pred) ({
    stablehlo.return %x, %s : tensor<2xf64>, tensor<f64>
  }, {
    %nx = stablehlo.negate %x : tensor<2xf64>
    %ns = stablehlo.negate %s : tensor<f64>
    stablehlo.return %nx, %ns : tensor<2xf64>, tensor<f64>
  }) : (tensor<i1>) -> (tensor<2xf64>, tensor<f64>)
  %early = stablehlo.add %r#1, %s : tensor<f64>
  %tail = stablehlo.slice %r#0 [1:2] : (tensor<2xf64>) -> tensor<1xf64>
  %value = stablehlo.reshape %tail : (tensor<1xf64>) -> tensor<f64>
  %result = stablehlo.add %early, %value : tensor<f64>
  return %result : tensor<f64>
}

// Repeat the case within a loop with a live trip count. The slice rewrite
// must preserve the loop and its lazy conditional, including zero iterations.
// CHECK-LABEL: func.func @loop_carried_early_unsliced_result
// CHECK: stablehlo.while
// CHECK: %[[LOOP_IF:.*]]:2 = "stablehlo.if"
// CHECK: stablehlo.slice
// CHECK: stablehlo.slice
// CHECK: %[[LOOP_EARLY:.*]] = stablehlo.add %[[LOOP_IF]]#1,
// CHECK: %[[LOOP_VALUE:.*]] = stablehlo.reshape %[[LOOP_IF]]#0
// CHECK: stablehlo.add %[[LOOP_EARLY]], %[[LOOP_VALUE]]
func.func @loop_carried_early_unsliced_result(%pred: tensor<i1>, %x: tensor<2xf64>, %s: tensor<f64>, %limit: tensor<i32>) -> tensor<f64> {
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %one = stablehlo.constant dense<1> : tensor<i32>
  %loop:2 = stablehlo.while(%i = %zero, %carry = %s) : tensor<i32>, tensor<f64>
  cond {
    %more = stablehlo.compare LT, %i, %limit : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %more : tensor<i1>
  } do {
    %r:2 = "stablehlo.if"(%pred) ({
      stablehlo.return %x, %carry : tensor<2xf64>, tensor<f64>
    }, {
      %nx = stablehlo.negate %x : tensor<2xf64>
      %ns = stablehlo.negate %carry : tensor<f64>
      stablehlo.return %nx, %ns : tensor<2xf64>, tensor<f64>
    }) : (tensor<i1>) -> (tensor<2xf64>, tensor<f64>)
    %early = stablehlo.add %r#1, %carry : tensor<f64>
    %tail = stablehlo.slice %r#0 [1:2] : (tensor<2xf64>) -> tensor<1xf64>
    %value = stablehlo.reshape %tail : (tensor<1xf64>) -> tensor<f64>
    %result = stablehlo.add %early, %value : tensor<f64>
    %next = stablehlo.add %i, %one : tensor<i32>
    stablehlo.return %next, %result : tensor<i32>, tensor<f64>
  }
  return %loop#1 : tensor<f64>
}
