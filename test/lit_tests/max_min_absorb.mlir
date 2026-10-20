// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=max_simplify;min_simplify" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @max_absorb(%a: tensor<4xf32>, %x: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %m = stablehlo.maximum %a, %x : tensor<4xf32>
  %0 = stablehlo.maximum %a, %m : tensor<4xf32>
  %1 = stablehlo.maximum %m, %x : tensor<4xf32>
  return %0, %1 : tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: func.func @max_absorb(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4xf32>, %[[V1:[^ ,:)]+]]: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.maximum %[[V0]], %[[V1]] : tensor<4xf32>
// CHECK-NEXT: return %[[V2]], %[[V2]] : tensor<4xf32>, tensor<4xf32>
// CHECK-NEXT: }

func.func @min_absorb(%a: tensor<4xi32>, %x: tensor<4xi32>) -> tensor<4xi32> {
  %m = stablehlo.minimum %x, %a : tensor<4xi32>
  %0 = stablehlo.minimum %a, %m : tensor<4xi32>
  return %0 : tensor<4xi32>
}

// CHECK-LABEL: func.func @min_absorb(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4xi32>, %[[V1:[^ ,:)]+]]: tensor<4xi32>) -> tensor<4xi32> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.minimum %[[V1]], %[[V0]] : tensor<4xi32>
// CHECK-NEXT: return %[[V2]] : tensor<4xi32>
// CHECK-NEXT: }

// Stays: the inner maximum is of other values.
func.func @other_values(%a: tensor<4xf32>, %x: tensor<4xf32>, %y: tensor<4xf32>) -> tensor<4xf32> {
  %m = stablehlo.maximum %x, %y : tensor<4xf32>
  %0 = stablehlo.maximum %a, %m : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @other_values(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4xf32>, %[[V1:[^ ,:)]+]]: tensor<4xf32>, %[[V2:[^ ,:)]+]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.maximum %[[V1]], %[[V2]] : tensor<4xf32>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.maximum %[[V0]], %[[V3]] : tensor<4xf32>
// CHECK-NEXT: return %[[V4]] : tensor<4xf32>
// CHECK-NEXT: }

// The BP5 grid-stride bound: a loop over lanes carrying a maximum, its body
// repeating max(v, .) on the carry, absorbed to one max: the carry is then
// one elementwise reduction, which the list lifts to a reduce.
func.func @lane_max(%v: tensor<16xi32>) -> tensor<i32> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %n = stablehlo.constant dense<16> : tensor<i64>
  %init = stablehlo.constant dense<-2147483648> : tensor<i32>
  %r:2 = stablehlo.while(%i = %c0, %acc = %init) : tensor<i64>, tensor<i32>
  cond {
    %lt = stablehlo.compare LT, %i, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %lt : tensor<i1>
  } do {
    %ds = stablehlo.dynamic_slice %v, %i, sizes = [1] : (tensor<16xi32>, tensor<i64>) -> tensor<1xi32>
    %e = stablehlo.reshape %ds : (tensor<1xi32>) -> tensor<i32>
    %m0 = stablehlo.maximum %e, %acc : tensor<i32>
    %m1 = stablehlo.maximum %e, %m0 : tensor<i32>
    %m2 = stablehlo.maximum %e, %m1 : tensor<i32>
    %next = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %next, %m2 : tensor<i64>, tensor<i32>
  }
  return %r#1 : tensor<i32>
}

// CHECK-LABEL: func.func @lane_max(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<16xi32>) -> tensor<i32> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<16> : tensor<i64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.constant dense<-2147483648> : tensor<i32>
// CHECK-NEXT: %[[V5:[^ ,:)]+]]:2 = stablehlo.while(%[[V6:[^ ,:)]+]] = %[[V1]], %[[V7:[^ ,:)]+]] = %[[V4]]) : tensor<i64>, tensor<i32>
// CHECK-NEXT: cond {
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.compare LT, %[[V6]], %[[V3]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT: stablehlo.return %[[V8]] : tensor<i1>
// CHECK-NEXT: } do {
// CHECK-NEXT: %[[V8]] = stablehlo.dynamic_slice %[[V0]], %[[V6]], sizes = [1] : (tensor<16xi32>, tensor<i64>) -> tensor<1xi32>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.reshape %[[V8]] : (tensor<1xi32>) -> tensor<i32>
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = stablehlo.maximum %[[V9]], %[[V7]] : tensor<i32>
// CHECK-NEXT: %[[V11:[^ ,:)]+]] = stablehlo.add %[[V6]], %[[V2]] : tensor<i64>
// CHECK-NEXT: stablehlo.return %[[V11]], %[[V10]] : tensor<i64>, tensor<i32>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[V5]]#1 : tensor<i32>
// CHECK-NEXT: }
