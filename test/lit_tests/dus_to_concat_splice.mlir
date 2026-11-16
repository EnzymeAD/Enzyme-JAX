// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=dynamic_update_to_concat" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// An update into a concatenate along its dim: the parts around the update
// are taken from the concatenate's own pieces, whole ones as they are.
func.func @into_concat(%a: tensor<4x3xf64>, %b: tensor<4x1xf64>, %c: tensor<4x4xf64>, %u: tensor<4x1xf64>) -> tensor<4x8xf64> {
  %0 = stablehlo.concatenate %a, %b, %c, dim = 1 : (tensor<4x3xf64>, tensor<4x1xf64>, tensor<4x4xf64>) -> tensor<4x8xf64>
  %i0 = stablehlo.constant dense<0> : tensor<i64>
  %i2 = stablehlo.constant dense<2> : tensor<i64>
  %1 = stablehlo.dynamic_update_slice %0, %u, %i0, %i2 : (tensor<4x8xf64>, tensor<4x1xf64>, tensor<i64>, tensor<i64>) -> tensor<4x8xf64>
  return %1 : tensor<4x8xf64>
}

// CHECK:  func.func @into_concat(%arg0: tensor<4x3xf64>, %arg1: tensor<4x1xf64>, %arg2: tensor<4x4xf64>, %arg3: tensor<4x1xf64>) -> tensor<4x8xf64> {
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [0:4, 0:2] : (tensor<4x3xf64>) -> tensor<4x2xf64>
// CHECK-NEXT:   %1 = stablehlo.concatenate %0, %arg3, %arg1, %arg2, dim = 1 : (tensor<4x2xf64>, tensor<4x1xf64>, tensor<4x1xf64>, tensor<4x4xf64>) -> tensor<4x8xf64>
// CHECK-NEXT:   return %1 : tensor<4x8xf64>
// CHECK-NEXT: }

// An update into a value that is no concatenate: slices of it.
func.func @into_value(%x: tensor<4x8xf64>, %u: tensor<4x1xf64>) -> tensor<4x8xf64> {
  %i0 = stablehlo.constant dense<0> : tensor<i64>
  %i2 = stablehlo.constant dense<2> : tensor<i64>
  %0 = stablehlo.dynamic_update_slice %x, %u, %i0, %i2 : (tensor<4x8xf64>, tensor<4x1xf64>, tensor<i64>, tensor<i64>) -> tensor<4x8xf64>
  return %0 : tensor<4x8xf64>
}

// CHECK:  func.func @into_value(%arg0: tensor<4x8xf64>, %arg1: tensor<4x1xf64>) -> tensor<4x8xf64> {
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [0:4, 0:2] : (tensor<4x8xf64>) -> tensor<4x2xf64>
// CHECK-NEXT:   %1 = stablehlo.slice %arg0 [0:4, 3:8] : (tensor<4x8xf64>) -> tensor<4x5xf64>
// CHECK-NEXT:   %2 = stablehlo.concatenate %0, %arg1, %1, dim = 1 : (tensor<4x2xf64>, tensor<4x1xf64>, tensor<4x5xf64>) -> tensor<4x8xf64>
// CHECK-NEXT:   return %2 : tensor<4x8xf64>
// CHECK-NEXT: }
