// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=recognize_extend" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// A wide concatenate whose only extend window comes after windows of other
// buffers and of pieces wider than one: the extend is still found.
func.func @late(%a: tensor<4x1xf64>, %b: tensor<4x2xf64>, %c: tensor<4x1xf64>, %x: tensor<4x8xf64>) -> tensor<4x14xf64> {
  %0 = stablehlo.slice %x [0:4, 0:1] : (tensor<4x8xf64>) -> tensor<4x1xf64>
  %1 = stablehlo.slice %x [0:4, 7:8] : (tensor<4x8xf64>) -> tensor<4x1xf64>
  %2 = stablehlo.concatenate %a, %b, %c, %0, %x, %1, dim = 1 : (tensor<4x1xf64>, tensor<4x2xf64>, tensor<4x1xf64>, tensor<4x1xf64>, tensor<4x8xf64>, tensor<4x1xf64>) -> tensor<4x14xf64>
  return %2 : tensor<4x14xf64>
}

// CHECK:  func.func @late(%arg0: tensor<4x1xf64>, %arg1: tensor<4x2xf64>, %arg2: tensor<4x1xf64>, %arg3: tensor<4x8xf64>) -> tensor<4x14xf64> {
// CHECK-NEXT:   %0 = "enzymexla.extend"(%arg3) <{dimension = 1 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<4x8xf64>) -> tensor<4x10xf64>
// CHECK-NEXT:   %1 = stablehlo.concatenate %arg0, %arg1, %arg2, %0, dim = 1 : (tensor<4x1xf64>, tensor<4x2xf64>, tensor<4x1xf64>, tensor<4x10xf64>) -> tensor<4x14xf64>
// CHECK-NEXT:   return %1 : tensor<4x14xf64>
// CHECK-NEXT: }

// The pieces around a buffer are of another buffer: no extend.
func.func @other_buffer(%x: tensor<4x8xf64>, %y: tensor<4x8xf64>) -> tensor<4x10xf64> {
  %0 = stablehlo.slice %y [0:4, 0:1] : (tensor<4x8xf64>) -> tensor<4x1xf64>
  %1 = stablehlo.slice %y [0:4, 7:8] : (tensor<4x8xf64>) -> tensor<4x1xf64>
  %2 = stablehlo.concatenate %0, %x, %1, dim = 1 : (tensor<4x1xf64>, tensor<4x8xf64>, tensor<4x1xf64>) -> tensor<4x10xf64>
  return %2 : tensor<4x10xf64>
}

// CHECK:  func.func @other_buffer(%arg0: tensor<4x8xf64>, %arg1: tensor<4x8xf64>) -> tensor<4x10xf64> {
// CHECK-NEXT:   %0 = stablehlo.slice %arg1 [0:4, 0:1] : (tensor<4x8xf64>) -> tensor<4x1xf64>
// CHECK-NEXT:   %1 = stablehlo.slice %arg1 [0:4, 7:8] : (tensor<4x8xf64>) -> tensor<4x1xf64>
// CHECK-NEXT:   %2 = stablehlo.concatenate %0, %arg0, %1, dim = 1 : (tensor<4x1xf64>, tensor<4x8xf64>, tensor<4x1xf64>) -> tensor<4x10xf64>
// CHECK-NEXT:   return %2 : tensor<4x10xf64>
// CHECK-NEXT: }
