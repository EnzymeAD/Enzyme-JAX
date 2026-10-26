// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=reshape_dynamic_slice(1)" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// A reshape of a dynamic slice that adds a leading dimension and drops a
// trailing one whose start index is not known to be zero. The pattern fills
// a zero index for the added dimension and then declines at the dropped
// one. It used to create that zero before deciding, so every attempt left a
// constant behind for the driver to revisit, and this never finished.
func.func @main(%x: tensor<8x3xf32>, %i: tensor<i64>, %j: tensor<i64>) -> tensor<1x5xf32> {
  %s = stablehlo.dynamic_slice %x, %i, %j, sizes = [5, 1] : (tensor<8x3xf32>, tensor<i64>, tensor<i64>) -> tensor<5x1xf32>
  %r = stablehlo.reshape %s : (tensor<5x1xf32>) -> tensor<1x5xf32>
  return %r : tensor<1x5xf32>
}

// CHECK:  func.func @main(%arg0: tensor<8x3xf32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x5xf32> {
// CHECK-NEXT:    %0 = stablehlo.dynamic_slice %arg0, %arg1, %arg2, sizes = [5, 1] : (tensor<8x3xf32>, tensor<i64>, tensor<i64>) -> tensor<5x1xf32>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<5x1xf32>) -> tensor<1x5xf32>
// CHECK-NEXT:    return %1 : tensor<1x5xf32>
// CHECK-NEXT:  }
