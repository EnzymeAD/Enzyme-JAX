// RUN: enzymexlamlir-opt %s --enzyme-batch | FileCheck %s

func.func @func1(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 1] : (tensor<2x2xi32>) -> tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }
  func.func @main() -> tensor<2x2x2xi32> {
    %c = stablehlo.constant dense<[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]> : tensor<2x2x2xi32>
    %0 = enzyme.batch @func1(%c) {batch_shape = array<i64: 2>} : (tensor<2x2x2xi32>) -> tensor<2x2x2xi32>
    return %0 : tensor<2x2x2xi32>
  }

// CHECK:  func.func private @batched_relu_broadcast_scalar(%arg0: tensor<2x5x3x4xf64>) -> tensor<2x5x4x3xf64> {
// CHECK-NEXT:    %0 = stablehlo.transpose %arg0, dims = [0, 1, 3, 2] : (tensor<2x5x3x4xf64>) -> tensor<2x5x4x3xf64>
// CHECK-NEXT:    return %0 : tensor<2x5x4x3xf64>
// CHECK-NEXT:  }
