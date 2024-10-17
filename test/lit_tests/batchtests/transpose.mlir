// RUN: enzymexlamlir-opt %s --enzyme-batch | FileCheck %s
  
func.func private @relu_broadcast_scalar(%arg0: tensor<3x4xf64>) -> (tensor<4x3xf64>) {
    %1 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<3x4xf64>) -> tensor<4x3xf64>
    return %1 : tensor<4x3xf64>
  }
  func.func @main(%arg0: tensor<2x5x3x4xf64>) -> (tensor<2x5x4x3xf64>) {
    %1 = enzyme.batch @relu_broadcast_scalar(%arg0) {batch_shape = array<i64: 2, 5>} : (tensor<2x5x3x4xf64>) -> (tensor<2x5x4x3xf64>)
    return %1 : tensor<2x5x4x3xf64>
  }

// CHECK:  func.func private @batched_relu_broadcast_scalar(%arg0: tensor<2x5x3x4xf64>) -> tensor<2x5x4x3xf64> {
// CHECK-NEXT:    %0 = stablehlo.transpose %arg0, dims = [0, 1, 3, 2] : (tensor<2x5x3x4xf64>) -> tensor<2x5x4x3xf64>
// CHECK-NEXT:    return %0 : tensor<2x5x4x3xf64>
// CHECK-NEXT:  }
