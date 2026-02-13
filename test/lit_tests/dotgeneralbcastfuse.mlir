// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<32x64xf32>) -> tensor<32x32x1x1xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2] : (tensor<32x64xf32>) -> tensor<32x32x64x1x1xf32>
    %1 = stablehlo.dot_general %0, %arg0, batching_dims = [1] x [0], contracting_dims = [2] x [1] : (tensor<32x32x64x1x1xf32>, tensor<32x64xf32>) -> tensor<32x32x1x1xf32>
    %2 = stablehlo.transpose %1, dims = [1, 0, 2, 3] : (tensor<32x32x1x1xf32>) -> tensor<32x32x1x1xf32>
    return %2 : tensor<32x32x1x1xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<32x64xf32>) -> tensor<32x32x1x1xf32> {
// CHECK-NEXT:     %0 = stablehlo.dot_general %arg0, %arg0, contracting_dims = [1] x [1] : (tensor<32x64xf32>, tensor<32x64xf32>) -> tensor<32x32xf32>
// CHECK-NEXT:     %1 = stablehlo.reshape %0 : (tensor<32x32xf32>) -> tensor<32x32x1x1xf32>
// CHECK-NEXT:     return %1 : tensor<32x32x1x1xf32>
// CHECK-NEXT: }
