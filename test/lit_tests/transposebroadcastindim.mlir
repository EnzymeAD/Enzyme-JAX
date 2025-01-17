// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<4x12xf32>) -> tensor<12x3x4xf32> {
    %1 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x12xf32>) -> tensor<12x4xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 2] : (tensor<12x4xf32>) -> tensor<12x3x4xf32>
    return %2 : tensor<12x3x4xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<4x12xf32>) -> tensor<12x3x4xf32> {
// CHECK-NEXT: %0 = stablehlo.broadcast_in_dim %arg0, dims = [2, 0] : (tensor<4x12xf32>) -> tensor<12x3x4xf32>
// CHECK-NEXT:     return %0 : tensor<12x3x4xf32>
// CHECK-NEXT: }
