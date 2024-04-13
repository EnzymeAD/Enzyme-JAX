// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {

  func.func @main(%a : tensor<f32>) -> tensor<1x2x1x1xf32> {
    %bc = stablehlo.broadcast_in_dim %a, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %conv = stablehlo.reshape %bc : (tensor<2xf32>) -> tensor<1x2x1x1xf32>
    return %conv : tensor<1x2x1x1xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<f32>) -> tensor<1x2x1x1xf32> {
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<1x2x1x1xf32>
// CHECK-NEXT:    return %0 : tensor<1x2x1x1xf32>
// CHECK-NEXT:  }
