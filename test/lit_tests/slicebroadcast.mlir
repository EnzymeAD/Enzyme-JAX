// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {

  func.func @main(%a : tensor<2x3x50xf32>) -> tensor<4x1x25x15x2x3xf32> {
    %bc = stablehlo.broadcast_in_dim %a, dims = [4, 5, 2] : (tensor<2x3x50xf32>) -> tensor<7x11x50x15x2x3xf32>
    %add = stablehlo.slice %bc [0:4, 7:9:2, 0:50:2, 0:15, 0:2, 0:3] : (tensor<7x11x50x15x2x3xf32>) -> tensor<4x1x25x15x2x3xf32>
    return %add : tensor<4x1x25x15x2x3xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<2x3x50xf32>) -> tensor<4x1x25x15x2x3xf32> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:2, 0:3, 0:50:2] : (tensor<2x3x50xf32>) -> tensor<2x3x25xf32>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %0, dims = [4, 5, 2] : (tensor<2x3x25xf32>) -> tensor<4x1x25x15x2x3xf32>
// CHECK-NEXT:    return %1 : tensor<4x1x25x15x2x3xf32>
// CHECK-NEXT:  }
