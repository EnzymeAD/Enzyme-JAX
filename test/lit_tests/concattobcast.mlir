// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%a : tensor<3x4xf32>) -> tensor<9x4xf32> {
    %concat = stablehlo.concatenate %a, %a, %a, dim=0 : (tensor<3x4xf32>, tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<9x4xf32>
    return %concat : tensor<9x4xf32>
  }
  func.func @main2(%a : tensor<1x4xf32>) -> tensor<3x4xf32> {
    %concat = stablehlo.concatenate %a, %a, %a, dim=0 : (tensor<1x4xf32>, tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
    return %concat : tensor<3x4xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<3x4xf32>) -> tensor<9x4xf32> {
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] : (tensor<3x4xf32>) -> tensor<3x3x4xf32>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<3x3x4xf32>) -> tensor<9x4xf32>
// CHECK-NEXT:    return %1 : tensor<9x4xf32>
// CHECK-NEXT:  }
// CHECK:  func.func @main2(%arg0: tensor<1x4xf32>) -> tensor<3x4xf32> {
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<3x4xf32>
// CHECK-NEXT:    return %0 : tensor<3x4xf32>
// CHECK-NEXT:  }
