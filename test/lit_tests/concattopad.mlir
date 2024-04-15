// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {

  func.func @main(%a : tensor<3x4xf32>, %b : tensor<5x4xf32>) -> tensor<10x4xf32> {
    %c = stablehlo.constant dense<3.140000e+00> : tensor<2x4xf32>
    %concat = stablehlo.concatenate %c, %a, %b, dim=0 : (tensor<2x4xf32>, tensor<3x4xf32>, tensor<5x4xf32>) -> tensor<10x4xf32>
    return %concat : tensor<10x4xf32>
  }

  func.func @main2(%a : tensor<3x4xf32>, %b : tensor<5x4xf32>) -> tensor<10x4xf32> {
    %c = stablehlo.constant dense<3.140000e+00> : tensor<2x4xf32>
    %concat = stablehlo.concatenate %a, %b, %c, dim=0 : (tensor<3x4xf32>, tensor<5x4xf32>, tensor<2x4xf32>) -> tensor<10x4xf32>
    return %concat : tensor<10x4xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<3x4xf32>, %arg1: tensor<5x4xf32>) -> tensor<10x4xf32> {
// CHECK-NEXT:    %[[i0:.+]] = stablehlo.constant dense<3.140000e+00> : tensor<f32>
// CHECK-NEXT:    %[[i1:.+]] = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<3x4xf32>, tensor<5x4xf32>) -> tensor<8x4xf32>
// CHECK-NEXT:    %[[i2:.+]] = stablehlo.pad %[[i1]], %[[i0]], low = [2, 0], high = [0, 0], interior = [0, 0] : (tensor<8x4xf32>, tensor<f32>) -> tensor<10x4xf32>
// CHECK-NEXT:    return %[[i2]] : tensor<10x4xf32>
// CHECK-NEXT:  }

// CHECK:  func.func @main2(%arg0: tensor<3x4xf32>, %arg1: tensor<5x4xf32>) -> tensor<10x4xf32> {
// CHECK-NEXT:    %[[i0:.+]] = stablehlo.constant dense<3.140000e+00> : tensor<f32>
// CHECK-NEXT:    %[[i1:.+]] = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<3x4xf32>, tensor<5x4xf32>) -> tensor<8x4xf32>
// CHECK-NEXT:    %[[i2:.+]] = stablehlo.pad %[[i1]], %[[i0]], low = [0, 0], high = [2, 0], interior = [0, 0] : (tensor<8x4xf32>, tensor<f32>) -> tensor<10x4xf32>
// CHECK-NEXT:    return %[[i2]] : tensor<10x4xf32>
// CHECK-NEXT:  }
