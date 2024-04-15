// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {

  func.func @main(%a : tensor<2xf32>, %b : tensor<1xf32>, %c : tensor<1xf32>) -> tensor<f32> {
    %cst0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %concat = stablehlo.concatenate %a, %b, %c, dim=0 : (tensor<2xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<4xf32>

    %1308 = stablehlo.reduce(%concat init: %cst0) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>

    return %1308 : tensor<f32>

  }

  func.func @main2(%a : tensor<2x2x3xf32>, %b : tensor<1x2x3xf32>, %c : tensor<1x2x3xf32>) -> tensor<2x3xf32> {
    %cst0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %concat = stablehlo.concatenate %a, %b, %c, dim=0 : (tensor<2x2x3xf32>, tensor<1x2x3xf32>, tensor<1x2x3xf32>) -> tensor<4x2x3xf32>

    %1308 = stablehlo.reduce(%concat init: %cst0) applies stablehlo.add across dimensions = [0] : (tensor<4x2x3xf32>, tensor<f32>) -> tensor<2x3xf32>

    return %1308 : tensor<2x3xf32>

  }
}

// CHECK:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> tensor<f32> {
// CHECK-NEXT:    %[[cst:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[i0:.+]] = stablehlo.reduce(%arg0 init: %[[cst]]) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:    %[[i1:.+]] = stablehlo.reshape %arg1 : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT:    %[[i2:.+]] = stablehlo.add %[[i0]], %[[i1]] : tensor<f32>
// CHECK-NEXT:    %[[i3:.+]] = stablehlo.reshape %arg2 : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT:    %[[i4:.+]] = stablehlo.add %[[i2]], %[[i3]] : tensor<f32>
// CHECK-NEXT:    return %[[i4]] : tensor<f32>
// CHECK-NEXT:  }

// CHECK:  func.func @main2(%arg0: tensor<2x2x3xf32>, %arg1: tensor<1x2x3xf32>, %arg2: tensor<1x2x3xf32>) -> tensor<2x3xf32> {
// CHECK-NEXT:    %[[i0:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[i1:.+]] = stablehlo.reduce(%arg0 init: %[[i0]]) applies stablehlo.add across dimensions = [0] : (tensor<2x2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
// CHECK-NEXT:    %[[i2:.+]] = stablehlo.reshape %arg1 : (tensor<1x2x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:    %[[i3:.+]] = stablehlo.add %[[i1]], %[[i2]] : tensor<2x3xf32>
// CHECK-NEXT:    %[[i4:.+]] = stablehlo.reshape %arg2 : (tensor<1x2x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:    %[[i5:.+]] = stablehlo.add %[[i3]], %[[i4]] : tensor<2x3xf32>
// CHECK-NEXT:    return %[[i5]] : tensor<2x3xf32>
// CHECK-NEXT:  }
