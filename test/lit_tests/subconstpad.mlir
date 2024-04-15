// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @sub_pad_1(%arg0: tensor<1x3x1024xf32>, %arg1: tensor<f32>) -> tensor<1x3x2048xf32> {
  %0 = stablehlo.constant dense<3.600000e+00> : tensor<1x3x2048xf32>
  %1 = stablehlo.pad %arg0, %arg1, low = [0, 0, 1024], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x3x1024xf32>, tensor<f32>) -> tensor<1x3x2048xf32>
  %2 = stablehlo.subtract %0, %1 : tensor<1x3x2048xf32>
  return %2 : tensor<1x3x2048xf32>
}

func.func @sub_pad_2(%arg0: tensor<1x3x1024xf32>, %arg1: tensor<f32>) -> tensor<1x3x2048xf32> {
  %0 = stablehlo.constant dense<3.600000e+00> : tensor<1x3x2048xf32>
  %1 = stablehlo.pad %arg0, %arg1, low = [0, 0, 1024], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x3x1024xf32>, tensor<f32>) -> tensor<1x3x2048xf32>
  %2 = stablehlo.subtract %1, %0 : tensor<1x3x2048xf32>
  return %2 : tensor<1x3x2048xf32>
}


// CHECK:  func.func @sub_pad_1(%arg0: tensor<1x3x1024xf32>, %arg1: tensor<f32>) -> tensor<1x3x2048xf32> {
// CHECK-NEXT:    %[[i0:.+]] = stablehlo.constant dense<3.600000e+00> : tensor<1x3x1024xf32>
// CHECK-NEXT:    %[[i1:.+]] = stablehlo.constant dense<3.600000e+00> : tensor<f32>
// CHECK-NEXT:    %[[i2:.+]] = stablehlo.subtract %[[i1]], %arg1 : tensor<f32>
// CHECK-NEXT:    %[[i3:.+]] = stablehlo.subtract %[[i0]], %arg0 : tensor<1x3x1024xf32>
// CHECK-NEXT:    %[[i4:.+]] = stablehlo.pad %[[i3]], %[[i2]], low = [0, 0, 1024], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x3x1024xf32>, tensor<f32>) -> tensor<1x3x2048xf32>
// CHECK-NEXT:    return %[[i4]] : tensor<1x3x2048xf32>
// CHECK-NEXT:  }
// CHECK:  func.func @sub_pad_2(%arg0: tensor<1x3x1024xf32>, %arg1: tensor<f32>) -> tensor<1x3x2048xf32> {
// CHECK-NEXT:    %[[i0:.+]] = stablehlo.constant dense<3.600000e+00> : tensor<1x3x1024xf32>
// CHECK-NEXT:    %[[i1:.+]] = stablehlo.constant dense<3.600000e+00> : tensor<f32>
// CHECK-NEXT:    %[[i2:.+]] = stablehlo.subtract %arg1, %[[i1]] : tensor<f32>
// CHECK-NEXT:    %[[i3:.+]] = stablehlo.subtract %arg0, %[[i0]] : tensor<1x3x1024xf32>
// CHECK-NEXT:    %[[i4:.+]] = stablehlo.pad %[[i3]], %[[i2]], low = [0, 0, 1024], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x3x1024xf32>, tensor<f32>) -> tensor<1x3x2048xf32>
// CHECK-NEXT:    return %[[i4]] : tensor<1x3x2048xf32>
// CHECK-NEXT:  }
