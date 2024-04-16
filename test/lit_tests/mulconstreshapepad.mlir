// RUN: enzymexlamlir-opt  --enzyme-hlo-generate-td="patterns=binop_const_reshape_pad<1>;binop_const_pad_mul<1>;" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @pad_multiply(%arg0: tensor<1x3x1024xf32>, %arg1: tensor<f32>) -> tensor<3x2048xf32> {
  %0 = stablehlo.constant dense<3.600000e+00> : tensor<3x2048xf32>
  %1 = stablehlo.pad %arg0, %arg1, low = [0, 0, 1024], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x3x1024xf32>, tensor<f32>) -> tensor<1x3x2048xf32>
  %z = stablehlo.reshape %1: (tensor<1x3x2048xf32>) -> tensor<3x2048xf32>
  %2 = stablehlo.multiply %0, %z : tensor<3x2048xf32>
  return %2 : tensor<3x2048xf32>
}

// CHECK:   func.func @pad_multiply(%arg0: tensor<1x3x1024xf32>, %arg1: tensor<f32>) -> tensor<3x2048xf32> {
// CHECK-NEXT:     %[[i0:.+]] = stablehlo.constant dense<3.600000e+00> : tensor<3x1024xf32>
// CHECK-NEXT:     %[[i1:.+]] = stablehlo.constant dense<3.600000e+00> : tensor<f32>
// CHECK-NEXT:     %[[i2:.+]] = stablehlo.reshape %arg0 : (tensor<1x3x1024xf32>) -> tensor<3x1024xf32>
// CHECK-NEXT:     %[[i3:.+]] = stablehlo.multiply %[[i1]], %arg1 : tensor<f32>
// CHECK-NEXT:     %[[i4:.+]] = stablehlo.multiply %[[i0]], %[[i2]] : tensor<3x1024xf32>
// CHECK-NEXT:     %[[i5:.+]] = stablehlo.pad %[[i4]], %[[i3]], low = [0, 1024], high = [0, 0], interior = [0, 0] : (tensor<3x1024xf32>, tensor<f32>) -> tensor<3x2048xf32>
// CHECK-NEXT:     return %[[i5]] : tensor<3x2048xf32>
// CHECK-NEXT:   }
