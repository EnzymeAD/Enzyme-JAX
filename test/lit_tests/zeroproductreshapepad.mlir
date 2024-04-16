// RUN: enzymexlamlir-opt  --enzyme-hlo-generate-td="patterns=zero_product_reshape_pad<1>;mul_zero_pad<1>;div_zero_pad<1>;" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @pad_multiply(%898: tensor<1x1x4x2048x2048xf32>, %696: tensor<1x4x1x2048x10240xf32>, %901: tensor<1x4x1x2048x10240xf32>) -> (tensor<1x4x1x2048x10240xf32>, tensor<1x4x1x2048x10240xf32>) {
    %cst_192 = stablehlo.constant dense<0.000000e+00> : tensor<f32> 

    %899 = stablehlo.pad %898, %cst_192, low = [0, 0, 0, 0, 8192], high = [0, 0, 0, 0, 0], interior = [0, 0, 0, 0, 0] : (tensor<1x1x4x2048x2048xf32>, tensor<f32>) -> tensor<1x1x4x2048x10240xf32>
    %900 = stablehlo.reshape %899 : (tensor<1x1x4x2048x10240xf32>) -> tensor<1x4x1x2048x10240xf32> 

    %907 = stablehlo.divide %900, %696 : tensor<1x4x1x2048x10240xf32>

    %902 = stablehlo.multiply %900, %901 : tensor<1x4x1x2048x10240xf32> 
    return %907, %902 : tensor<1x4x1x2048x10240xf32>, tensor<1x4x1x2048x10240xf32>
}

// CHECK:  func.func @pad_multiply(%arg0: tensor<1x1x4x2048x2048xf32>, %arg1: tensor<1x4x1x2048x10240xf32>, %arg2: tensor<1x4x1x2048x10240xf32>) -> (tensor<1x4x1x2048x10240xf32>, tensor<1x4x1x2048x10240xf32>) {
// CHECK-NEXT:    %[[i0:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[i1:.+]] = stablehlo.reshape %arg0 : (tensor<1x1x4x2048x2048xf32>) -> tensor<1x4x1x2048x2048xf32>
// CHECK-NEXT:    %[[i2:.+]] = stablehlo.slice %arg1 [0:1, 0:4, 0:1, 0:2048, 8192:10240] : (tensor<1x4x1x2048x10240xf32>) -> tensor<1x4x1x2048x2048xf32>
// CHECK-NEXT:    %[[i3:.+]] = stablehlo.divide %[[i1]], %[[i2]] : tensor<1x4x1x2048x2048xf32>
// CHECK-NEXT:    %[[i4:.+]] = stablehlo.pad %[[i3]], %[[i0]], low = [0, 0, 0, 0, 8192], high = [0, 0, 0, 0, 0], interior = [0, 0, 0, 0, 0] : (tensor<1x4x1x2048x2048xf32>, tensor<f32>) -> tensor<1x4x1x2048x10240xf32>
// CHECK-NEXT:    %[[i5:.+]] = stablehlo.slice %arg2 [0:1, 0:4, 0:1, 0:2048, 8192:10240] : (tensor<1x4x1x2048x10240xf32>) -> tensor<1x4x1x2048x2048xf32>
// CHECK-NEXT:    %[[i6:.+]] = stablehlo.multiply %[[i1]], %[[i5]] : tensor<1x4x1x2048x2048xf32>
// CHECK-NEXT:    %[[i7:.+]] = stablehlo.pad %[[i6]], %[[i0]], low = [0, 0, 0, 0, 8192], high = [0, 0, 0, 0, 0], interior = [0, 0, 0, 0, 0] : (tensor<1x4x1x2048x2048xf32>, tensor<f32>) -> tensor<1x4x1x2048x10240xf32>
// CHECK-NEXT:    return %[[i4]], %[[i7]] : tensor<1x4x1x2048x10240xf32>, tensor<1x4x1x2048x10240xf32>
// CHECK-NEXT:  }
