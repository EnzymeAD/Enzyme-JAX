// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

func.func @main1(%arg0: tensor<2x3xf32>, %arg1: tensor<3xf32>) -> tensor<2xf32> {
    %0 = stablehlo.reshape %arg1 : (tensor<3xf32>) -> tensor<3x1xf32>
    %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0] : (tensor<2x3xf32>, tensor<3x1xf32>) -> tensor<2x1xf32>
    %2 = stablehlo.reshape %1 : (tensor<2x1xf32>) -> tensor<2xf32>
    return %2 : tensor<2xf32>
}

// CHECK: func.func @main1(%arg0: tensor<2x3xf32>, %arg1: tensor<3xf32>) -> tensor<2xf32> {
// CHECK-NEXT:     %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2xf32>
// CHECK-NEXT:     return %0 : tensor<2xf32>
// CHECK-NEXT: }

func.func @main2(%arg0: tensor<2x3xf32>, %arg1: tensor<2xf32>) -> tensor<3xf32> {
    %0 = stablehlo.reshape %arg1 : (tensor<2xf32>) -> tensor<1x2x1xf32>
    %1 = stablehlo.dot_general %0, %arg0, contracting_dims = [1] x [0] : (tensor<1x2x1xf32>, tensor<2x3xf32>) -> tensor<1x1x3xf32>
    %2 = stablehlo.reshape %1 : (tensor<1x1x3xf32>) -> tensor<3xf32>
    return %2 : tensor<3xf32>
}

// CHECK: func.func @main2(%arg0: tensor<2x3xf32>, %arg1: tensor<2xf32>) -> tensor<3xf32> {
// CHECK-NEXT:     %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2x3xf32>) -> tensor<3xf32>
// CHECK-NEXT:     return %0 : tensor<3xf32>
// CHECK-NEXT: }

func.func @main3(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<f32> {
    %0 = stablehlo.reshape %arg0 : (tensor<3xf32>) -> tensor<3x1x1xf32>
    %1 = stablehlo.reshape %arg1 : (tensor<3xf32>) -> tensor<1x3x1xf32>
    %2 = stablehlo.dot_general %0, %1, contracting_dims = [0] x [1] : (tensor<3x1x1xf32>, tensor<1x3x1xf32>) -> tensor<1x1x1x1xf32>
    %3 = stablehlo.reshape %2 : (tensor<1x1x1x1xf32>) -> tensor<f32>
    return %3 : tensor<f32>
}

// CHECK: func.func @main3(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<f32> {
// CHECK-NEXT:     %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [0] : (tensor<3xf32>, tensor<3xf32>) -> tensor<f32>
// CHECK-NEXT:     return %0 : tensor<f32>
// CHECK-NEXT: }
