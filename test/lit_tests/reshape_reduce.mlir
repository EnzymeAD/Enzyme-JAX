// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main1(%arg0: tensor<128x128x2x1xf32>) -> tensor<128x1xf32> {
    %cst = stablehlo.constant dense<0.0> : tensor<f32>
    %0 = stablehlo.reshape %arg0 : (tensor<128x128x2x1xf32>) -> tensor<128x256x1xf32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x256x1xf32>, tensor<f32>) -> tensor<128x1xf32>
    return %1 : tensor<128x1xf32>
}

// CHECK: func.func @main1(%arg0: tensor<128x128x2x1xf32>) -> tensor<128x1xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<128x128x2x1xf32>, tensor<f32>) -> tensor<128x1xf32>
// CHECK-NEXT:     return %0 : tensor<128x1xf32>
// CHECK-NEXT: }

// shouldn't apply here
func.func @main2_fail(%arg0: tensor<128x128x2x6x5xf32>) -> tensor<128x1x2x3x5xf32> {
    %cst = stablehlo.constant dense<0.0> : tensor<f32>
    %0 = stablehlo.reshape %arg0 : (tensor<128x128x2x6x5xf32>) -> tensor<128x256x1x2x3x5xf32>
    // CHECK: %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x256x1x2x3x5xf32>, tensor<f32>) -> tensor<128x1x2x3x5xf32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x256x1x2x3x5xf32>, tensor<f32>) -> tensor<128x1x2x3x5xf32>
    return %1 : tensor<128x1x2x3x5xf32>
}

func.func @main3(%arg0: tensor<128x128x2x6x5xf32>) -> tensor<128x2x5xf32> {
    %cst = stablehlo.constant dense<0.0> : tensor<f32>
    %0 = stablehlo.reshape %arg0 : (tensor<128x128x2x6x5xf32>) -> tensor<128x128x2x2x3x5xf32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 3, 4] : (tensor<128x128x2x2x3x5xf32>, tensor<f32>) -> tensor<128x2x5xf32>
    return %1 : tensor<128x2x5xf32>
}

// CHECK: func.func @main3(%arg0: tensor<128x128x2x6x5xf32>) -> tensor<128x2x5xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1, 3] : (tensor<128x128x2x6x5xf32>, tensor<f32>) -> tensor<128x2x5xf32>
// CHECK-NEXT:     return %0 : tensor<128x2x5xf32>
// CHECK-NEXT: }
