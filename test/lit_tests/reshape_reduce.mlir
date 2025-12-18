// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main(%arg0: tensor<128x128x2x1xf32>) -> tensor<128x1xf32> {
    %cst = stablehlo.constant dense<0.0> : tensor<f32>
    %0 = stablehlo.reshape %arg0 : (tensor<128x128x2x1xf32>) -> tensor<128x256x1xf32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x256x1xf32>, tensor<f32>) -> tensor<128x1xf32>
    return %1 : tensor<128x1xf32>
}

func.func @main(%arg0: tensor<128x128x2x6x5xf32>) -> tensor<128x1x2x3x5xf32> {
    %cst = stablehlo.constant dense<0.0> : tensor<f32>
    %0 = stablehlo.reshape %arg0 : (tensor<128x128x2x6x5xf32>) -> tensor<128x256x1x2x3x5xf32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x256x1x2x3x5xf32>, tensor<f32>) -> tensor<128x1x2x3x5xf32>
    return %1 : tensor<128x1x2x3x5xf32>
}
