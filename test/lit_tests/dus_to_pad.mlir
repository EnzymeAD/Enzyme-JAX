// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main1(%arg0: tensor<f32>) -> tensor<5x3x4xf32> attributes {enzymexla.memory_effects = []} {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<5x3x4xf32>
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x1x4xf32>
    %1 = stablehlo.dynamic_update_slice %cst, %0, %c_0, %c_0, %c : (tensor<5x3x4xf32>, tensor<2x1x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x3x4xf32>
    return %1 : tensor<5x3x4xf32>
}

func.func @main2(%arg0: tensor<f32>, %arg1: tensor<i64>) -> tensor<5x3x4xf32> attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<5x3x4xf32>
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
    %1 = stablehlo.subtract %0, %c : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x1x4xf32>
    %3 = stablehlo.dynamic_update_slice %cst, %2, %c, %1, %c_0 : (tensor<5x3x4xf32>, tensor<2x1x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x3x4xf32>
    return %3 : tensor<5x3x4xf32>
}

func.func @main3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<5x3x4xf32> attributes {enzymexla.memory_effects = []} {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<5x3x4xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<2x1x4xf32>
    %2 = stablehlo.dynamic_update_slice %0, %1, %c_0, %c_0, %c : (tensor<5x3x4xf32>, tensor<2x1x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x3x4xf32>
    return %2 : tensor<5x3x4xf32>
}

func.func @main4(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i64>) -> tensor<5x3x4xf32> attributes {enzymexla.memory_effects = []} {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.convert %arg2 : (tensor<i64>) -> tensor<i32>
    %1 = stablehlo.subtract %0, %c : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<5x3x4xf32>
    %3 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<2x1x4xf32>
    %4 = stablehlo.dynamic_update_slice %2, %3, %c, %1, %c_0 : (tensor<5x3x4xf32>, tensor<2x1x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x3x4xf32>
    return %4 : tensor<5x3x4xf32>
}

// TODO: test for clamped update
