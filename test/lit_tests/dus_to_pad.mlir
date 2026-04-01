// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main1(%arg0: tensor<f32>) -> tensor<5x3x4xf32> attributes {enzymexla.memory_effects = []} {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<5x3x4xf32>
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x1x4xf32>
    %1 = stablehlo.dynamic_update_slice %cst, %0, %c_0, %c_0, %c : (tensor<5x3x4xf32>, tensor<2x1x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x3x4xf32>
    return %1 : tensor<5x3x4xf32>
}

// CHECK: func.func @main1(%arg0: tensor<f32>) -> tensor<5x3x4xf32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x1x4xf32>
// CHECK-NEXT:     %1 = stablehlo.pad %0, %cst, low = [1, 1, 0], high = [2, 1, 0], interior = [0, 0, 0] : (tensor<2x1x4xf32>, tensor<f32>) -> tensor<5x3x4xf32>
// CHECK-NEXT:     return %1 : tensor<5x3x4xf32>
// CHECK-NEXT: }

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

// CHECK: stablehlo.dynamic_update_slice

func.func @main3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<5x3x4xf32> attributes {enzymexla.memory_effects = []} {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<5x3x4xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<2x1x4xf32>
    %2 = stablehlo.dynamic_update_slice %0, %1, %c_0, %c_0, %c : (tensor<5x3x4xf32>, tensor<2x1x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x3x4xf32>
    return %2 : tensor<5x3x4xf32>
}

// CHECK: func.func @main3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<5x3x4xf32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:     %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<2x1x4xf32>
// CHECK-NEXT:     %1 = stablehlo.pad %0, %arg0, low = [1, 1, 0], high = [2, 1, 0], interior = [0, 0, 0] : (tensor<2x1x4xf32>, tensor<f32>) -> tensor<5x3x4xf32>
// CHECK-NEXT:     return %1 : tensor<5x3x4xf32>
// CHECK-NEXT: }

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

// CHECK: stablehlo.dynamic_update_slice

func.func @main5(%arg0: tensor<2x2x3xf32>) -> tensor<2x4x3xf32> attributes {enzymexla.memory_effects = []} {
    %c = stablehlo.constant dense<-3> : tensor<i32>
    %c_0 = stablehlo.constant dense<-1> : tensor<i32>
    %c_1 = stablehlo.constant dense<1> : tensor<i32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<2x4x3xf32>
    %0 = stablehlo.dynamic_update_slice %cst, %arg0, %c_1, %c_0, %c : (tensor<2x4x3xf32>, tensor<2x2x3xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x4x3xf32>
    return %0 : tensor<2x4x3xf32>
}

// CHECK: func.func @main5(%arg0: tensor<2x2x3xf32>) -> tensor<2x4x3xf32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 2, 0], interior = [0, 0, 0] : (tensor<2x2x3xf32>, tensor<f32>) -> tensor<2x4x3xf32>
// CHECK-NEXT:     return %0 : tensor<2x4x3xf32>
// CHECK-NEXT: }
