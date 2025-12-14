// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
    func.func @main(%arg0: tensor<f32>, %arg1: tensor<i32>) -> tensor<2x3x4xf32> {
        %c = stablehlo.constant dense<0> : tensor<i32>
        %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<7x8x9xf32>
        %1 = stablehlo.dynamic_slice %0, %c, %arg1, %c, sizes = [2, 3, 4] : (tensor<7x8x9xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x3x4xf32>
        return %1 : tensor<2x3x4xf32>
    }
}

// CHECK: func.func @main(%arg0: tensor<f32>, %arg1: tensor<i32>) -> tensor<2x3x4xf32> {
// CHECK-NEXT:     %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x3x4xf32>
// CHECK-NEXT:     return %0 : tensor<2x3x4xf32>
// CHECK-NEXT: }

module {
    func.func @main(%arg1: tensor<f32>, %arg2: tensor<i32>) -> tensor<2x3x4xf32> {
        %c = stablehlo.constant dense<0> : tensor<i32>
        %cst = stablehlo.constant dense<1.000000e+01> : tensor<7x8x9xf32>
        %0 = stablehlo.dynamic_slice %cst, %c, %arg2, %c, sizes = [2, 3, 4] : (tensor<7x8x9xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x3x4xf32>
        return %0 : tensor<2x3x4xf32>
    }
}

// CHECK: func.func @main(%arg0: tensor<f32>, %arg1: tensor<i32>) -> tensor<2x3x4xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.000000e+01> : tensor<2x3x4xf32>
// CHECK-NEXT:     return %cst : tensor<2x3x4xf32>
// CHECK-NEXT: }
