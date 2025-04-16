// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{passses=65536})" %s | FileCheck %s

func.func @transpose_dynamic_slice(%arg0: tensor<12x16x4xf32>, %arg2: tensor<i64>) -> tensor<12x1x4xf32> {
    %18 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<12x16x4xf32>) -> tensor<4x16x12xf32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %21 = stablehlo.convert %arg2 : (tensor<i64>) -> tensor<i32>
    %22 = stablehlo.dynamic_slice %18, %c, %21, %c, sizes = [4, 1, 12] : (tensor<4x16x12xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x1x12xf32>
    %23 = stablehlo.transpose %22, dims = [2, 1, 0] : (tensor<4x1x12xf32>) -> tensor<12x1x4xf32>
    return %23 : tensor<12x1x4xf32>
}

// CHECK: func.func @transpose_dynamic_slice(%arg0: tensor<12x16x4xf32>, %arg1: tensor<i64>) -> tensor<12x1x4xf32> {
// CHECK-NEXT:     %c = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<12x16x4xf32>) -> tensor<4x16x12xf32>
// CHECK-NEXT:     %1 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:     %2 = stablehlo.transpose %0, dims = [2, 1, 0] : (tensor<4x16x12xf32>) -> tensor<12x16x4xf32>
// CHECK-NEXT:     %3 = stablehlo.dynamic_slice %2, %c, %1, %c, sizes = [12, 1, 4] : (tensor<12x16x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<12x1x4xf32>
// CHECK-NEXT:     return %3 : tensor<12x1x4xf32>
// CHECK-NEXT: }
