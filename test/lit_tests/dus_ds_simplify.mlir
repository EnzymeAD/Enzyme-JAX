// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @full_overlap(%arg0: tensor<4x3xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> (tensor<4x3xf32>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<32x32xf32>
    %0 = stablehlo.dynamic_update_slice %cst, %arg0, %arg1, %arg2 : (tensor<32x32xf32>, tensor<4x3xf32>, tensor<i32>, tensor<i32>) -> tensor<32x32xf32>
    %1 = stablehlo.dynamic_slice %0, %arg1, %arg2, sizes = [4, 3] : (tensor<32x32xf32>, tensor<i32>, tensor<i32>) -> tensor<4x3xf32>
    return %1 : tensor<4x3xf32>
}

// CHECK: func.func @full_overlap(%arg0: tensor<4x3xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<4x3xf32> {
// CHECK-NEXT:     return %arg0 : tensor<4x3xf32>
// CHECK-NEXT: }
