// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=add_reduce_slice_fusion" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

func.func @main_partial_add(%arg0: tensor<16x4xf64>) -> tensor<1x3xf64> {
    %0 = stablehlo.slice %arg0 [0:1, 1:4] : (tensor<16x4xf64>) -> tensor<1x3xf64>
    %1 = stablehlo.slice %arg0 [1:2, 1:4] : (tensor<16x4xf64>) -> tensor<1x3xf64>
    %2 = stablehlo.slice %arg0 [2:3, 1:4] : (tensor<16x4xf64>) -> tensor<1x3xf64>
    %3 = stablehlo.add %0, %1 : tensor<1x3xf64>
    %4 = stablehlo.add %3, %2 : tensor<1x3xf64>
    return %4 : tensor<1x3xf64>
}

// CHECK: func.func @main_partial_add(%arg0: tensor<16x4xf64>) -> tensor<1x3xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.slice %arg0 [0:3, 1:4] : (tensor<16x4xf64>) -> tensor<3x3xf64>
// CHECK-NEXT:     %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3x3xf64>, tensor<f64>) -> tensor<3xf64>
// CHECK-NEXT:     %2 = stablehlo.reshape %1 : (tensor<3xf64>) -> tensor<1x3xf64>
// CHECK-NEXT:     return %2 : tensor<1x3xf64>
// CHECK-NEXT: }
