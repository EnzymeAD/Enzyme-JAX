// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=concat_reshape_elementwise},transform-interpreter,enzyme-hlo-remove-transform,enzyme-hlo-opt)" %s | FileCheck %s

func.func @main(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x2x4xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<3x4xf32>
    %1 = stablehlo.add %arg1, %arg2 : tensor<3x4xf32>
    %2 = stablehlo.reshape %0 : (tensor<3x4xf32>) -> tensor<3x1x4xf32>
    %3 = stablehlo.reshape %1 : (tensor<3x4xf32>) -> tensor<3x1x4xf32>
    %4 = stablehlo.concatenate %2, %3, dim = 1 : (tensor<3x1x4xf32>, tensor<3x1x4xf32>) -> tensor<3x2x4xf32>
    return %4 : tensor<3x2x4xf32>
}

// CHECK: func.func @main(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x2x4xf32> {
// CHECK-NEXT:     %0 = stablehlo.reshape %arg0 : (tensor<3x4xf32>) -> tensor<3x1x4xf32>
// CHECK-NEXT:     %1 = stablehlo.reshape %arg1 : (tensor<3x4xf32>) -> tensor<3x1x4xf32>
// CHECK-NEXT:     %2 = stablehlo.concatenate %0, %1, dim = 1 : (tensor<3x1x4xf32>, tensor<3x1x4xf32>) -> tensor<3x2x4xf32>
// CHECK-NEXT:     %3 = stablehlo.reshape %arg2 : (tensor<3x4xf32>) -> tensor<3x1x4xf32>
// CHECK-NEXT:     %4 = stablehlo.concatenate %1, %3, dim = 1 : (tensor<3x1x4xf32>, tensor<3x1x4xf32>) -> tensor<3x2x4xf32>
// CHECK-NEXT:     %5 = stablehlo.add %2, %4 : tensor<3x2x4xf32>
// CHECK-NEXT:     return %5 : tensor<3x2x4xf32>
// CHECK-NEXT: }
