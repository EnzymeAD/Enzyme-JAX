// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=elementwise_reshape_like},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

func.func @main(%arg0: tensor<3x2xf32>, %arg1: tensor<3x2xf32>) -> tensor<3x1x2xf32> {
    %0 = stablehlo.reshape %arg0 : (tensor<3x2xf32>) -> tensor<3x1x2xf32>
    %1 = stablehlo.reshape %arg1 : (tensor<3x2xf32>) -> tensor<3x1x2xf32>
    %2 = stablehlo.add %0, %1 : tensor<3x1x2xf32>
    return %2 : tensor<3x1x2xf32>
}

// CHECK: func.func @main(%arg0: tensor<3x2xf32>, %arg1: tensor<3x2xf32>) -> tensor<3x1x2xf32> {
// CHECK-NEXT:     %0 = stablehlo.add %arg0, %arg1 : tensor<3x2xf32>
// CHECK-NEXT:     %1 = stablehlo.reshape %0 : (tensor<3x2xf32>) -> tensor<3x1x2xf32>
// CHECK-NEXT:     return %1 : tensor<3x1x2xf32>
// CHECK-NEXT: }

func.func @main2(%arg0: tensor<3x2xf32>, %arg1: tensor<3x2xf32>) -> tensor<3x1x4x2xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 3] : (tensor<3x2xf32>) -> tensor<3x1x4x2xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [0, 3] : (tensor<3x2xf32>) -> tensor<3x1x4x2xf32>
    %2 = stablehlo.add %0, %1 : tensor<3x1x4x2xf32>
    return %2 : tensor<3x1x4x2xf32>
}

// CHECK: func.func @main2(%arg0: tensor<3x2xf32>, %arg1: tensor<3x2xf32>) -> tensor<3x1x4x2xf32> {
// CHECK-NEXT:     %0 = stablehlo.add %arg0, %arg1 : tensor<3x2xf32>
// CHECK-NEXT:     %1 = stablehlo.broadcast_in_dim %0, dims = [0, 3] : (tensor<3x2xf32>) -> tensor<3x1x4x2xf32>
// CHECK-NEXT:     return %1 : tensor<3x1x4x2xf32>
// CHECK-NEXT: }

func.func @main3(%arg0: tensor<3x2xf32>) -> tensor<3x1x2xf64> {
    %0 = stablehlo.reshape %arg0 : (tensor<3x2xf32>) -> tensor<3x1x2xf32>
    %1 = stablehlo.convert %0 : (tensor<3x1x2xf32>) -> tensor<3x1x2xf64>
    return %1 : tensor<3x1x2xf64>
}

// CHECK: func.func @main3(%arg0: tensor<3x2xf32>) -> tensor<3x1x2xf64> {
// CHECK-NEXT:     %0 = stablehlo.convert %arg0 : (tensor<3x2xf32>) -> tensor<3x2xf64>
// CHECK-NEXT:     %1 = stablehlo.reshape %0 : (tensor<3x2xf64>) -> tensor<3x1x2xf64>
// CHECK-NEXT:     return %1 : tensor<3x1x2xf64>
// CHECK-NEXT: }
