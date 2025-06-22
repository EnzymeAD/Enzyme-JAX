// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=reshape_deletions_broadcast_in_dim_simplify;reshape_insertions_broadcast_in_dim_simplify},transform-interpreter,enzyme-hlo-remove-transform)" | FileCheck %s

func.func @main1(%arg0: tensor<64x32x2xf32>) -> tensor<16x2x32x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [7, 5, 3] : (tensor<64x32x2xf32>) -> tensor<16x1x1x2x1x32x1x64xf32>
    %1 = stablehlo.reshape %0 : (tensor<16x1x1x2x1x32x1x64xf32>) -> tensor<16x2x32x64xf32>
    return %1 : tensor<16x2x32x64xf32>
}

// CHECK: func.func @main1(%arg0: tensor<64x32x2xf32>) -> tensor<16x2x32x64xf32> {
// CHECK-NEXT:     %0 = stablehlo.broadcast_in_dim %arg0, dims = [3, 2, 1] : (tensor<64x32x2xf32>) -> tensor<16x2x32x64xf32>
// CHECK-NEXT:     return %0 : tensor<16x2x32x64xf32>
// CHECK-NEXT: }

func.func @main2(%arg0: tensor<64x32x2xf32>) -> tensor<16x1x1x2x1x32x1x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [3, 2, 1] : (tensor<64x32x2xf32>) -> tensor<16x2x32x64xf32>
    %1 = stablehlo.reshape %0 : (tensor<16x2x32x64xf32>) -> tensor<16x1x1x2x1x32x1x64xf32>
    return %1 : tensor<16x1x1x2x1x32x1x64xf32>
}

// CHECK: func.func @main2(%arg0: tensor<64x32x2xf32>) -> tensor<16x1x1x2x1x32x1x64xf32> {
// CHECK-NEXT:     %0 = stablehlo.broadcast_in_dim %arg0, dims = [7, 5, 3] : (tensor<64x32x2xf32>) -> tensor<16x1x1x2x1x32x1x64xf32>
// CHECK-NEXT:     return %0 : tensor<16x1x1x2x1x32x1x64xf32>
// CHECK-NEXT: }
