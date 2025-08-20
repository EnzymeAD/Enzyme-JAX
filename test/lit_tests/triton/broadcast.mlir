// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzymexla-stablehlo-to-triton-compatible-dialect)" %s | FileCheck %s

func.func @main1(%arg0: tensor<4xbf16>) -> tensor<2x4x8xbf16> {
    %x = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<4xbf16>) -> tensor<2x4x8xbf16>
    return %x : tensor<2x4x8xbf16>
}

// CHECK: func.func @main1(%arg0: tensor<4xbf16>) -> tensor<2x4x8xbf16> {
// CHECK-NEXT:     %0 = tt.reshape %arg0 : tensor<4xbf16> -> tensor<4x1x1xbf16>
// CHECK-NEXT:     %1 = tt.trans %0 {order = array<i32: 1, 0, 2>} : tensor<4x1x1xbf16> -> tensor<1x4x1xbf16>
// CHECK-NEXT:     %2 = tt.broadcast %1 : tensor<1x4x1xbf16> -> tensor<2x4x8xbf16>
// CHECK-NEXT:     return %2 : tensor<2x4x8xbf16>
// CHECK-NEXT: }
