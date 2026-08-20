// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=attention_warp_specialized_accumulator outfn= argTys=enzyme_dup,enzyme_dup retTys=enzyme_dup mode=ForwardMode" --canonicalize | FileCheck %s

// Reduced from the TTIR generated for benchmark/06-fused-attention.py with
// BLOCK_M=128, HEAD_DIM=128, warp_specialize=true, and IS_HOPPER=false.

// CHECK-LABEL: tt.func @attention_warp_specialized_accumulator(
// CHECK:       %[[DRESHAPE:.+]] = tt.reshape %{{.+}} : tensor<128x128xf32> -> tensor<128x2x64xf32>
// CHECK-NEXT:  %[[RESHAPE:.+]] = tt.reshape %{{.+}} : tensor<128x128xf32> -> tensor<128x2x64xf32>
// CHECK-NEXT:  %[[DTRANS:.+]] = tt.trans %[[DRESHAPE]] {order = array<i32: 0, 2, 1>}
// CHECK-NEXT:  %[[TRANS:.+]] = tt.trans %[[RESHAPE]] {order = array<i32: 0, 2, 1>}
// CHECK-NEXT:  %[[DLHS:.+]], %[[DRHS:.+]] = tt.split %[[DTRANS]]
// CHECK-NEXT:  %[[LHS:.+]], %[[RHS:.+]] = tt.split %[[TRANS]]
// CHECK:       %[[DJOIN:.+]] = tt.join %{{.+}}, %{{.+}} : tensor<128x64xf32> -> tensor<128x64x2xf32>
// CHECK-NEXT:  %[[JOIN:.+]] = tt.join %{{.+}}, %{{.+}} : tensor<128x64xf32> -> tensor<128x64x2xf32>
// CHECK-NEXT:  %[[DTRANS_BACK:.+]] = tt.trans %[[DJOIN]] {order = array<i32: 0, 2, 1>}
// CHECK-NEXT:  %[[TRANS_BACK:.+]] = tt.trans %[[JOIN]] {order = array<i32: 0, 2, 1>}
// CHECK-NEXT:  %[[DRESULT:.+]] = tt.reshape %[[DTRANS_BACK]] : tensor<128x2x64xf32> -> tensor<128x128xf32>
// CHECK-NEXT:  %[[RESULT:.+]] = tt.reshape %[[TRANS_BACK]] : tensor<128x2x64xf32> -> tensor<128x128xf32>
// CHECK-NEXT:  tt.return %[[RESULT]], %[[DRESULT]]

module {
  tt.func @attention_warp_specialized_accumulator(
      %acc: tensor<128x128xf32>, %alpha: tensor<128xf32>)
      -> tensor<128x128xf32> {
    %reshaped = tt.reshape %acc
        : tensor<128x128xf32> -> tensor<128x2x64xf32>
    %transposed = tt.trans %reshaped {order = array<i32: 0, 2, 1>}
        : tensor<128x2x64xf32> -> tensor<128x64x2xf32>
    %lhs, %rhs = tt.split %transposed
        : tensor<128x64x2xf32> -> tensor<128x64xf32>
    %alpha_expanded = tt.expand_dims %alpha {axis = 1 : i32}
        : tensor<128xf32> -> tensor<128x1xf32>
    %alpha_broadcast = tt.broadcast %alpha_expanded
        : tensor<128x1xf32> -> tensor<128x64xf32>
    %scaled_lhs = arith.mulf %lhs, %alpha_broadcast
        : tensor<128x64xf32>
    %scaled_rhs = arith.mulf %rhs, %alpha_broadcast
        : tensor<128x64xf32>
    %joined = tt.join %scaled_lhs, %scaled_rhs
        : tensor<128x64xf32> -> tensor<128x64x2xf32>
    %transposed_back = tt.trans %joined {order = array<i32: 0, 2, 1>}
        : tensor<128x64x2xf32> -> tensor<128x2x64xf32>
    %result = tt.reshape %transposed_back
        : tensor<128x2x64xf32> -> tensor<128x128xf32>
    tt.return %result : tensor<128x128xf32>
  }
}
