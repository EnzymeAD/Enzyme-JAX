// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=compare_op_canon<16>;transpose_transpose<16>;broadcast_in_dim_op_canon<16>;convert_op_canon<16>;transpose_is_reshape<16>;transpose_reshape_to_broadcast;reshape_transpose_to_broadcast;reshape_dus},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

module @reactant_vcat attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<3x1xf32>) -> (tensor<1x4xf32>) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x1xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<1x1xf32>
    %2 = stablehlo.dynamic_update_slice %cst, %cst_1, %c, %c : (tensor<4x1xf32>, tensor<1x1xf32>, tensor<i32>, tensor<i32>) -> tensor<4x1xf32>
    %3 = stablehlo.transpose %arg0, dims = [0, 1] : (tensor<3x1xf32>) -> tensor<3x1xf32>
    %4 = stablehlo.dynamic_update_slice %2, %3, %c_0, %c : (tensor<4x1xf32>, tensor<3x1xf32>, tensor<i32>, tensor<i32>) -> tensor<4x1xf32>
    %5 = stablehlo.transpose %4, dims = [1, 0] : (tensor<4x1xf32>) -> tensor<1x4xf32>
    return %5 : tensor<1x4xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<3x1xf32>) -> tensor<1x4xf32> {
// CHECK-NEXT:     %c = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x1xf32>
// CHECK-NEXT:     %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<1x1xf32>
// CHECK-NEXT:     %0 = stablehlo.reshape %cst : (tensor<4x1xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:     %1 = stablehlo.dynamic_update_slice %0, %cst_1, %c, %c : (tensor<1x4xf32>, tensor<1x1xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
// CHECK-NEXT:     %2 = stablehlo.reshape %arg0 : (tensor<3x1xf32>) -> tensor<1x3xf32>
// CHECK-NEXT:     %3 = stablehlo.dynamic_update_slice %1, %2, %c, %c_0 : (tensor<1x4xf32>, tensor<1x3xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
// CHECK-NEXT:     return %3 : tensor<1x4xf32>
// CHECK-NEXT: }
