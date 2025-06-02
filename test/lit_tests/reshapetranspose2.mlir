// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=reshape_transpose_to_broadcast},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

module @reactant_fn attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<4x3x2xf64>) -> tensor<3x2x4xf64> {
    %0 = stablehlo.slice %arg0 [0:4, 0:1, 0:2] : (tensor<4x3x2xf64>) -> tensor<4x1x2xf64>
    %1 = stablehlo.reshape %0 : (tensor<4x1x2xf64>) -> tensor<4x2xf64>
    
    %3 = stablehlo.slice %arg0 [0:4, 1:2, 0:2] : (tensor<4x3x2xf64>) -> tensor<4x1x2xf64>
    %4 = stablehlo.reshape %3 : (tensor<4x1x2xf64>) -> tensor<4x2xf64>
    %5 = stablehlo.transpose %4, dims = [1, 0] : (tensor<4x2xf64>) -> tensor<2x4xf64>
    %6 = stablehlo.slice %arg0 [0:4, 2:3, 0:2] : (tensor<4x3x2xf64>) -> tensor<4x1x2xf64>
    %7 = stablehlo.reshape %6 : (tensor<4x1x2xf64>) -> tensor<4x2xf64>
    %8 = stablehlo.transpose %7, dims = [1, 0] : (tensor<4x2xf64>) -> tensor<2x4xf64>
    
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<4x2xf64>) -> tensor<2x4xf64>
    %9 = stablehlo.reshape %2 : (tensor<2x4xf64>) -> tensor<1x2x4xf64>
  
    %10 = stablehlo.reshape %5 : (tensor<2x4xf64>) -> tensor<1x2x4xf64>
    %11 = stablehlo.reshape %8 : (tensor<2x4xf64>) -> tensor<1x2x4xf64>
    %12 = stablehlo.concatenate %9, %10, %11, dim = 0 : (tensor<1x2x4xf64>, tensor<1x2x4xf64>, tensor<1x2x4xf64>) -> tensor<3x2x4xf64>
    return %12 : tensor<3x2x4xf64>
  }
}

// CHECK: func.func @main(%arg0: tensor<4x3x2xf64>) -> tensor<3x2x4xf64> {
// CHECK-NEXT:     %0 = stablehlo.slice %arg0 [0:4, 0:1, 0:2] : (tensor<4x3x2xf64>) -> tensor<4x1x2xf64>
// CHECK-NEXT:     %1 = stablehlo.reshape %0 : (tensor<4x1x2xf64>) -> tensor<4x2xf64>
// CHECK-NEXT:     %2 = stablehlo.slice %arg0 [0:4, 1:2, 0:2] : (tensor<4x3x2xf64>) -> tensor<4x1x2xf64>
// CHECK-NEXT:     %3 = stablehlo.reshape %2 : (tensor<4x1x2xf64>) -> tensor<4x2xf64>
// CHECK-NEXT:     %4 = stablehlo.slice %arg0 [0:4, 2:3, 0:2] : (tensor<4x3x2xf64>) -> tensor<4x1x2xf64>
// CHECK-NEXT:     %5 = stablehlo.reshape %4 : (tensor<4x1x2xf64>) -> tensor<4x2xf64>
// CHECK-NEXT:     %6 = stablehlo.broadcast_in_dim %1, dims = [2, 1] : (tensor<4x2xf64>) -> tensor<1x2x4xf64>
// CHECK-NEXT:     %7 = stablehlo.broadcast_in_dim %3, dims = [2, 1] : (tensor<4x2xf64>) -> tensor<1x2x4xf64>
// CHECK-NEXT:     %8 = stablehlo.broadcast_in_dim %5, dims = [2, 1] : (tensor<4x2xf64>) -> tensor<1x2x4xf64>
// CHECK-NEXT:     %9 = stablehlo.concatenate %6, %7, %8, dim = 0 : (tensor<1x2x4xf64>, tensor<1x2x4xf64>, tensor<1x2x4xf64>) -> tensor<3x2x4xf64>
// CHECK-NEXT:     return %9 : tensor<3x2x4xf64>
// CHECK-NEXT: }
