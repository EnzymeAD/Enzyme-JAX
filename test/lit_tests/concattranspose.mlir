// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=concat_transpose},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

module @reactant_fn attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<4x3x2xf64>) -> tensor<3x2x4xf64> {
    %0 = stablehlo.slice %arg0 [0:4, 0:1, 0:2] : (tensor<4x3x2xf64>) -> tensor<4x1x2xf64>
    %1 = stablehlo.slice %arg0 [0:4, 1:2, 0:2] : (tensor<4x3x2xf64>) -> tensor<4x1x2xf64>
    %2 = stablehlo.slice %arg0 [0:4, 2:3, 0:2] : (tensor<4x3x2xf64>) -> tensor<4x1x2xf64>
    %3 = stablehlo.transpose %0, dims = [1, 2, 0] : (tensor<4x1x2xf64>) -> tensor<1x2x4xf64>
    %4 = stablehlo.transpose %1, dims = [1, 2, 0] : (tensor<4x1x2xf64>) -> tensor<1x2x4xf64>
    %5 = stablehlo.transpose %2, dims = [1, 2, 0] : (tensor<4x1x2xf64>) -> tensor<1x2x4xf64>
    %6 = stablehlo.concatenate %3, %4, %5, dim = 0 : (tensor<1x2x4xf64>, tensor<1x2x4xf64>, tensor<1x2x4xf64>) -> tensor<3x2x4xf64>
    return %6 : tensor<3x2x4xf64>
  }
}

// CHECK: func.func @main(%arg0: tensor<4x3x2xf64>) -> tensor<3x2x4xf64> {
// CHECK-NEXT:     %0 = stablehlo.slice %arg0 [0:4, 0:1, 0:2] : (tensor<4x3x2xf64>) -> tensor<4x1x2xf64>
// CHECK-NEXT:     %1 = stablehlo.slice %arg0 [0:4, 1:2, 0:2] : (tensor<4x3x2xf64>) -> tensor<4x1x2xf64>
// CHECK-NEXT:     %2 = stablehlo.slice %arg0 [0:4, 2:3, 0:2] : (tensor<4x3x2xf64>) -> tensor<4x1x2xf64>
// CHECK-NEXT:     %3 = stablehlo.concatenate %0, %1, %2, dim = 1 : (tensor<4x1x2xf64>, tensor<4x1x2xf64>, tensor<4x1x2xf64>) -> tensor<4x3x2xf64>
// CHECK-NEXT:     %4 = stablehlo.transpose %3, dims = [1, 2, 0] : (tensor<4x3x2xf64>) -> tensor<3x2x4xf64>
// CHECK-NEXT:     return %4 : tensor<3x2x4xf64>
// CHECK-NEXT: }
