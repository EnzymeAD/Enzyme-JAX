// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=dot_transpose},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

module @reactant_fgrad attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<4x3x8x2xf64> {tf.aliasing_output = 3 : i32}, %arg1: tensor<3x8x2xf64> {tf.aliasing_output = 4 : i32}, %arg2: tensor<4xf64> {tf.aliasing_output = 5 : i32}) -> (tensor<4x3x8x2xf64>, tensor<3x8x2xf64>, tensor<4xf64>, tensor<4x3x8x2xf64>, tensor<3x8x2xf64>, tensor<4xf64>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %0 = stablehlo.transpose %arg0, dims = [3, 2, 1, 0] : (tensor<4x3x8x2xf64>) -> tensor<2x8x3x4xf64>
    %1 = stablehlo.transpose %arg1, dims = [2, 1, 0] : (tensor<3x8x2xf64>) -> tensor<2x8x3xf64>
    %2 = stablehlo.dot_general %0, %1, contracting_dims = [0, 1, 2] x [0, 1, 2], precision = [DEFAULT, DEFAULT] : (tensor<2x8x3x4xf64>, tensor<2x8x3xf64>) -> tensor<4xf64>
    %3 = stablehlo.dot_general %2, %cst, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<f64>) -> tensor<4xf64>
    %4 = stablehlo.dot_general %cst, %arg2, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<f64>, tensor<4xf64>) -> tensor<4xf64>
    %5 = stablehlo.dot_general %4, %1, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<2x8x3xf64>) -> tensor<4x2x8x3xf64>
    %6 = stablehlo.dot_general %4, %0, contracting_dims = [0] x [3], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<2x8x3x4xf64>) -> tensor<2x8x3xf64>
    %7 = stablehlo.transpose %6, dims = [2, 1, 0] : (tensor<2x8x3xf64>) -> tensor<3x8x2xf64>
    %8 = stablehlo.transpose %5, dims = [0, 3, 2, 1] : (tensor<4x2x8x3xf64>) -> tensor<4x3x8x2xf64>
    return %8, %7, %3, %arg0, %arg1, %arg2 : tensor<4x3x8x2xf64>, tensor<3x8x2xf64>, tensor<4xf64>, tensor<4x3x8x2xf64>, tensor<3x8x2xf64>, tensor<4xf64>
  }
}

// CHECK: func.func @main(%arg0: tensor<4x3x8x2xf64> {tf.aliasing_output = 3 : i32}, %arg1: tensor<3x8x2xf64> {tf.aliasing_output = 4 : i32}, %arg2: tensor<4xf64> {tf.aliasing_output = 5 : i32}) -> (tensor<4x3x8x2xf64>, tensor<3x8x2xf64>, tensor<4xf64>, tensor<4x3x8x2xf64>, tensor<3x8x2xf64>, tensor<4xf64>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [3, 2, 1, 0] : (tensor<4x3x8x2xf64>) -> tensor<2x8x3x4xf64>
// CHECK-NEXT:     %1 = stablehlo.transpose %arg1, dims = [2, 1, 0] : (tensor<3x8x2xf64>) -> tensor<2x8x3xf64>
// CHECK-NEXT:     %2 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [3, 2, 1] x [2, 1, 0], precision = [DEFAULT, DEFAULT] : (tensor<4x3x8x2xf64>, tensor<3x8x2xf64>) -> tensor<4xf64>
// CHECK-NEXT:     %3 = stablehlo.dot_general %2, %cst, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:     %4 = stablehlo.dot_general %cst, %arg2, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<f64>, tensor<4xf64>) -> tensor<4xf64>
// CHECK-NEXT:     %5 = stablehlo.dot_general %4, %1, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<2x8x3xf64>) -> tensor<4x2x8x3xf64>
// CHECK-NEXT:     %6 = stablehlo.dot_general %4, %0, contracting_dims = [0] x [3], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<2x8x3x4xf64>) -> tensor<2x8x3xf64>
// CHECK-NEXT:     %7 = stablehlo.transpose %6, dims = [2, 1, 0] : (tensor<2x8x3xf64>) -> tensor<3x8x2xf64>
// CHECK-NEXT:     %8 = stablehlo.transpose %5, dims = [0, 3, 2, 1] : (tensor<4x2x8x3xf64>) -> tensor<4x3x8x2xf64>
// CHECK-NEXT:     return %8, %7, %3, %arg0, %arg1, %arg2 : tensor<4x3x8x2xf64>, tensor<3x8x2xf64>, tensor<4xf64>, tensor<4x3x8x2xf64>, tensor<3x8x2xf64>, tensor<4xf64>
// CHECK-NEXT: }
