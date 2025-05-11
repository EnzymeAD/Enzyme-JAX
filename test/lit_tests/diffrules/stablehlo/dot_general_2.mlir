// RUN: enzymexlamlir-opt --enzyme --arith-raise --remove-unnecessary-enzyme-ops --enzyme-hlo-opt --inline --inline --enzyme-hlo-opt --inline --enzyme-batch --inline --remove-unnecessary-enzyme-ops --inline --enzyme-hlo-opt %s | FileCheck %s

module @reactant_fgrad attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func private @"Const{typeof(fprimal)}(Main.fprimal)_autodiff"(%arg0: tensor<4x3x8x2xf64>, %arg1: tensor<3x8x2xf64>, %arg2: tensor<4xf64>) -> (tensor<f64>, tensor<4x3x8x2xf64>, tensor<3x8x2xf64>, tensor<4xf64>) {
    %0 = stablehlo.transpose %arg0, dims = [3, 2, 1, 0] : (tensor<4x3x8x2xf64>) -> tensor<2x8x3x4xf64>
    %1 = stablehlo.transpose %arg1, dims = [2, 1, 0] : (tensor<3x8x2xf64>) -> tensor<2x8x3xf64>
    %2 = stablehlo.dot_general %0, %1, contracting_dims = [0, 1, 2] x [0, 1, 2], precision = [DEFAULT, DEFAULT] : (tensor<2x8x3x4xf64>, tensor<2x8x3xf64>) -> tensor<4xf64>
    %3 = stablehlo.dot_general %arg2, %2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    return %3, %arg0, %arg1, %arg2 : tensor<f64>, tensor<4x3x8x2xf64>, tensor<3x8x2xf64>, tensor<4xf64>
  }
  func.func @main(%arg0: tensor<4x3x8x2xf64> {tf.aliasing_output = 3 : i32}, %arg1: tensor<3x8x2xf64> {tf.aliasing_output = 4 : i32}, %arg2: tensor<4xf64> {tf.aliasing_output = 5 : i32}) -> (tensor<4x3x8x2xf64>, tensor<3x8x2xf64>, tensor<4xf64>, tensor<4x3x8x2xf64>, tensor<3x8x2xf64>, tensor<4xf64>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<4x3x8x2xf64>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<3x8x2xf64>
    %0:6 = enzyme.autodiff @"Const{typeof(fprimal)}(Main.fprimal)_autodiff"(%arg0, %arg1, %arg2, %cst, %cst_1, %cst_2, %cst_0) {activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>]} : (tensor<4x3x8x2xf64>, tensor<3x8x2xf64>, tensor<4xf64>, tensor<f64>, tensor<4x3x8x2xf64>, tensor<3x8x2xf64>, tensor<4xf64>) -> (tensor<4x3x8x2xf64>, tensor<3x8x2xf64>, tensor<4xf64>, tensor<4x3x8x2xf64>, tensor<3x8x2xf64>, tensor<4xf64>)
    return %0#3, %0#4, %0#5, %0#0, %0#1, %0#2 : tensor<4x3x8x2xf64>, tensor<3x8x2xf64>, tensor<4xf64>, tensor<4x3x8x2xf64>, tensor<3x8x2xf64>, tensor<4xf64>
  }
}

// CHECK: func.func @main(%arg0: tensor<4x3x8x2xf64> {tf.aliasing_output = 3 : i32}, %arg1: tensor<3x8x2xf64> {tf.aliasing_output = 4 : i32}, %arg2: tensor<4xf64> {tf.aliasing_output = 5 : i32}) -> (tensor<4x3x8x2xf64>, tensor<3x8x2xf64>, tensor<4xf64>, tensor<4x3x8x2xf64>, tensor<3x8x2xf64>, tensor<4xf64>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [3, 2, 1, 0] : (tensor<4x3x8x2xf64>) -> tensor<2x8x3x4xf64>
// CHECK-NEXT:     %1 = stablehlo.transpose %arg1, dims = [2, 1, 0] : (tensor<3x8x2xf64>) -> tensor<2x8x3xf64>
// CHECK-NEXT:     %2 = stablehlo.dot_general %0, %1, contracting_dims = [0, 1, 2] x [0, 1, 2], precision = [DEFAULT, DEFAULT] : (tensor<2x8x3x4xf64>, tensor<2x8x3xf64>) -> tensor<4xf64>
// CHECK-NEXT:     %3 = stablehlo.dot_general %2, %cst, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:     %4 = stablehlo.dot_general %cst, %arg2, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<f64>, tensor<4xf64>) -> tensor<4xf64>
// CHECK-NEXT:     %5 = stablehlo.dot_general %4, %1, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<2x8x3xf64>) -> tensor<4x2x8x3xf64>
// CHECK-NEXT:     %6 = stablehlo.dot_general %4, %0, contracting_dims = [0] x [3], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<2x8x3x4xf64>) -> tensor<2x8x3xf64>
// CHECK-NEXT:     %7 = stablehlo.transpose %6, dims = [2, 1, 0] : (tensor<2x8x3xf64>) -> tensor<3x8x2xf64>
// CHECK-NEXT:     %8 = stablehlo.transpose %5, dims = [0, 3, 2, 1] : (tensor<4x2x8x3xf64>) -> tensor<4x3x8x2xf64>
// CHECK-NEXT:     return %8, %7, %3, %arg0, %arg1, %arg2 : tensor<4x3x8x2xf64>, tensor<3x8x2xf64>, tensor<4xf64>, tensor<4x3x8x2xf64>, tensor<3x8x2xf64>, tensor<4xf64>
// CHECK-NEXT: }
