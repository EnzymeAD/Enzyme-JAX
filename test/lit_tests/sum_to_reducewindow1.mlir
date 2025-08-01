// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=sum_to_reducewindow;convert_simplify;reshape_op_canon" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module @reactant_simple_... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<1092xf64> {tf.aliasing_output = 0 : i32}, %arg1: tensor<1095xf64>) -> tensor<1092xf64> {
    %0 = stablehlo.slice %arg1 [0:1092] : (tensor<1095xf64>) -> tensor<1092xf64>
    %1 = stablehlo.slice %arg1 [1:1093] : (tensor<1095xf64>) -> tensor<1092xf64>
    %2 = stablehlo.slice %arg1 [2:1094] : (tensor<1095xf64>) -> tensor<1092xf64>
    %3 = stablehlo.add %0, %1 : tensor<1092xf64>
    %4 = stablehlo.add %3, %2 : tensor<1092xf64>
    return %4 : tensor<1092xf64>
  }
  func.func @main2(%arg0: tensor<1092xf64> {tf.aliasing_output = 0 : i32}, %arg1: tensor<1095xf64>) -> tensor<1092xf64> {
    %0 = stablehlo.slice %arg1 [0:1092] : (tensor<1095xf64>) -> tensor<1092xf64>
    %1 = stablehlo.slice %arg1 [1:1093] : (tensor<1095xf64>) -> tensor<1092xf64>
    %2 = stablehlo.slice %arg1 [2:1094] : (tensor<1095xf64>) -> tensor<1092xf64>
    %3 = stablehlo.add %0, %1 : tensor<1092xf64>
    %4 = stablehlo.subtract %3, %2 : tensor<1092xf64>
    return %4 : tensor<1092xf64>
  }
  func.func @main3(%arg0: tensor<1092xf64> {tf.aliasing_output = 0 : i32}, %arg1: tensor<1095xf64>) -> tensor<1092xf64> {
    %0 = stablehlo.slice %arg1 [0:1092] : (tensor<1095xf64>) -> tensor<1092xf64>
    %1 = stablehlo.slice %arg1 [1:1093] : (tensor<1095xf64>) -> tensor<1092xf64>
    %2 = stablehlo.slice %arg1 [2:1094] : (tensor<1095xf64>) -> tensor<1092xf64>
    %3 = stablehlo.add %0, %1 : tensor<1092xf64>
    %4 = stablehlo.negate %2 : tensor<1092xf64>
    %5 = stablehlo.add %3, %4 : tensor<1092xf64>
    return %5 : tensor<1092xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<1092xf64> {tf.aliasing_output = 0 : i32}, %arg1: tensor<1095xf64>) -> tensor<1092xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [0:1094] : (tensor<1095xf64>) -> tensor<1094xf64>
// CHECK-NEXT:    %1 = "stablehlo.reduce_window"(%0, %cst) <{base_dilations = array<i64: 1>, padding = dense<0> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 3>, window_strides = array<i64: 1>}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      %2 = stablehlo.add %arg2, %arg3 : tensor<f64>
// CHECK-NEXT:      stablehlo.return %2 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<1094xf64>, tensor<f64>) -> tensor<1092xf64>
// CHECK-NEXT:    return %1 : tensor<1092xf64>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @main2(%arg0: tensor<1092xf64> {tf.aliasing_output = 0 : i32}, %arg1: tensor<1095xf64>) -> tensor<1092xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [2:1094] : (tensor<1095xf64>) -> tensor<1092xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %arg1 [0:1093] : (tensor<1095xf64>) -> tensor<1093xf64>
// CHECK-NEXT:    %2 = "stablehlo.reduce_window"(%1, %cst) <{base_dilations = array<i64: 1>, padding = dense<0> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 2>, window_strides = array<i64: 1>}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      %4 = stablehlo.add %arg2, %arg3 : tensor<f64>
// CHECK-NEXT:      stablehlo.return %4 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<1093xf64>, tensor<f64>) -> tensor<1092xf64>
// CHECK-NEXT:    %3 = stablehlo.subtract %2, %0 : tensor<1092xf64>
// CHECK-NEXT:    return %3 : tensor<1092xf64>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @main3(%arg0: tensor<1092xf64> {tf.aliasing_output = 0 : i32}, %arg1: tensor<1095xf64>) -> tensor<1092xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [2:1094] : (tensor<1095xf64>) -> tensor<1092xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %arg1 [0:1093] : (tensor<1095xf64>) -> tensor<1093xf64>
// CHECK-NEXT:    %2 = "stablehlo.reduce_window"(%1, %cst) <{base_dilations = array<i64: 1>, padding = dense<0> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 2>, window_strides = array<i64: 1>}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      %5 = stablehlo.add %arg2, %arg3 : tensor<f64>
// CHECK-NEXT:      stablehlo.return %5 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<1093xf64>, tensor<f64>) -> tensor<1092xf64>
// CHECK-NEXT:    %3 = stablehlo.negate %0 : tensor<1092xf64>
// CHECK-NEXT:    %4 = stablehlo.add %2, %3 : tensor<1092xf64>
// CHECK-NEXT:    return %4 : tensor<1092xf64>
// CHECK-NEXT:  }
