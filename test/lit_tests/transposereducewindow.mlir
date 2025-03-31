// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{passses=65536})" %s | FileCheck %s

module {
  func.func @main(%0 : tensor<20x9xf64>, %cst: tensor<f64>) -> tensor<9x20xf64> {
    %2 = "stablehlo.reduce_window"(%0, %cst) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 0]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 9>, window_strides = array<i64: 1, 1>}> ({
    ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
      %6 = stablehlo.add %arg3, %arg4 : tensor<f64>
      stablehlo.return %6 : tensor<f64>
    }) : (tensor<20x9xf64>, tensor<f64>) -> tensor<20x9xf64>
    %t = stablehlo.transpose %2, dims = [1, 0] : (tensor<20x9xf64>) -> tensor<9x20xf64>
    return %t : tensor<9x20xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<20x9xf64>, %arg1: tensor<f64>) -> tensor<9x20xf64> {
// CHECK-NEXT:    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<20x9xf64>) -> tensor<9x20xf64>
// CHECK-NEXT{LITERAL}:    %1 = "stablehlo.reduce_window"(%0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<[[8, 0], [0, 0]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 9, 1>, window_strides = array<i64: 1, 1>}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      %2 = stablehlo.add %arg2, %arg3 : tensor<f64>
// CHECK-NEXT:      stablehlo.return %2 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<9x20xf64>, tensor<f64>) -> tensor<9x20xf64>
// CHECK-NEXT:    return %1 : tensor<9x20xf64>
// CHECK-NEXT:  }