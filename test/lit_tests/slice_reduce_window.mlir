// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{passses=1})" %s | FileCheck %s

module {
  func.func private @foo_raised(%arg0: tensor<85x180x18xf64>) -> tensor<85x180xf64> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<85x180x1xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = "stablehlo.reduce_window"(%arg0, %cst_0) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [17, 0]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 18>, window_strides = array<i64: 1, 1, 1>}> ({
    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
      %4 = stablehlo.add %arg2, %arg3 : tensor<f64>
      stablehlo.return %4 : tensor<f64>
    }) : (tensor<85x180x18xf64>, tensor<f64>) -> tensor<85x180x18xf64>
    %1 = stablehlo.slice %0 [0:85, 0:180, 17:18] : (tensor<85x180x18xf64>) -> tensor<85x180x1xf64>
    %2 = stablehlo.add %1, %cst : tensor<85x180x1xf64>
    %3 = stablehlo.reshape %2 : (tensor<85x180x1xf64>) -> tensor<85x180xf64>
    return %3 : tensor<85x180xf64>
  }
}

// CHECK-LABEL: @foo_raised
// CHECK-SAME:    %[[ARG0:.*]]: tensor<85x180x18xf64>
// CHECK:         %[[REDUCE:.*]] = stablehlo.reduce(%[[ARG0]] {{.*}} applies stablehlo.add across dimensions = [2]
// CHECK:         %[[RESHAPE:.*]] = stablehlo.reshape %[[REDUCE]]
// CHECK:         stablehlo.add %[[RESHAPE]]

// -----

module {
  func.func private @foo_raised(%arg0: tensor<85x180x18xf64>) -> tensor<85x18xf64> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<85x1x18xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = "stablehlo.reduce_window"(%arg0, %cst_0) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [179, 0], [0, 0]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 180, 1>, window_strides = array<i64: 1, 1, 1>}> ({
    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
      %4 = stablehlo.add %arg2, %arg3 : tensor<f64>
      stablehlo.return %4 : tensor<f64>
    }) : (tensor<85x180x18xf64>, tensor<f64>) -> tensor<85x180x18xf64>
    %1 = stablehlo.slice %0 [0:85, 179:180, 0:18] : (tensor<85x180x18xf64>) -> tensor<85x1x18xf64>
    %2 = stablehlo.add %1, %cst : tensor<85x1x18xf64>
    %3 = stablehlo.reshape %2 : (tensor<85x1x18xf64>) -> tensor<85x18xf64>
    return %3 : tensor<85x18xf64>
  }
}

// CHECK-LABEL: @foo_raised
// CHECK-SAME:    %[[ARG0:.*]]: tensor<85x180x18xf64>
// CHECK:         %[[REDUCE:.*]] = stablehlo.reduce(%[[ARG0]] {{.*}} applies stablehlo.add across dimensions = [1]
// CHECK:         %[[RESHAPE:.*]] = stablehlo.reshape %[[REDUCE]]
// CHECK:         stablehlo.add %[[RESHAPE]]
