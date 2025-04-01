// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{passses=131072})" %s | FileCheck %s

module {
  func.func private @foo_raised(%arg0: tensor<85x180x18xf64>) -> tensor<85x180x18x1xf64> {
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = "stablehlo.reduce_window"(%arg0, %cst_0) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [17, 0]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 18>, window_strides = array<i64: 1, 1, 1>}> ({
    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
      %4 = stablehlo.add %arg2, %arg3 : tensor<f64>
      stablehlo.return %4 : tensor<f64>
    }) : (tensor<85x180x18xf64>, tensor<f64>) -> tensor<85x180x18xf64>
    %3 = stablehlo.reshape %0 : (tensor<85x180x18xf64>) -> tensor<85x180x18x1xf64>
    return %3 : tensor<85x180x18x1xf64>
  }
}

// CHECK-LABEL: @foo_raised
// CHECK-SAME:    %[[ARG0:.*]]: tensor<85x180x18xf64>
// CHECK:         %[[RESHAPE:.*]] = stablehlo.reshape %[[ARG0]]
// CHECK:         %[[RED_WIN:.*]] = "stablehlo.reduce_window"(%[[RESHAPE]]
// CHECK:         return %[[RED_WIN]]

// -----

module {
  func.func private @foo_raised(%arg0: tensor<85x180x18xf64>) -> tensor<1x85x1x180x18x1xf64> {
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = "stablehlo.reduce_window"(%arg0, %cst_0) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [17, 0]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 18>, window_strides = array<i64: 1, 1, 1>}> ({
    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
      %4 = stablehlo.add %arg2, %arg3 : tensor<f64>
      stablehlo.return %4 : tensor<f64>
    }) : (tensor<85x180x18xf64>, tensor<f64>) -> tensor<85x180x18xf64>
    %3 = stablehlo.reshape %0 : (tensor<85x180x18xf64>) -> tensor<1x85x1x180x18x1xf64>
    return %3 : tensor<1x85x1x180x18x1xf64>
  }
}

// CHECK-LABEL: @foo_raised
// CHECK-SAME:    %[[ARG0:.*]]: tensor<85x180x18xf64>
// CHECK:         %[[RESHAPE:.*]] = stablehlo.reshape %[[ARG0]]
// CHECK:         %[[RED_WIN:.*]] = "stablehlo.reduce_window"(%[[RESHAPE]]
// CHECK:         return %[[RED_WIN]]

// -----

module {
  func.func private @foo_raised(%arg0: tensor<1x85x180x18xf64>) -> tensor<1x85x1x180x18x1xf64> {
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = "stablehlo.reduce_window"(%arg0, %cst_0) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [17, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 18>, window_strides = array<i64: 1, 1, 1, 1>}> ({
    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
      %4 = stablehlo.add %arg2, %arg3 : tensor<f64>
      stablehlo.return %4 : tensor<f64>
    }) : (tensor<1x85x180x18xf64>, tensor<f64>) -> tensor<1x85x180x18xf64>
    %3 = stablehlo.reshape %0 : (tensor<1x85x180x18xf64>) -> tensor<1x85x1x180x18x1xf64>
    return %3 : tensor<1x85x1x180x18x1xf64>
  }
}

// CHECK-LABEL: @foo_raised
// CHECK-SAME:    %[[ARG0:.*]]: tensor<1x85x180x18xf64>
// CHECK:         %[[RESHAPE:.*]] = stablehlo.reshape %[[ARG0]]
// CHECK:         %[[RED_WIN:.*]] = "stablehlo.reduce_window"(%[[RESHAPE]]
// CHECK:         return %[[RED_WIN]]

// -----

module {
  func.func private @foo_raised(%arg0: tensor<1x85x180x18xf64>) -> tensor<85x180x18xf64> {
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = "stablehlo.reduce_window"(%arg0, %cst_0) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [17, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 18>, window_strides = array<i64: 1, 1, 1, 1>}> ({
    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
      %4 = stablehlo.add %arg2, %arg3 : tensor<f64>
      stablehlo.return %4 : tensor<f64>
    }) : (tensor<1x85x180x18xf64>, tensor<f64>) -> tensor<1x85x180x18xf64>
    %3 = stablehlo.reshape %0 : (tensor<1x85x180x18xf64>) -> tensor<85x180x18xf64>
    return %3 : tensor<85x180x18xf64>
  }
}

// CHECK-LABEL: @foo_raised
// CHECK-SAME:    %[[ARG0:.*]]: tensor<1x85x180x18xf64>
// CHECK:         %[[RESHAPE:.*]] = stablehlo.reshape %[[ARG0]]
// CHECK:         %[[RED_WIN:.*]] = "stablehlo.reduce_window"(%[[RESHAPE]]
// CHECK:         return %[[RED_WIN]]
