// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=1" %s | FileCheck %s

func.func @reduce_window_sum(%arg0: tensor<4x4xf64>) -> tensor<4x4xf64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 1, 1>,
    padding = dense<[[2, 0], [0, 0]]> : tensor<2x2xi64>
  }> ({
  ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
    %1 = stablehlo.add %arg1, %arg2 : tensor<f64>
    stablehlo.return %1 : tensor<f64>
  }) : (tensor<4x4xf64>, tensor<f64>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}

// CHECK-LABEL: func.func @reduce_window_sum
// CHECK: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK: %[[RES:.*]] = "stablehlo.reduce_window"(%{{.*}}, %[[CST]]) <{padding = dense<{{\[\[2, 0\], \[0, 0\]\]}}> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
// CHECK: ^bb0(%[[ARG1:.*]]: tensor<f32>, %[[ARG2:.*]]: tensor<f32>):
// CHECK:   %[[ADD:.*]] = stablehlo.add %[[ARG1]], %[[ARG2]] : tensor<f32>
// CHECK:   stablehlo.return %[[ADD]] : tensor<f32>
// CHECK: }) : (tensor<4x4xf32>, tensor<f32>) -> tensor<4x4xf32>
// CHECK: return %{{.*}}
