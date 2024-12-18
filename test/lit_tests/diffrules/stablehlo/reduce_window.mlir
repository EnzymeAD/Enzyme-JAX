// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | stablehlo-translate - --interpret
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | FileCheck %s --check-prefix=REVERSE

module {
  func.func @maxpool(%arg0: tensor<3xf32>) -> tensor<1xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<0> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 3>, window_strides = array<i64: 1>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %3 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
    }) : (tensor<3xf32>, tensor<f32>) -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }

  func.func @main() {
    %value = stablehlo.constant dense<[42.0, 42.0, 0.0]> : tensor<3xf32>
    %diff_result = stablehlo.constant dense<[1.0]> : tensor<1xf32>

    %result_diff:2 = enzyme.autodiff @maxpool(%value, %diff_result) {
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_active>]
    } : (tensor<3xf32>, tensor<1xf32>) -> (tensor<1xf32>, tensor<3xf32>)

    check.expect_eq_const %result_diff#0, dense<[42.0]> : tensor<1xf32>
    check.expect_eq_const %result_diff#1, dense<[1.0, 0.0, 0.0]> : tensor<3xf32>

    func.return
  }
}

// REVERSE:  func.func private @diffemaxpool(%arg0: tensor<3xf32>, %arg1: tensor<1xf32>) -> (tensor<1xf32>, tensor<3xf32>) {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// REVERSE-NEXT:    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
// REVERSE-NEXT:    %0 = "stablehlo.reduce_window"(%arg0, %cst_0) <{padding = dense<0> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 3>, window_strides = array<i64: 1>}> ({
// REVERSE-NEXT:    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// REVERSE-NEXT:      %2 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
// REVERSE-NEXT:      stablehlo.return %2 : tensor<f32>
// REVERSE-NEXT:    }) : (tensor<3xf32>, tensor<f32>) -> tensor<1xf32>
// REVERSE-NEXT:    %1 = "stablehlo.select_and_scatter"(%arg0, %arg1, %cst) <{padding = dense<0> : tensor<1x2xi64>, window_dimensions = array<i64: 3>, window_strides = array<i64: 1>}> ({
// REVERSE-NEXT:    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// REVERSE-NEXT:      %2 = stablehlo.compare  GE, %arg2, %arg3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
// REVERSE-NEXT:      stablehlo.return %2 : tensor<i1>
// REVERSE-NEXT:    }, {
// REVERSE-NEXT:    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// REVERSE-NEXT:      %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
// REVERSE-NEXT:      stablehlo.return %2 : tensor<f32>
// REVERSE-NEXT:    }) : (tensor<3xf32>, tensor<1xf32>, tensor<f32>) -> tensor<3xf32>
// REVERSE-NEXT:    return %0, %1 : tensor<1xf32>, tensor<3xf32>
// REVERSE-NEXT:  }
