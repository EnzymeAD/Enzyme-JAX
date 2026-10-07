// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise | stablehlo-translate - --interpret
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise | FileCheck %s --check-prefix=REVERSE

module {
  func.func @cumsum(%arg0: tensor<5xf64>) -> tensor<5xf64> {
    %cst = stablehlo.constant dense<0.0> : tensor<f64>
    %0 = "stablehlo.reduce_window"(%arg0, %cst) <{base_dilations = array<i64: 1>, padding = dense<[[4, 0]]> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 5>, window_strides = array<i64: 1>}> ({
    ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f64>
      stablehlo.return %1 : tensor<f64>
    }) : (tensor<5xf64>, tensor<f64>) -> tensor<5xf64>
    return %0 : tensor<5xf64>
  }

  func.func @revcumsum(%arg0: tensor<5xf64>) -> tensor<5xf64> {
    %cst = stablehlo.constant dense<0.0> : tensor<f64>
    %0 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<[[0, 4]]> : tensor<1x2xi64>, window_dimensions = array<i64: 5>}> ({
    ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f64>
      stablehlo.return %1 : tensor<f64>
    }) : (tensor<5xf64>, tensor<f64>) -> tensor<5xf64>
    return %0 : tensor<5xf64>
  }

  func.func @cumsum2d(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %cst = stablehlo.constant dense<0.0> : tensor<f32>
    %0 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<[[0, 0], [3, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 1, 4>, window_strides = array<i64: 1, 1>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }

  func.func @window3(%arg0: tensor<5xf64>) -> tensor<5xf64> {
    %cst = stablehlo.constant dense<0.0> : tensor<f64>
    %0 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<[[2, 0]]> : tensor<1x2xi64>, window_dimensions = array<i64: 3>}> ({
    ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f64>
      stablehlo.return %1 : tensor<f64>
    }) : (tensor<5xf64>, tensor<f64>) -> tensor<5xf64>
    return %0 : tensor<5xf64>
  }

  func.func @main() {
    %x = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0]> : tensor<5xf64>
    %dy = stablehlo.constant dense<[1.0, 10.0, 100.0, 1000.0, 10000.0]> : tensor<5xf64>

    %r0:2 = enzyme.autodiff @cumsum(%x, %dy) {
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_active>]
    } : (tensor<5xf64>, tensor<5xf64>) -> (tensor<5xf64>, tensor<5xf64>)
    check.expect_eq_const %r0#0, dense<[1.0, 3.0, 6.0, 10.0, 15.0]> : tensor<5xf64>
    check.expect_eq_const %r0#1, dense<[11111.0, 11110.0, 11100.0, 11000.0, 10000.0]> : tensor<5xf64>

    %r1:2 = enzyme.autodiff @revcumsum(%x, %dy) {
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_active>]
    } : (tensor<5xf64>, tensor<5xf64>) -> (tensor<5xf64>, tensor<5xf64>)
    check.expect_eq_const %r1#0, dense<[15.0, 14.0, 12.0, 9.0, 5.0]> : tensor<5xf64>
    check.expect_eq_const %r1#1, dense<[1.0, 11.0, 111.0, 1111.0, 11111.0]> : tensor<5xf64>

    %r3:2 = enzyme.autodiff @window3(%x, %dy) {
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_active>]
    } : (tensor<5xf64>, tensor<5xf64>) -> (tensor<5xf64>, tensor<5xf64>)
    check.expect_eq_const %r3#0, dense<[1.0, 3.0, 6.0, 9.0, 12.0]> : tensor<5xf64>
    check.expect_eq_const %r3#1, dense<[111.0, 1110.0, 11100.0, 11000.0, 10000.0]> : tensor<5xf64>

    %m = stablehlo.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>
    %dm = stablehlo.constant dense<[[1.0, 10.0, 100.0, 1000.0], [2.0, 20.0, 200.0, 2000.0]]> : tensor<2x4xf32>
    %r2:2 = enzyme.autodiff @cumsum2d(%m, %dm) {
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_active>]
    } : (tensor<2x4xf32>, tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>)
    check.expect_eq_const %r2#0, dense<[[1.0, 3.0, 6.0, 10.0], [5.0, 11.0, 18.0, 26.0]]> : tensor<2x4xf32>
    check.expect_eq_const %r2#1, dense<[[1111.0, 1110.0, 1100.0, 1000.0], [2222.0, 2220.0, 2200.0, 2000.0]]> : tensor<2x4xf32>

    func.return
  }
}

// REVERSE-LABEL: func.func private @diffecumsum(
// REVERSE-SAME:      %[[X:[a-z0-9_]+]]: tensor<5xf64>, %[[DY:[a-z0-9_]+]]: tensor<5xf64>) -> (tensor<5xf64>, tensor<5xf64>) {
// REVERSE-NOT:   stablehlo.pad
// REVERSE:       %[[FWD:[a-z0-9_]+]] = "stablehlo.reduce_window"(%[[X]], %{{[a-z0-9_]+}}) <{base_dilations = array<i64: 1>, padding = dense<{{\[\[}}4, 0]]> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 5>, window_strides = array<i64: 1>}> ({
// REVERSE-NOT:   stablehlo.pad
// REVERSE:       %[[G:[a-z0-9_]+]] = stablehlo.add %[[DY]], %{{[a-z0-9_]+}} : tensor<5xf64>
// REVERSE-NEXT:  %[[ADJ:[a-z0-9_]+]] = "stablehlo.reduce_window"(%[[G]], %{{[a-z0-9_]+}}) <{base_dilations = array<i64: 1>, padding = dense<{{\[\[}}0, 4]]> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 5>, window_strides = array<i64: 1>}> ({
// REVERSE-NEXT:  ^bb0(%[[A:[a-z0-9_]+]]: tensor<f64>, %[[B:[a-z0-9_]+]]: tensor<f64>):
// REVERSE-NEXT:    %[[S:[a-z0-9_]+]] = stablehlo.add %[[A]], %[[B]] : tensor<f64>
// REVERSE-NEXT:    stablehlo.return %[[S]] : tensor<f64>
// REVERSE-NEXT:  }) : (tensor<5xf64>, tensor<f64>) -> tensor<5xf64>
// REVERSE-NEXT:  %[[R:[a-z0-9_]+]] = stablehlo.add %[[ADJ]], %{{[a-z0-9_]+}} : tensor<5xf64>
// REVERSE-NEXT:  return %[[FWD]], %[[R]] : tensor<5xf64>, tensor<5xf64>

// REVERSE-LABEL: func.func private @differevcumsum(
// REVERSE-SAME:      %[[X:[a-z0-9_]+]]: tensor<5xf64>, %[[DY:[a-z0-9_]+]]: tensor<5xf64>) -> (tensor<5xf64>, tensor<5xf64>) {
// REVERSE-NOT:   stablehlo.pad
// REVERSE:       %[[FWD:[a-z0-9_]+]] = "stablehlo.reduce_window"(%[[X]], %{{[a-z0-9_]+}}) <{padding = dense<{{\[\[}}0, 4]]> : tensor<1x2xi64>, window_dimensions = array<i64: 5>}> ({
// REVERSE-NOT:   stablehlo.pad
// REVERSE:       %[[G:[a-z0-9_]+]] = stablehlo.add %[[DY]], %{{[a-z0-9_]+}} : tensor<5xf64>
// REVERSE-NEXT:  %[[ADJ:[a-z0-9_]+]] = "stablehlo.reduce_window"(%[[G]], %{{[a-z0-9_]+}}) <{padding = dense<{{\[\[}}4, 0]]> : tensor<1x2xi64>, window_dimensions = array<i64: 5>}> ({
// REVERSE-NEXT:  ^bb0(%[[A:[a-z0-9_]+]]: tensor<f64>, %[[B:[a-z0-9_]+]]: tensor<f64>):
// REVERSE-NEXT:    %[[S:[a-z0-9_]+]] = stablehlo.add %[[A]], %[[B]] : tensor<f64>
// REVERSE-NEXT:    stablehlo.return %[[S]] : tensor<f64>
// REVERSE-NEXT:  }) : (tensor<5xf64>, tensor<f64>) -> tensor<5xf64>
// REVERSE-NEXT:  %[[R:[a-z0-9_]+]] = stablehlo.add %[[ADJ]], %{{[a-z0-9_]+}} : tensor<5xf64>
// REVERSE-NEXT:  return %[[FWD]], %[[R]] : tensor<5xf64>, tensor<5xf64>

// A window shorter than the dimension keeps the general adjoint.

// REVERSE-LABEL: func.func private @diffewindow3(
// REVERSE-SAME:      %[[X:[a-z0-9_]+]]: tensor<5xf64>, %[[DY:[a-z0-9_]+]]: tensor<5xf64>) -> (tensor<5xf64>, tensor<5xf64>) {
// REVERSE:       %[[G:[a-z0-9_]+]] = stablehlo.add %[[DY]], %{{[a-z0-9_]+}} : tensor<5xf64>
// REVERSE-NEXT:  %[[P:[a-z0-9_]+]] = stablehlo.pad %[[G]], %[[Z:[a-z0-9_]+]], low = [0], high = [2], interior = [0] : (tensor<5xf64>, tensor<f64>) -> tensor<7xf64>
// REVERSE-NEXT:  %[[ADJ:[a-z0-9_]+]] = "stablehlo.reduce_window"(%[[P]], %[[Z]]) <{base_dilations = array<i64: 1>, padding = dense<0> : tensor<1x2xi64>, window_dimensions = array<i64: 3>}> ({
// REVERSE-NEXT:  ^bb0(%[[A:[a-z0-9_]+]]: tensor<f64>, %[[B:[a-z0-9_]+]]: tensor<f64>):
// REVERSE-NEXT:    %[[S:[a-z0-9_]+]] = stablehlo.add %[[A]], %[[B]] : tensor<f64>
// REVERSE-NEXT:    stablehlo.return %[[S]] : tensor<f64>
// REVERSE-NEXT:  }) : (tensor<7xf64>, tensor<f64>) -> tensor<5xf64>

// REVERSE-LABEL: func.func private @diffecumsum2d(
// REVERSE-SAME:      %[[X:[a-z0-9_]+]]: tensor<2x4xf32>, %[[DY:[a-z0-9_]+]]: tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>) {
// REVERSE-NOT:   stablehlo.pad
// REVERSE:       %[[G:[a-z0-9_]+]] = stablehlo.add %[[DY]], %{{[a-z0-9_]+}} : tensor<2x4xf32>
// REVERSE-NEXT:  %[[ADJ:[a-z0-9_]+]] = "stablehlo.reduce_window"(%[[G]], %{{[a-z0-9_]+}}) <{padding = dense<{{\[\[}}0, 0], [0, 3]]> : tensor<2x2xi64>, window_dimensions = array<i64: 1, 4>, window_strides = array<i64: 1, 1>}> ({
// REVERSE-NEXT:  ^bb0(%[[A:[a-z0-9_]+]]: tensor<f32>, %[[B:[a-z0-9_]+]]: tensor<f32>):
// REVERSE-NEXT:    %[[S:[a-z0-9_]+]] = stablehlo.add %[[A]], %[[B]] : tensor<f32>
// REVERSE-NEXT:    stablehlo.return %[[S]] : tensor<f32>
// REVERSE-NEXT:  }) : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
