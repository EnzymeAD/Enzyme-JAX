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

    %result_diff:2 = enzyme.autodiff @maxpool(%value, %diff_result) <{
      activity = [#enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>]
    }> : (tensor<3xf32>, tensor<1xf32>) -> (tensor<1xf32>, tensor<3xf32>)

    check.expect_eq_const %result_diff#0, dense<[42.0]> : tensor<1xf32>
    check.expect_eq_const %result_diff#1, dense<[1.0, 0.0, 0.0]> : tensor<3xf32>

    %result_meanpool_diff:2 = enzyme.autodiff @meanpool(%value, %diff_result) <{
      activity = [#enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>]
    }> : (tensor<3xf32>, tensor<1xf32>) -> (tensor<1xf32>, tensor<3xf32>)

    check.expect_eq_const %result_meanpool_diff#0, dense<[84.0]> : tensor<1xf32>
    check.expect_eq_const %result_meanpool_diff#1, dense<[1.0, 1.0, 1.0]> : tensor<3xf32>

    %dvalue = stablehlo.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %fwd_maxpool:2 = enzyme.fwddiff @maxpool(%value, %dvalue) <{
      activity = [#enzyme.activity<enzyme_dup>],
      ret_activity = [#enzyme.activity<enzyme_dup>]
    }> : (tensor<3xf32>, tensor<3xf32>) -> (tensor<1xf32>, tensor<1xf32>)

    check.expect_eq_const %fwd_maxpool#0, dense<[42.0]> : tensor<1xf32>
    check.expect_eq_const %fwd_maxpool#1, dense<[1.0]> : tensor<1xf32>

    %fwd_meanpool:2 = enzyme.fwddiff @meanpool(%value, %dvalue) <{
      activity = [#enzyme.activity<enzyme_dup>],
      ret_activity = [#enzyme.activity<enzyme_dup>]
    }> : (tensor<3xf32>, tensor<3xf32>) -> (tensor<1xf32>, tensor<1xf32>)

    check.expect_eq_const %fwd_meanpool#0, dense<[84.0]> : tensor<1xf32>
    check.expect_eq_const %fwd_meanpool#1, dense<[6.0]> : tensor<1xf32>

    %x5 = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0]> : tensor<5xf32>
    %dx5 = stablehlo.constant dense<[1.0, 1.0, 1.0, 1.0, 1.0]> : tensor<5xf32>
    %fwd_cumsum:2 = enzyme.fwddiff @cumsum_sum_abs2(%x5, %dx5) <{
      activity = [#enzyme.activity<enzyme_dup>],
      ret_activity = [#enzyme.activity<enzyme_dup>]
    }> : (tensor<5xf32>, tensor<5xf32>) -> (tensor<f32>, tensor<f32>)

    check.expect_eq_const %fwd_cumsum#0, dense<371.0> : tensor<f32>
    check.expect_eq_const %fwd_cumsum#1, dense<280.0> : tensor<f32>

    func.return
  }

  func.func @cumsum_sum_abs2(%arg0: tensor<5xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.0> : tensor<f32>
    %0 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<[[4, 0]]> : tensor<1x2xi64>, window_dimensions = array<i64: 5>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %3 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
    }) : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
    %1 = stablehlo.multiply %0, %0 : tensor<5xf32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<5xf32>, tensor<f32>) -> tensor<f32>
    return %2 : tensor<f32>
  }

  func.func @meanpool(%arg0: tensor<3xf32>) -> tensor<1xf32> {
    %cst = stablehlo.constant dense<0.0> : tensor<f32>
    %0 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<0> : tensor<1x2xi64>,
                                                  window_dilations = array<i64: 1>,
                                                  window_dimensions = array<i64: 3>,
                                                  window_strides = array<i64: 1>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %3 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
    }) : (tensor<3xf32>, tensor<f32>) -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}

// REVERSE:  func.func private @fwddiffemaxpool(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> (tensor<1xf32>, tensor<1xf32>) {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// REVERSE-NEXT:    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
// REVERSE-NEXT:    %0:2 = stablehlo.reduce(%arg0 init: %cst_0), (%arg1 init: %cst) across dimensions = [0] : (tensor<3xf32>, tensor<3xf32>, tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
// REVERSE-NEXT:     reducer(%arg2: tensor<f32>, %arg4: tensor<f32>) (%arg3: tensor<f32>, %arg5: tensor<f32>)  {
// REVERSE-NEXT:      %3 = stablehlo.compare LT, %arg2, %arg4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
// REVERSE-NEXT:      %4 = stablehlo.select %3, %arg5, %arg3 : tensor<i1>, tensor<f32>
// REVERSE-NEXT:      %5 = stablehlo.maximum %arg2, %arg4 : tensor<f32>
// REVERSE-NEXT:      stablehlo.return %5, %4 : tensor<f32>, tensor<f32>
// REVERSE-NEXT:    }
// REVERSE-NEXT:    %1 = stablehlo.reshape %0#0 : (tensor<f32>) -> tensor<1xf32>
// REVERSE-NEXT:    %2 = stablehlo.reshape %0#1 : (tensor<f32>) -> tensor<1xf32>
// REVERSE-NEXT:    return %1, %2 : tensor<1xf32>, tensor<1xf32>
// REVERSE-NEXT:  }

// REVERSE:  func.func private @fwddiffemeanpool(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> (tensor<1xf32>, tensor<1xf32>) {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// REVERSE-NEXT:    %0 = stablehlo.reduce(%arg1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
// REVERSE-NEXT:    %1 = stablehlo.reshape %0 : (tensor<f32>) -> tensor<1xf32>
// REVERSE-NEXT:    %2 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
// REVERSE-NEXT:    %3 = stablehlo.reshape %2 : (tensor<f32>) -> tensor<1xf32>
// REVERSE-NEXT:    return %3, %1 : tensor<1xf32>, tensor<1xf32>
// REVERSE-NEXT:  }

// REVERSE:  func.func private @fwddiffecumsum_sum_abs2(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>) -> (tensor<f32>, tensor<f32>) {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// REVERSE-NEXT:    %0 = "stablehlo.reduce_window"(%arg1, %cst) <{padding = dense<{{\[\[}}4, 0{{\]\]}}> : tensor<1x2xi64>, window_dimensions = array<i64: 5>}> ({
// REVERSE-NEXT:    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// REVERSE-NEXT:      %6 = stablehlo.add %arg2, %arg3 : tensor<f32>
// REVERSE-NEXT:      stablehlo.return %6 : tensor<f32>
// REVERSE-NEXT:    }) : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
// REVERSE-NEXT:    %1 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<{{\[\[}}4, 0{{\]\]}}> : tensor<1x2xi64>, window_dimensions = array<i64: 5>}> ({
// REVERSE-NEXT:    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// REVERSE-NEXT:      %6 = stablehlo.add %arg2, %arg3 : tensor<f32>
// REVERSE-NEXT:      stablehlo.return %6 : tensor<f32>
// REVERSE-NEXT:    }) : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
// REVERSE-NEXT:    %2 = stablehlo.multiply %0, %1 : tensor<5xf32>
// REVERSE-NEXT:    %3 = stablehlo.add %2, %2 : tensor<5xf32>
// REVERSE-NEXT:    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<5xf32>, tensor<f32>) -> tensor<f32>
// REVERSE-NEXT:    %5 = stablehlo.dot_general %1, %1, contracting_dims = [0] x [0] : (tensor<5xf32>, tensor<5xf32>) -> tensor<f32>
// REVERSE-NEXT:    return %5, %4 : tensor<f32>, tensor<f32>
// REVERSE-NEXT:  }

// REVERSE:  func.func private @diffemaxpool(%arg0: tensor<3xf32>, %arg1: tensor<1xf32>) -> (tensor<1xf32>, tensor<3xf32>) {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// REVERSE-NEXT:    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
// REVERSE-NEXT:    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.maximum across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
// REVERSE-NEXT:    %1 = stablehlo.reshape %0 : (tensor<f32>) -> tensor<1xf32>
// REVERSE-NEXT:    %2 = "stablehlo.select_and_scatter"(%arg0, %arg1, %cst) <{padding = dense<0> : tensor<1x2xi64>, window_dimensions = array<i64: 3>, window_strides = array<i64: 1>}> ({
// REVERSE-NEXT:    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// REVERSE-NEXT:      %3 = stablehlo.compare  GE, %arg2, %arg3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
// REVERSE-NEXT:      stablehlo.return %3 : tensor<i1>
// REVERSE-NEXT:    }, {
// REVERSE-NEXT:    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// REVERSE-NEXT:      %3 = stablehlo.add %arg2, %arg3 : tensor<f32>
// REVERSE-NEXT:      stablehlo.return %3 : tensor<f32>
// REVERSE-NEXT:    }) : (tensor<3xf32>, tensor<1xf32>, tensor<f32>) -> tensor<3xf32>
// REVERSE-NEXT:    return %1, %2 : tensor<1xf32>, tensor<3xf32>
// REVERSE-NEXT:  }

// REVERSE:  func.func private @diffemeanpool(%arg0: tensor<3xf32>, %arg1: tensor<1xf32>) -> (tensor<1xf32>, tensor<3xf32>) {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// REVERSE-NEXT:    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
// REVERSE-NEXT:    %1 = stablehlo.reshape %0 : (tensor<f32>) -> tensor<1xf32>
// REVERSE-NEXT:    %2 = "stablehlo.reduce_window"(%arg1, %cst) <{base_dilations = array<i64: 1>, padding = dense<2> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 3>}> ({
// REVERSE-NEXT:    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// REVERSE-NEXT:      %3 = stablehlo.add %arg2, %arg3 : tensor<f32>
// REVERSE-NEXT:      stablehlo.return %3 : tensor<f32>
// REVERSE-NEXT:    }) : (tensor<1xf32>, tensor<f32>) -> tensor<3xf32>
// REVERSE-NEXT:    return %1, %2 : tensor<1xf32>, tensor<3xf32>
// REVERSE-NEXT:  }
