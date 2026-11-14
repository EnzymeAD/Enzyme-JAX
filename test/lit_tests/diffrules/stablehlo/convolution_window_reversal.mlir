// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=conv_rev_1d outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | FileCheck %s
// RUN: enzymexlamlir-opt %s --pass-pipeline="any(enzyme{postpasses=arith-raise{stablehlo=true},canonicalize,cse,canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math,canonicalize,cse,canonicalize,arith-raise{stablehlo=true}},inline{default-pipeline=canonicalize inlining-threshold=4294967295 max-iterations=4},enzyme-hlo-opt,remove-unnecessary-enzyme-ops)" | stablehlo-translate - --interpret --allow-unregistered-dialect

module {
  func.func @conv_rev_1d(%x: tensor<1x1x4xf64>, %w: tensor<1x1x3xf64>) -> tensor<1x1x2xf64> {
    %0 = "stablehlo.convolution"(%x, %w) <{
      batch_group_count = 1 : i64,
      dimension_numbers = #stablehlo.conv<[b, f, 0]x[o, i, 0]->[b, f, 0]>,
      feature_group_count = 1 : i64,
      lhs_dilation = array<i64: 1>,
      padding = dense<0> : tensor<1x2xi64>,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
      rhs_dilation = array<i64: 1>,
      window_reversal = array<i1: true>,
      window_strides = array<i64: 1>
    }> : (tensor<1x1x4xf64>, tensor<1x1x3xf64>) -> tensor<1x1x2xf64>
    return %0 : tensor<1x1x2xf64>
  }

  func.func private @op_conv_sq(%x: tensor<1x1x4xf64>, %w: tensor<1x1x1xf64>) -> tensor<f64> {
    %y = "stablehlo.convolution"(%x, %w) <{
      batch_group_count = 1 : i64,
      dimension_numbers = #stablehlo.conv<[b, f, 0]x[o, i, 0]->[b, f, 0]>,
      feature_group_count = 1 : i64,
      lhs_dilation = array<i64: 1>,
      padding = dense<0> : tensor<1x2xi64>,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
      rhs_dilation = array<i64: 1>,
      window_reversal = array<i1: true>,
      window_strides = array<i64: 1>
    }> : (tensor<1x1x4xf64>, tensor<1x1x1xf64>) -> tensor<1x1x4xf64>
    %sq = stablehlo.multiply %y, %y : tensor<1x1x4xf64>
    %z = stablehlo.constant dense<0.0> : tensor<f64>
    %s = stablehlo.reduce(%sq init: %z) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x4xf64>, tensor<f64>) -> tensor<f64>
    return %s : tensor<f64>
  }

  func.func private @grad_x_norm_sq(%x: tensor<1x1x4xf64>, %w: tensor<1x1x1xf64>) -> tensor<f64> {
    %one = stablehlo.constant dense<1.0> : tensor<f64>
    %dx = enzyme.autodiff @op_conv_sq(%x, %w, %one) <{
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    }> : (tensor<1x1x4xf64>, tensor<1x1x1xf64>, tensor<f64>) -> tensor<1x1x4xf64>
    %sq = stablehlo.multiply %dx, %dx : tensor<1x1x4xf64>
    %z = stablehlo.constant dense<0.0> : tensor<f64>
    %s = stablehlo.reduce(%sq init: %z) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x4xf64>, tensor<f64>) -> tensor<f64>
    return %s : tensor<f64>
  }

  func.func @main() {
    %x = stablehlo.constant dense<[[[1.0, 2.0, 4.0, 8.0]]]> : tensor<1x1x4xf64>
    %w = stablehlo.constant dense<[[[10.0, 20.0, 30.0]]]> : tensor<1x1x3xf64>
    %dy = stablehlo.constant dense<[[[5.0, 7.0]]]> : tensor<1x1x2xf64>

    %grads:2 = enzyme.autodiff @conv_rev_1d(%x, %w, %dy) <{
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    }> : (tensor<1x1x4xf64>, tensor<1x1x3xf64>, tensor<1x1x2xf64>) -> (tensor<1x1x4xf64>, tensor<1x1x3xf64>)

    check.expect_almost_eq_const %grads#0, dense<[[[150.0, 310.0, 190.0, 70.0]]]> : tensor<1x1x4xf64>
    check.expect_almost_eq_const %grads#1, dense<[[[76.0, 38.0, 19.0]]]> : tensor<1x1x3xf64>

    // Issue Reactant.jl#3064: nested reverse-over-reverse gradient wrt w
    %x2 = stablehlo.constant dense<[[[1.0, 2.0, 3.0, 4.0]]]> : tensor<1x1x4xf64>
    %w2 = stablehlo.constant dense<[[[0.7]]]> : tensor<1x1x1xf64>
    %one = stablehlo.constant dense<1.0> : tensor<f64>
    %dw2 = enzyme.autodiff @grad_x_norm_sq(%x2, %w2, %one) <{
      activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    }> : (tensor<1x1x4xf64>, tensor<1x1x1xf64>, tensor<f64>) -> tensor<1x1x1xf64>

    check.expect_almost_eq_const %dw2, dense<[[[164.64]]]> : tensor<1x1x1xf64>
    return
  }
}

// CHECK: func.func @conv_rev_1d(%arg0: tensor<1x1x4xf64>, %arg1: tensor<1x1x3xf64>, %arg2: tensor<1x1x2xf64>) -> (tensor<1x1x4xf64>, tensor<1x1x3xf64>)
// CHECK: stablehlo.convolution(%arg0, %arg2) dim_numbers = [f, b, 0]x[i, o, 0]->[f, b, 0], window = {stride = [1], pad = {{\[\[}}0, 0]], lhs_dilate = [1], rhs_dilate = [1], reverse = [false]}
// CHECK: stablehlo.reverse {{%.*}}, dims = [2] : tensor<1x1x3xf64>
