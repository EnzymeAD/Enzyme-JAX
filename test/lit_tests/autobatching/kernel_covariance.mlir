// RUN: enzymexlamlir-opt --pass-pipeline="any(mark-func-memory-effects,inline,canonicalize,enzyme-hlo-generate-td{patterns=while_simplify<1>(1);while_deadresult;while_dus;while_dus_ds_simplify;greedy_while_loop_batch_fission;elementwise_licm(0);while_licm<1>(1);pad_licm(0)},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

module {
  func.func @kernel_covariance(%arg0: tensor<32x64xf32>) -> (tensor<32x32xf32>, tensor<32x64xf32>) {
    %cst = stablehlo.constant dense<0.015873015873015872> : tensor<f64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<1x1xf32>
    %cst_1 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %c = stablehlo.constant dense<64> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<i32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<1xf32>
    %c_4 = stablehlo.constant dense<32> : tensor<i64>
    %c_5 = stablehlo.constant dense<0> : tensor<i64>
    %c_6 = stablehlo.constant dense<1> : tensor<i64>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<32x32xf32>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<32xf32>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<32x64xf32>) -> tensor<64x32xf32>
    %1:3 = stablehlo.while(%iterArg = %c_5, %iterArg_9 = %cst_8, %iterArg_10 = %0) : tensor<i64>, tensor<32xf32>, tensor<64x32xf32> attributes {enzyme.disable_mincut}
    cond {
      %6 = stablehlo.compare  LT, %iterArg, %c_4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %6 : tensor<i1>
    } do {
      %6 = stablehlo.add %c_6, %iterArg {enzymexla.bounds = [[1, 32]]} : tensor<i64>
      %7 = stablehlo.convert %6 {enzymexla.bounds = [[1, 32]]} : (tensor<i64>) -> tensor<i32>
      %8 = stablehlo.subtract %7, %c_2 {enzymexla.bounds = [[0, 31]]} : tensor<i32>
      %9 = stablehlo.dynamic_update_slice %iterArg_9, %cst_3, %8 : (tensor<32xf32>, tensor<1xf32>, tensor<i32>) -> tensor<32xf32>
      %10:4 = stablehlo.while(%iterArg_11 = %c_5, %iterArg_12 = %9, %iterArg_13 = %6, %iterArg_14 = %iterArg_10) : tensor<i64>, tensor<32xf32>, tensor<i64>, tensor<64x32xf32> attributes {enzyme.disable_mincut}
      cond {
        %23 = stablehlo.compare  LT, %iterArg_11, %c : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %23 : tensor<i1>
      } do {
        %23 = stablehlo.add %c_6, %iterArg_11 {enzymexla.bounds = [[1, 64]]} : tensor<i64>
        %24 = stablehlo.convert %iterArg_13 : (tensor<i64>) -> tensor<i32>
        %25 = stablehlo.subtract %24, %c_2 : tensor<i32>
        %26 = stablehlo.dynamic_slice %iterArg_12, %25, sizes = [1] : (tensor<32xf32>, tensor<i32>) -> tensor<1xf32>
        %27 = stablehlo.reshape %26 : (tensor<1xf32>) -> tensor<f32>
        %28 = stablehlo.convert %23 {enzymexla.bounds = [[1, 64]]} : (tensor<i64>) -> tensor<i32>
        %29 = stablehlo.subtract %28, %c_2 {enzymexla.bounds = [[0, 63]]} : tensor<i32>
        %30 = stablehlo.dynamic_slice %iterArg_14, %29, %25, sizes = [1, 1] {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<64x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
        %31 = stablehlo.reshape %30 : (tensor<1x1xf32>) -> tensor<f32>
        %32 = stablehlo.add %27, %31 : tensor<f32>
        %33 = stablehlo.reshape %32 : (tensor<f32>) -> tensor<1xf32>
        %34 = stablehlo.subtract %iterArg_13, %c_6 : tensor<i64>
        %35 = stablehlo.remainder %34, %c_4 : tensor<i64>
        %36 = stablehlo.add %35, %c_6 : tensor<i64>
        %37 = stablehlo.convert %36 : (tensor<i64>) -> tensor<i32>
        %38 = stablehlo.subtract %37, %c_2 : tensor<i32>
        %39 = stablehlo.dynamic_update_slice %iterArg_12, %33, %38 : (tensor<32xf32>, tensor<1xf32>, tensor<i32>) -> tensor<32xf32>
        stablehlo.return %23, %39, %iterArg_13, %iterArg_14 : tensor<i64>, tensor<32xf32>, tensor<i64>, tensor<64x32xf32>
      }
      %11 = stablehlo.convert %10#2 : (tensor<i64>) -> tensor<i32>
      %12 = stablehlo.subtract %11, %c_2 : tensor<i32>
      %13 = stablehlo.dynamic_slice %10#1, %12, sizes = [1] : (tensor<32xf32>, tensor<i32>) -> tensor<1xf32>
      %14 = stablehlo.reshape %13 : (tensor<1xf32>) -> tensor<f32>
      %15 = stablehlo.multiply %14, %cst_1 : tensor<f32>
      %16 = stablehlo.reshape %15 : (tensor<f32>) -> tensor<1xf32>
      %17 = stablehlo.subtract %10#2, %c_6 : tensor<i64>
      %18 = stablehlo.remainder %17, %c_4 : tensor<i64>
      %19 = stablehlo.add %18, %c_6 : tensor<i64>
      %20 = stablehlo.convert %19 : (tensor<i64>) -> tensor<i32>
      %21 = stablehlo.subtract %20, %c_2 : tensor<i32>
      %22 = stablehlo.dynamic_update_slice %10#1, %16, %21 : (tensor<32xf32>, tensor<1xf32>, tensor<i32>) -> tensor<32xf32>
      stablehlo.return %6, %22, %10#3 : tensor<i64>, tensor<32xf32>, tensor<64x32xf32>
    }
    %2:3 = stablehlo.while(%iterArg = %c_5, %iterArg_9 = %1#1, %iterArg_10 = %1#2) : tensor<i64>, tensor<32xf32>, tensor<64x32xf32> attributes {enzyme.disable_mincut}
    cond {
      %6 = stablehlo.compare  LT, %iterArg, %c : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %6 : tensor<i1>
    } do {
      %6 = stablehlo.add %c_6, %iterArg {enzymexla.bounds = [[1, 64]]} : tensor<i64>
      %7:4 = stablehlo.while(%iterArg_11 = %c_5, %iterArg_12 = %iterArg_9, %iterArg_13 = %iterArg_10, %iterArg_14 = %6) : tensor<i64>, tensor<32xf32>, tensor<64x32xf32>, tensor<i64> attributes {enzyme.disable_mincut}
      cond {
        %8 = stablehlo.compare  LT, %iterArg_11, %c_4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %8 : tensor<i1>
      } do {
        %8 = stablehlo.add %c_6, %iterArg_11 {enzymexla.bounds = [[1, 32]]} : tensor<i64>
        %9 = stablehlo.convert %iterArg_14 : (tensor<i64>) -> tensor<i32>
        %10 = stablehlo.subtract %9, %c_2 : tensor<i32>
        %11 = stablehlo.convert %8 {enzymexla.bounds = [[1, 32]]} : (tensor<i64>) -> tensor<i32>
        %12 = stablehlo.subtract %11, %c_2 {enzymexla.bounds = [[0, 31]]} : tensor<i32>
        %13 = stablehlo.dynamic_slice %iterArg_13, %10, %12, sizes = [1, 1] {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<64x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
        %14 = stablehlo.reshape %13 : (tensor<1x1xf32>) -> tensor<f32>
        %15 = stablehlo.dynamic_slice %iterArg_12, %12, sizes = [1] : (tensor<32xf32>, tensor<i32>) -> tensor<1xf32>
        %16 = stablehlo.reshape %15 : (tensor<1xf32>) -> tensor<f32>
        %17 = stablehlo.subtract %14, %16 : tensor<f32>
        %18 = stablehlo.reshape %17 : (tensor<f32>) -> tensor<1x1xf32>
        %19 = stablehlo.dynamic_update_slice %iterArg_13, %18, %10, %12 : (tensor<64x32xf32>, tensor<1x1xf32>, tensor<i32>, tensor<i32>) -> tensor<64x32xf32>
        stablehlo.return %8, %iterArg_12, %19, %iterArg_14 : tensor<i64>, tensor<32xf32>, tensor<64x32xf32>, tensor<i64>
      }
      stablehlo.return %6, %7#1, %7#2 : tensor<i64>, tensor<32xf32>, tensor<64x32xf32>
    }
    // CHECK: stablehlo.while
    %3:3 = stablehlo.while(%iterArg = %c_5, %iterArg_9 = %2#2, %iterArg_10 = %cst_7) : tensor<i64>, tensor<64x32xf32>, tensor<32x32xf32> attributes {enzyme.disable_mincut, enzymexla.symmetric_matrix = [#enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed NOTGUARANTEED>]}
    cond {
      %6 = stablehlo.compare  LT, %iterArg, %c_4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %6 : tensor<i1>
    } do {
      %6 = stablehlo.add %c_6, %iterArg {enzymexla.bounds = [[1, 32]]} : tensor<i64>
      %7:4 = stablehlo.while(%iterArg_11 = %c_5, %iterArg_12 = %iterArg_9, %iterArg_13 = %iterArg_10, %iterArg_14 = %6) : tensor<i64>, tensor<64x32xf32>, tensor<32x32xf32>, tensor<i64> attributes {enzyme.disable_mincut}
      cond {
        %8 = stablehlo.compare  LT, %iterArg_11, %c_4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %8 : tensor<i1>
      } do {
        %8 = stablehlo.add %c_6, %iterArg_11 {enzymexla.bounds = [[1, 32]]} : tensor<i64>
        %9 = stablehlo.convert %iterArg_14 : (tensor<i64>) -> tensor<i32>
        %10 = stablehlo.subtract %9, %c_2 : tensor<i32>
        %11 = stablehlo.convert %8 {enzymexla.bounds = [[1, 32]]} : (tensor<i64>) -> tensor<i32>
        %12 = stablehlo.subtract %11, %c_2 {enzymexla.bounds = [[0, 31]]} : tensor<i32>
        %13 = stablehlo.dynamic_update_slice %iterArg_13, %cst_0, %10, %12 : (tensor<32x32xf32>, tensor<1x1xf32>, tensor<i32>, tensor<i32>) -> tensor<32x32xf32>
        %14:5 = stablehlo.while(%iterArg_15 = %c_5, %iterArg_16 = %8, %iterArg_17 = %iterArg_12, %iterArg_18 = %13, %iterArg_19 = %iterArg_14) : tensor<i64>, tensor<i64>, tensor<64x32xf32>, tensor<32x32xf32>, tensor<i64> attributes {enzyme.disable_mincut}
        cond {
          %28 = stablehlo.compare  LT, %iterArg_15, %c : (tensor<i64>, tensor<i64>) -> tensor<i1>
          stablehlo.return %28 : tensor<i1>
        } do {
          %28 = stablehlo.add %c_6, %iterArg_15 {enzymexla.bounds = [[1, 64]]} : tensor<i64>
          %29 = stablehlo.convert %iterArg_19 : (tensor<i64>) -> tensor<i32>
          %30 = stablehlo.subtract %29, %c_2 : tensor<i32>
          %31 = stablehlo.convert %iterArg_16 : (tensor<i64>) -> tensor<i32>
          %32 = stablehlo.subtract %31, %c_2 : tensor<i32>
          %33 = stablehlo.dynamic_slice %iterArg_18, %30, %32, sizes = [1, 1] {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<32x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
          %34 = stablehlo.reshape %33 : (tensor<1x1xf32>) -> tensor<f32>
          %35 = stablehlo.convert %28 {enzymexla.bounds = [[1, 64]]} : (tensor<i64>) -> tensor<i32>
          %36 = stablehlo.subtract %35, %c_2 {enzymexla.bounds = [[0, 63]]} : tensor<i32>
          %37 = stablehlo.dynamic_slice %iterArg_17, %36, %30, sizes = [1, 1] {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<64x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
          %38 = stablehlo.reshape %37 : (tensor<1x1xf32>) -> tensor<f32>
          %39 = stablehlo.dynamic_slice %iterArg_17, %36, %32, sizes = [1, 1] {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<64x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
          %40 = stablehlo.reshape %39 : (tensor<1x1xf32>) -> tensor<f32>
          %41 = stablehlo.multiply %38, %40 : tensor<f32>
          %42 = stablehlo.add %34, %41 : tensor<f32>
          %43 = stablehlo.reshape %42 : (tensor<f32>) -> tensor<1x1xf32>
          %44 = stablehlo.dynamic_update_slice %iterArg_18, %43, %30, %32 : (tensor<32x32xf32>, tensor<1x1xf32>, tensor<i32>, tensor<i32>) -> tensor<32x32xf32>
          stablehlo.return %28, %iterArg_16, %iterArg_17, %44, %iterArg_19 : tensor<i64>, tensor<i64>, tensor<64x32xf32>, tensor<32x32xf32>, tensor<i64>
        }
        %15 = stablehlo.convert %14#4 : (tensor<i64>) -> tensor<i32>
        %16 = stablehlo.subtract %15, %c_2 : tensor<i32>
        %17 = stablehlo.convert %14#1 : (tensor<i64>) -> tensor<i32>
        %18 = stablehlo.subtract %17, %c_2 : tensor<i32>
        %19 = stablehlo.dynamic_slice %14#3, %16, %18, sizes = [1, 1] {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<32x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
        %20 = stablehlo.convert %19 : (tensor<1x1xf32>) -> tensor<1x1xf64>
        %21 = stablehlo.reshape %20 : (tensor<1x1xf64>) -> tensor<f64>
        %22 = stablehlo.multiply %21, %cst : tensor<f64>
        %23 = stablehlo.convert %22 : (tensor<f64>) -> tensor<f32>
        %24 = stablehlo.reshape %23 : (tensor<f32>) -> tensor<1x1xf32>
        %25 = stablehlo.dynamic_update_slice %14#3, %24, %16, %18 : (tensor<32x32xf32>, tensor<1x1xf32>, tensor<i32>, tensor<i32>) -> tensor<32x32xf32>
        %26 = stablehlo.dynamic_slice %25, %18, %16, sizes = [1, 1] {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<32x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
        %27 = stablehlo.dynamic_update_slice %14#3, %26, %16, %18 : (tensor<32x32xf32>, tensor<1x1xf32>, tensor<i32>, tensor<i32>) -> tensor<32x32xf32>
        stablehlo.return %8, %14#2, %27, %14#4 : tensor<i64>, tensor<64x32xf32>, tensor<32x32xf32>, tensor<i64>
      }
      stablehlo.return %6, %7#1, %7#2 : tensor<i64>, tensor<64x32xf32>, tensor<32x32xf32>
    }
    %4 = stablehlo.transpose %3#2, dims = [1, 0] : (tensor<32x32xf32>) -> tensor<32x32xf32>
    %5 = stablehlo.transpose %3#1, dims = [1, 0] : (tensor<64x32xf32>) -> tensor<32x64xf32>
    return %4, %5 : tensor<32x32xf32>, tensor<32x64xf32>
  }
}
