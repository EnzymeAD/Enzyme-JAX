// RUN: enzymexlamlir-opt %s --enzyme-batch --inline --enzyme-hlo-opt --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --lower-enzymexla-ml --inline --enzyme-hlo-opt --drop-unsupported-attributes --symbol-dce --canonicalize | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-batch --inline --enzyme-hlo-opt --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --lower-enzymexla-ml --inline --enzyme-hlo-opt --drop-unsupported-attributes --symbol-dce | stablehlo-translate --interpret

module @reactant_df attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func private @"*_broadcast_scalar"(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>) {
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<f32>
    return %0, %arg0, %arg1 : tensor<f32>, tensor<f32>, tensor<f32>
  }
  func.func private @cos_broadcast_scalar(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %0 = stablehlo.cosine %arg0 : tensor<f32>
    return %0, %arg0 : tensor<f32>, tensor<f32>
  }
  func.func private @"*_broadcast_scalar_1"(%arg0: tensor<i64>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<i64>, tensor<f32>) {
    %0 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<f32>
    %1 = stablehlo.multiply %0, %arg1 : tensor<f32>
    return %1, %arg0, %arg1 : tensor<f32>, tensor<i64>, tensor<f32>
  }
  func.func private @"Const{typeof(myf)}(Main.myf)_autodiff"(%arg0: tensor<3xf32>) -> tensor<3xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    %cst = stablehlo.constant dense<6.28318548> : tensor<f32>
    %c = stablehlo.constant dense<3> : tensor<i64>
    %c_0 = stablehlo.constant dense<30> : tensor<i64>
    %c_1 = stablehlo.constant dense<3> : tensor<i64>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %1:6 = stablehlo.while(%iterArg = %c_2, %iterArg_3 = %c_1, %iterArg_4 = %c_0, %iterArg_5 = %c, %iterArg_6 = %0, %iterArg_7 = %cst) : tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<f32> attributes {enzyme.disable_mincut, enzyme.binomial_checkpointing, enzyme.checkpoint_period = 3 : i64, enzyme.enable_checkpointing = true}
    cond {
      %3 = stablehlo.subtract %iterArg_4, %iterArg_5 : tensor<i64>
      %4 = stablehlo.divide %3, %iterArg_3 : tensor<i64>
      %c_8 = stablehlo.constant dense<1> : tensor<i64>
      %5 = stablehlo.add %4, %c_8 : tensor<i64>
      %6 = stablehlo.compare LT, %iterArg, %5 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %6 : tensor<i1>
    } do {
      %3 = stablehlo.multiply %iterArg, %iterArg_3 : tensor<i64>
      %4 = stablehlo.add %iterArg_5, %3 : tensor<i64>
      %c_8 = stablehlo.constant dense<1> : tensor<i64>
      %5 = stablehlo.add %iterArg, %c_8 : tensor<i64>
      %6 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i64>) -> tensor<3xi64>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<3xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<3xf32>
      %7 = stablehlo.broadcast_in_dim %iterArg_7, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %8 = stablehlo.broadcast_in_dim %iterArg_6, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
      %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
      %10:3 = enzyme.batch @"*_broadcast_scalar"(%7, %9) {batch_shape = array<i64: 3>} : (tensor<3xf32>, tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>, tensor<3xf32>)
      %11 = stablehlo.broadcast_in_dim %10#0, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
      %12:2 = enzyme.batch @cos_broadcast_scalar(%11) {batch_shape = array<i64: 3>} : (tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>)
      %13 = stablehlo.broadcast_in_dim %12#0, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
      %14:3 = enzyme.batch @"*_broadcast_scalar_1"(%6, %13) {batch_shape = array<i64: 3>} : (tensor<3xi64>, tensor<3xf32>) -> (tensor<3xf32>, tensor<3xi64>, tensor<3xf32>)
      stablehlo.return %5, %iterArg_3, %iterArg_4, %iterArg_5, %14#0, %iterArg_7 : tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<f32>
    }
    %2 = stablehlo.transpose %1#4, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    return %2 : tensor<3xf32>
  }
  func.func @main() -> () {
    %expected_y = stablehlo.constant dense<[30.0, -23.808477, 26.87121]> : tensor<3xf32>
    %expected_dx = stablehlo.constant dense<[0.0, 3.2691625e18, -3.634229e17]> : tensor<3xf32>
    %cst = stablehlo.constant dense<[0.0, 0.7853982, 0.3926991]> : tensor<3xf32>
    %one = stablehlo.constant dense<1.000> : tensor<3xf32>
    %y, %dx = enzyme.autodiff @"Const{typeof(myf)}(Main.myf)_autodiff"(%cst, %one) {activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_active>]} : (tensor<3xf32>, tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>)
    check.expect_close %y, %expected_y, max_ulp_difference = 10, min_ulp_difference = 0 : tensor<3xf32>, tensor<3xf32>
    check.expect_close %dx, %expected_dx, max_ulp_difference = 10, min_ulp_difference = 0 : tensor<3xf32>, tensor<3xf32>
    return
  }
}

// CHECK:  func.func @main() {
// CHECK-NEXT:    %cst = stablehlo.constant dense<[3.000000e+01, -23.8084774, 26.8712101]> : tensor<3xf32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<[0.000000e+00, 3.26916253E+18, -3.63422884E+17]> : tensor<3xf32>
// CHECK-NEXT:    %cst_1 = stablehlo.constant dense<[0.000000e+00, 0.785398185, 0.392699093]> : tensor<3xf32>
// CHECK-NEXT:    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
// CHECK-NEXT:    %c = stablehlo.constant dense<{{\[}}[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 1, 1], [3, 3, 1, 1], [4, 4, 2, 1], [5, 5, 2, 2], [6, 6, 3, 2], [7, 7, 4, 2], [8, 8, 5, 3], [9, 9, 5, 3], [10, 10, 6, 4]]> : tensor<11x4xi64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<0> : tensor<3xi64>
// CHECK-NEXT:    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<3x3xf32>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_8 = stablehlo.constant dense<10> : tensor<i64>
// CHECK-NEXT:    %cst_9 = stablehlo.constant dense<6.28318548> : tensor<3xf32>
// CHECK-NEXT:    %0:5 = stablehlo.while(%iterArg = %c_6, %iterArg_10 = %c_6, %iterArg_11 = %cst_1, %iterArg_12 = %cst_4, %iterArg_13 = %c_3) : tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<3x3xf32>, tensor<3xi64> attributes {enzyme.disable_mincut, enzymexla.checkpoint_segment}
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %2 = stablehlo.compare LT, %iterArg, %c_5 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %2 = stablehlo.reshape %iterArg_11 : (tensor<3xf32>) -> tensor<1x3xf32>
// CHECK-NEXT:      %3 = stablehlo.dynamic_update_slice %iterArg_12, %2, %iterArg, %c_6 : (tensor<3x3xf32>, tensor<1x3xf32>, tensor<i64>, tensor<i64>) -> tensor<3x3xf32>
// CHECK-NEXT:      %4 = stablehlo.reshape %iterArg_10 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:      %5 = stablehlo.dynamic_update_slice %iterArg_13, %4, %iterArg : (tensor<3xi64>, tensor<1xi64>, tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:      %6 = stablehlo.subtract %c_8, %iterArg_10 : tensor<i64>
// CHECK-NEXT:      %7 = stablehlo.subtract %c_5, %iterArg : tensor<i64>
// CHECK-NEXT:      %8 = stablehlo.minimum %7, %6 : tensor<i64>
// CHECK-NEXT:      %9 = stablehlo.dynamic_slice %c, %6, %8, sizes = [1, 1] : (tensor<11x4xi64>, tensor<i64>, tensor<i64>) -> tensor<1x1xi64>
// CHECK-NEXT:      %10 = stablehlo.reshape %9 : (tensor<1x1xi64>) -> tensor<i64>
// CHECK-NEXT:      %11:2 = stablehlo.while(%iterArg_14 = %c_6, %iterArg_15 = %iterArg_11) : tensor<i64>, tensor<3xf32> attributes {enzyme.disable_mincut, enzymexla.checkpoint_segment}
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %14 = stablehlo.compare LT, %iterArg_14, %10 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %14 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %14 = stablehlo.add %iterArg_10, %iterArg_14 : tensor<i64>
// CHECK-NEXT:        %15 = stablehlo.multiply %14, %c_5 : tensor<i64>
// CHECK-NEXT:        %16 = stablehlo.add %c_5, %15 : tensor<i64>
// CHECK-NEXT:        %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:        %18 = stablehlo.multiply %cst_9, %iterArg_15 : tensor<3xf32>
// CHECK-NEXT:        %19 = stablehlo.cosine %18 : tensor<3xf32>
// CHECK-NEXT:        %20 = stablehlo.convert %17 : (tensor<3xi64>) -> tensor<3xf32>
// CHECK-NEXT:        %21 = stablehlo.multiply %20, %19 : tensor<3xf32>
// CHECK-NEXT:        %22 = stablehlo.add %iterArg_14, %c_7 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %22, %21 : tensor<i64>, tensor<3xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %12 = stablehlo.add %iterArg_10, %10 : tensor<i64>
// CHECK-NEXT:      %13 = stablehlo.add %iterArg, %c_7 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %13, %12, %11#1, %3, %5 : tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<3x3xf32>, tensor<3xi64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %1:5 = stablehlo.while(%iterArg = %c_6, %iterArg_10 = %c_5, %iterArg_11 = %cst_2, %iterArg_12 = %0#3, %iterArg_13 = %0#4) : tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<3x3xf32>, tensor<3xi64> attributes {enzyme.disable_mincut, enzymexla.checkpoint_segment}
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %2 = stablehlo.compare LT, %iterArg, %c_8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %2 = stablehlo.subtract %iterArg_10, %c_7 : tensor<i64>
// CHECK-NEXT:      %3 = stablehlo.subtract %c_8, %iterArg : tensor<i64>
// CHECK-NEXT:      %4 = stablehlo.dynamic_slice %iterArg_12, %2, %c_6, sizes = [1, 3] : (tensor<3x3xf32>, tensor<i64>, tensor<i64>) -> tensor<1x3xf32>
// CHECK-NEXT:      %5 = stablehlo.reshape %4 : (tensor<1x3xf32>) -> tensor<3xf32>
// CHECK-NEXT:      %6 = stablehlo.dynamic_slice %iterArg_13, %2, sizes = [1] : (tensor<3xi64>, tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:      %7 = stablehlo.reshape %6 : (tensor<1xi64>) -> tensor<i64>
// CHECK-NEXT:      %8:5 = stablehlo.while(%iterArg_14 = %7, %iterArg_15 = %2, %iterArg_16 = %5, %iterArg_17 = %iterArg_12, %iterArg_18 = %iterArg_13) : tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<3x3xf32>, tensor<3xi64> attributes {enzymexla.checkpoint_segment}
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %21 = stablehlo.add %iterArg_14, %c_7 : tensor<i64>
// CHECK-NEXT:        %22 = stablehlo.compare LT, %21, %3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %22 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %21 = stablehlo.subtract %3, %iterArg_14 : tensor<i64>
// CHECK-NEXT:        %22 = stablehlo.subtract %c_5, %iterArg_15 : tensor<i64>
// CHECK-NEXT:        %23 = stablehlo.minimum %22, %21 : tensor<i64>
// CHECK-NEXT:        %24 = stablehlo.dynamic_slice %c, %21, %23, sizes = [1, 1] : (tensor<11x4xi64>, tensor<i64>, tensor<i64>) -> tensor<1x1xi64>
// CHECK-NEXT:        %25 = stablehlo.reshape %24 : (tensor<1x1xi64>) -> tensor<i64>
// CHECK-NEXT:        %26 = stablehlo.reshape %iterArg_16 : (tensor<3xf32>) -> tensor<1x3xf32>
// CHECK-NEXT:        %27 = stablehlo.dynamic_update_slice %iterArg_17, %26, %iterArg_15, %c_6 : (tensor<3x3xf32>, tensor<1x3xf32>, tensor<i64>, tensor<i64>) -> tensor<3x3xf32>
// CHECK-NEXT:        %28 = stablehlo.reshape %iterArg_14 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:        %29 = stablehlo.dynamic_update_slice %iterArg_18, %28, %iterArg_15 : (tensor<3xi64>, tensor<1xi64>, tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:        %30 = stablehlo.add %iterArg_14, %25 : tensor<i64>
// CHECK-NEXT:        %31 = stablehlo.compare EQ, %30, %3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        %32 = stablehlo.subtract %30, %c_7 : tensor<i64>
// CHECK-NEXT:        %33 = stablehlo.select %31, %32, %30 : tensor<i1>, tensor<i64>
// CHECK-NEXT:        %34:2 = stablehlo.while(%iterArg_19 = %iterArg_14, %iterArg_20 = %iterArg_16) : tensor<i64>, tensor<3xf32> attributes {enzyme.disable_mincut, enzymexla.checkpoint_segment}
// CHECK-NEXT:        cond {
// CHECK-NEXT:          %36 = stablehlo.compare LT, %iterArg_19, %33 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:          stablehlo.return %36 : tensor<i1>
// CHECK-NEXT:        } do {
// CHECK-NEXT:          %36 = stablehlo.multiply %iterArg_19, %c_5 : tensor<i64>
// CHECK-NEXT:          %37 = stablehlo.add %c_5, %36 : tensor<i64>
// CHECK-NEXT:          %38 = stablehlo.broadcast_in_dim %37, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:          %39 = stablehlo.multiply %cst_9, %iterArg_20 : tensor<3xf32>
// CHECK-NEXT:          %40 = stablehlo.cosine %39 : tensor<3xf32>
// CHECK-NEXT:          %41 = stablehlo.convert %38 : (tensor<3xi64>) -> tensor<3xf32>
// CHECK-NEXT:          %42 = stablehlo.multiply %41, %40 : tensor<3xf32>
// CHECK-NEXT:          %43 = stablehlo.add %iterArg_19, %c_7 : tensor<i64>
// CHECK-NEXT:          stablehlo.return %43, %42 : tensor<i64>, tensor<3xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        %35 = stablehlo.add %iterArg_15, %c_7 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %30, %35, %34#1, %27, %29 : tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<3x3xf32>, tensor<3xi64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %9 = stablehlo.subtract %3, %c_7 : tensor<i64>
// CHECK-NEXT:      %10 = stablehlo.multiply %9, %c_5 : tensor<i64>
// CHECK-NEXT:      %11 = stablehlo.add %c_5, %10 : tensor<i64>
// CHECK-NEXT:      %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:      %13 = stablehlo.multiply %cst_9, %8#2 : tensor<3xf32>
// CHECK-NEXT:      %14 = stablehlo.convert %12 : (tensor<3xi64>) -> tensor<3xf32>
// CHECK-NEXT:      %15 = stablehlo.multiply %iterArg_11, %14 : tensor<3xf32>
// CHECK-NEXT:      %16 = stablehlo.sine %13 : tensor<3xf32>
// CHECK-NEXT:      %17 = stablehlo.negate %16 : tensor<3xf32>
// CHECK-NEXT:      %18 = stablehlo.multiply %15, %17 : tensor<3xf32>
// CHECK-NEXT:      %19 = stablehlo.multiply %18, %cst_9 : tensor<3xf32>
// CHECK-NEXT:      %20 = stablehlo.add %iterArg, %c_7 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %20, %8#1, %19, %8#3, %8#4 : tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<3x3xf32>, tensor<3xi64>
// CHECK-NEXT:    }
// CHECK-NEXT:    check.expect_close %0#2, %cst, max_ulp_difference = 10 : tensor<3xf32>, tensor<3xf32>
// CHECK-NEXT:    check.expect_close %1#2, %cst_0, max_ulp_difference = 10 : tensor<3xf32>, tensor<3xf32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
