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
    %y, %dx = enzyme.autodiff @"Const{typeof(myf)}(Main.myf)_autodiff"(%cst, %one) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_active>]} : (tensor<3xf32>, tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>)
    check.expect_close %y, %expected_y, max_ulp_difference = 10, min_ulp_difference = 0 : tensor<3xf32>, tensor<3xf32>
    check.expect_close %dx, %expected_dx, max_ulp_difference = 10, min_ulp_difference = 0 : tensor<3xf32>, tensor<3xf32>
    return
  }
}

// CHECK:  func.func @main() {
// CHECK-NEXT:    %c = stablehlo.constant dense<9> : tensor<i64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<[3.000000e+01, -23.8084774, 26.8712101]> : tensor<3xf32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<[0.000000e+00, 3.26916253E+18, -3.63422884E+17]> : tensor<3xf32>
// CHECK-NEXT:    %cst_1 = stablehlo.constant dense<[0.000000e+00, 0.785398185, 0.392699093]> : tensor<3xf32>
// CHECK-NEXT:    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<3xi64>
// CHECK-NEXT:    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<3x3xf32>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_8 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_9 = stablehlo.constant dense<10> : tensor<i64>
// CHECK-NEXT:    %cst_10 = stablehlo.constant dense<6.28318548> : tensor<3xf32>
// CHECK-NEXT:    %0:5 = stablehlo.while(%iterArg = %c_7, %iterArg_11 = %c_7, %iterArg_12 = %cst_1, %iterArg_13 = %cst_5, %iterArg_14 = %c_4) : tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<3x3xf32>, tensor<3xi64> attributes {enzyme.disable_mincut, enzymexla.checkpoint_segment}
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %2 = stablehlo.compare LT, %iterArg, %c_6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %2 = stablehlo.reshape %iterArg_12 : (tensor<3xf32>) -> tensor<1x3xf32>
// CHECK-NEXT:      %3 = stablehlo.dynamic_update_slice %iterArg_13, %2, %iterArg, %c_7 : (tensor<3x3xf32>, tensor<1x3xf32>, tensor<i64>, tensor<i64>) -> tensor<3x3xf32>
// CHECK-NEXT:      %4 = stablehlo.reshape %iterArg_11 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:      %5 = stablehlo.dynamic_update_slice %iterArg_14, %4, %iterArg : (tensor<3xi64>, tensor<1xi64>, tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:      %6 = stablehlo.subtract %c_9, %iterArg_11 : tensor<i64>
// CHECK-NEXT:      %7 = stablehlo.subtract %c_6, %iterArg : tensor<i64>
// CHECK-NEXT:      %8 = stablehlo.minimum %7, %6 : tensor<i64>
// CHECK-NEXT:      %9 = stablehlo.compare GE, %iterArg_11, %c : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      %10 = stablehlo.compare LE, %8, %c_8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      %11 = stablehlo.or %9, %10 : tensor<i1>
// CHECK-NEXT:      %12 = "stablehlo.if"(%11) ({
// CHECK-NEXT:        stablehlo.return %6 : tensor<i64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        %16:2 = stablehlo.while(%iterArg_15 = %c_7, %iterArg_16 = %c_8) : tensor<i64>, tensor<i64>
// CHECK-NEXT:        cond {
// CHECK-NEXT:          %32 = stablehlo.compare LT, %iterArg_16, %6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:          stablehlo.return %32 : tensor<i1>
// CHECK-NEXT:        } do {
// CHECK-NEXT:          %32 = stablehlo.add %iterArg_15, %c_8 : tensor<i64>
// CHECK-NEXT:          %33 = stablehlo.add %8, %32 : tensor<i64>
// CHECK-NEXT:          %34 = stablehlo.multiply %iterArg_16, %33 : tensor<i64>
// CHECK-NEXT:          %35 = stablehlo.divide %34, %32 : tensor<i64>
// CHECK-NEXT:          stablehlo.return %32, %35 : tensor<i64>, tensor<i64>
// CHECK-NEXT:        }
// CHECK-NEXT:        %17 = stablehlo.add %8, %16#0 : tensor<i64>
// CHECK-NEXT:        %18 = stablehlo.multiply %16#1, %8 : tensor<i64>
// CHECK-NEXT:        %19 = stablehlo.divide %18, %17 : tensor<i64>
// CHECK-NEXT:        %20 = stablehlo.subtract %6, %19 : tensor<i64>
// CHECK-NEXT:        %21 = stablehlo.multiply %16#1, %16#0 : tensor<i64>
// CHECK-NEXT:        %22 = stablehlo.divide %21, %17 : tensor<i64>
// CHECK-NEXT:        %23 = stablehlo.maximum %20, %c_8 : tensor<i64>
// CHECK-NEXT:        %24 = stablehlo.subtract %6, %c_8 : tensor<i64>
// CHECK-NEXT:        %25 = stablehlo.minimum %22, %24 : tensor<i64>
// CHECK-NEXT:        %26 = stablehlo.add %23, %25 : tensor<i64>
// CHECK-NEXT:        %27 = stablehlo.divide %26, %c_3 : tensor<i64>
// CHECK-NEXT:        %28 = stablehlo.subtract %8, %c_8 : tensor<i64>
// CHECK-NEXT:        %29 = stablehlo.subtract %6, %28 : tensor<i64>
// CHECK-NEXT:        %30 = stablehlo.minimum %27, %29 : tensor<i64>
// CHECK-NEXT:        %31 = stablehlo.maximum %30, %c_8 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %31 : tensor<i64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<i64>
// CHECK-NEXT:      %13:2 = stablehlo.while(%iterArg_15 = %c_7, %iterArg_16 = %iterArg_12) : tensor<i64>, tensor<3xf32> attributes {enzyme.disable_mincut, enzymexla.checkpoint_segment}
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %16 = stablehlo.compare LT, %iterArg_15, %12 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %16 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %16 = stablehlo.add %iterArg_11, %iterArg_15 : tensor<i64>
// CHECK-NEXT:        %17 = stablehlo.multiply %16, %c_6 : tensor<i64>
// CHECK-NEXT:        %18 = stablehlo.add %c_6, %17 : tensor<i64>
// CHECK-NEXT:        %19 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:        %20 = stablehlo.multiply %cst_10, %iterArg_16 : tensor<3xf32>
// CHECK-NEXT:        %21 = stablehlo.cosine %20 : tensor<3xf32>
// CHECK-NEXT:        %22 = stablehlo.convert %19 : (tensor<3xi64>) -> tensor<3xf32>
// CHECK-NEXT:        %23 = stablehlo.multiply %22, %21 : tensor<3xf32>
// CHECK-NEXT:        %24 = stablehlo.add %iterArg_15, %c_8 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %24, %23 : tensor<i64>, tensor<3xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %14 = stablehlo.add %iterArg_11, %12 : tensor<i64>
// CHECK-NEXT:      %15 = stablehlo.add %iterArg, %c_8 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %15, %14, %13#1, %3, %5 : tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<3x3xf32>, tensor<3xi64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %1:5 = stablehlo.while(%iterArg = %c_7, %iterArg_11 = %c_6, %iterArg_12 = %cst_2, %iterArg_13 = %0#3, %iterArg_14 = %0#4) : tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<3x3xf32>, tensor<3xi64> attributes {enzyme.disable_mincut, enzymexla.checkpoint_segment}
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %2 = stablehlo.compare LT, %iterArg, %c_9 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %2 = stablehlo.subtract %iterArg_11, %c_8 : tensor<i64>
// CHECK-NEXT:      %3 = stablehlo.subtract %c_9, %iterArg : tensor<i64>
// CHECK-NEXT:      %4 = stablehlo.dynamic_slice %iterArg_13, %2, %c_7, sizes = [1, 3] : (tensor<3x3xf32>, tensor<i64>, tensor<i64>) -> tensor<1x3xf32>
// CHECK-NEXT:      %5 = stablehlo.reshape %4 : (tensor<1x3xf32>) -> tensor<3xf32>
// CHECK-NEXT:      %6 = stablehlo.dynamic_slice %iterArg_14, %2, sizes = [1] : (tensor<3xi64>, tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:      %7 = stablehlo.reshape %6 : (tensor<1xi64>) -> tensor<i64>
// CHECK-NEXT:      %8:5 = stablehlo.while(%iterArg_15 = %7, %iterArg_16 = %2, %iterArg_17 = %5, %iterArg_18 = %iterArg_13, %iterArg_19 = %iterArg_14) : tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<3x3xf32>, tensor<3xi64>
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %21 = stablehlo.add %iterArg_15, %c_8 : tensor<i64>
// CHECK-NEXT:        %22 = stablehlo.compare LT, %21, %3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %22 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %21 = stablehlo.subtract %3, %iterArg_15 : tensor<i64>
// CHECK-NEXT:        %22 = stablehlo.subtract %c_6, %iterArg_16 : tensor<i64>
// CHECK-NEXT:        %23 = stablehlo.minimum %22, %21 : tensor<i64>
// CHECK-NEXT:        %24 = stablehlo.compare LE, %21, %c_8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        %25 = stablehlo.compare LE, %23, %c_8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        %26 = stablehlo.or %24, %25 : tensor<i1>
// CHECK-NEXT:        %27 = "stablehlo.if"(%26) ({
// CHECK-NEXT:          stablehlo.return %21 : tensor<i64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %38:2 = stablehlo.while(%iterArg_20 = %c_7, %iterArg_21 = %c_8) : tensor<i64>, tensor<i64>
// CHECK-NEXT:          cond {
// CHECK-NEXT:            %54 = stablehlo.compare LT, %iterArg_21, %21 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:            stablehlo.return %54 : tensor<i1>
// CHECK-NEXT:          } do {
// CHECK-NEXT:            %54 = stablehlo.add %iterArg_20, %c_8 : tensor<i64>
// CHECK-NEXT:            %55 = stablehlo.add %23, %54 : tensor<i64>
// CHECK-NEXT:            %56 = stablehlo.multiply %iterArg_21, %55 : tensor<i64>
// CHECK-NEXT:            %57 = stablehlo.divide %56, %54 : tensor<i64>
// CHECK-NEXT:            stablehlo.return %54, %57 : tensor<i64>, tensor<i64>
// CHECK-NEXT:          }
// CHECK-NEXT:          %39 = stablehlo.add %23, %38#0 : tensor<i64>
// CHECK-NEXT:          %40 = stablehlo.multiply %38#1, %23 : tensor<i64>
// CHECK-NEXT:          %41 = stablehlo.divide %40, %39 : tensor<i64>
// CHECK-NEXT:          %42 = stablehlo.subtract %21, %41 : tensor<i64>
// CHECK-NEXT:          %43 = stablehlo.multiply %38#1, %38#0 : tensor<i64>
// CHECK-NEXT:          %44 = stablehlo.divide %43, %39 : tensor<i64>
// CHECK-NEXT:          %45 = stablehlo.maximum %42, %c_8 : tensor<i64>
// CHECK-NEXT:          %46 = stablehlo.subtract %21, %c_8 : tensor<i64>
// CHECK-NEXT:          %47 = stablehlo.minimum %44, %46 : tensor<i64>
// CHECK-NEXT:          %48 = stablehlo.add %45, %47 : tensor<i64>
// CHECK-NEXT:          %49 = stablehlo.divide %48, %c_3 : tensor<i64>
// CHECK-NEXT:          %50 = stablehlo.subtract %23, %c_8 : tensor<i64>
// CHECK-NEXT:          %51 = stablehlo.subtract %21, %50 : tensor<i64>
// CHECK-NEXT:          %52 = stablehlo.minimum %49, %51 : tensor<i64>
// CHECK-NEXT:          %53 = stablehlo.maximum %52, %c_8 : tensor<i64>
// CHECK-NEXT:          stablehlo.return %53 : tensor<i64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<i64>
// CHECK-NEXT:        %28 = stablehlo.reshape %iterArg_17 : (tensor<3xf32>) -> tensor<1x3xf32>
// CHECK-NEXT:        %29 = stablehlo.dynamic_update_slice %iterArg_18, %28, %iterArg_16, %c_7 : (tensor<3x3xf32>, tensor<1x3xf32>, tensor<i64>, tensor<i64>) -> tensor<3x3xf32>
// CHECK-NEXT:        %30 = stablehlo.reshape %iterArg_15 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:        %31 = stablehlo.dynamic_update_slice %iterArg_19, %30, %iterArg_16 : (tensor<3xi64>, tensor<1xi64>, tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:        %32 = stablehlo.add %iterArg_15, %27 : tensor<i64>
// CHECK-NEXT:        %33 = stablehlo.compare EQ, %32, %3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        %34 = stablehlo.subtract %32, %c_8 : tensor<i64>
// CHECK-NEXT:        %35 = stablehlo.select %33, %34, %32 : tensor<i1>, tensor<i64>
// CHECK-NEXT:        %36:2 = stablehlo.while(%iterArg_20 = %iterArg_15, %iterArg_21 = %iterArg_17) : tensor<i64>, tensor<3xf32> attributes {enzyme.disable_mincut, enzymexla.checkpoint_segment}
// CHECK-NEXT:        cond {
// CHECK-NEXT:          %38 = stablehlo.compare LT, %iterArg_20, %35 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:          stablehlo.return %38 : tensor<i1>
// CHECK-NEXT:        } do {
// CHECK-NEXT:          %38 = stablehlo.multiply %iterArg_20, %c_6 : tensor<i64>
// CHECK-NEXT:          %39 = stablehlo.add %c_6, %38 : tensor<i64>
// CHECK-NEXT:          %40 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:          %41 = stablehlo.multiply %cst_10, %iterArg_21 : tensor<3xf32>
// CHECK-NEXT:          %42 = stablehlo.cosine %41 : tensor<3xf32>
// CHECK-NEXT:          %43 = stablehlo.convert %40 : (tensor<3xi64>) -> tensor<3xf32>
// CHECK-NEXT:          %44 = stablehlo.multiply %43, %42 : tensor<3xf32>
// CHECK-NEXT:          %45 = stablehlo.add %iterArg_20, %c_8 : tensor<i64>
// CHECK-NEXT:          stablehlo.return %45, %44 : tensor<i64>, tensor<3xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        %37 = stablehlo.add %iterArg_16, %c_8 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %32, %37, %36#1, %29, %31 : tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<3x3xf32>, tensor<3xi64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %9 = stablehlo.subtract %3, %c_8 : tensor<i64>
// CHECK-NEXT:      %10 = stablehlo.multiply %9, %c_6 : tensor<i64>
// CHECK-NEXT:      %11 = stablehlo.add %c_6, %10 : tensor<i64>
// CHECK-NEXT:      %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:      %13 = stablehlo.multiply %cst_10, %8#2 : tensor<3xf32>
// CHECK-NEXT:      %14 = stablehlo.convert %12 : (tensor<3xi64>) -> tensor<3xf32>
// CHECK-NEXT:      %15 = stablehlo.multiply %iterArg_12, %14 : tensor<3xf32>
// CHECK-NEXT:      %16 = stablehlo.sine %13 : tensor<3xf32>
// CHECK-NEXT:      %17 = stablehlo.negate %16 : tensor<3xf32>
// CHECK-NEXT:      %18 = stablehlo.multiply %15, %17 : tensor<3xf32>
// CHECK-NEXT:      %19 = stablehlo.multiply %18, %cst_10 : tensor<3xf32>
// CHECK-NEXT:      %20 = stablehlo.add %iterArg, %c_8 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %20, %8#1, %19, %8#3, %8#4 : tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<3x3xf32>, tensor<3xi64>
// CHECK-NEXT:    }
// CHECK-NEXT:    check.expect_close %0#2, %cst, max_ulp_difference = 10 : tensor<3xf32>, tensor<3xf32>
// CHECK-NEXT:    check.expect_close %1#2, %cst_0, max_ulp_difference = 10 : tensor<3xf32>, tensor<3xf32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
