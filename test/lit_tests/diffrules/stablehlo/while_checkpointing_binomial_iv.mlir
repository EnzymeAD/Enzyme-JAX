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
    %1:6 = stablehlo.while(%iterArg = %c_2, %iterArg_3 = %c_1, %iterArg_4 = %c_0, %iterArg_5 = %c, %iterArg_6 = %0, %iterArg_7 = %cst) : tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<f32> attributes {enzyme.disable_mincut, enzymexla.binomial_checkpointing, enzymexla.checkpoint_period = 3 : i64, enzymexla.enable_checkpointing = true}
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
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<[3.000000e+01, -23.8084774, 26.8712101]> : tensor<3xf32>
// CHECK-NEXT:    %cst_1 = stablehlo.constant dense<[0.000000e+00, 3.26916253E+18, -3.63422884E+17]> : tensor<3xf32>
// CHECK-NEXT:    %cst_2 = stablehlo.constant dense<[0.000000e+00, 0.785398185, 0.392699093]> : tensor<3xf32>
// CHECK-NEXT:    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<3xi64>
// CHECK-NEXT:    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<3x3xf32>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_8 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_9 = stablehlo.constant dense<10> : tensor<i64>
// CHECK-NEXT:    %cst_10 = stablehlo.constant dense<6.28318548> : tensor<3xf32>
// CHECK-NEXT:    %0:5 = stablehlo.while(%iterArg = %c_7, %iterArg_11 = %cst_2, %iterArg_12 = %cst_5, %iterArg_13 = %c_7, %iterArg_14 = %c_4) : tensor<i64>, tensor<3xf32>, tensor<3x3xf32>, tensor<i64>, tensor<3xi64>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %2 = stablehlo.compare LT, %iterArg, %c_6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %2 = stablehlo.reshape %iterArg_11 : (tensor<3xf32>) -> tensor<1x3xf32>
// CHECK-NEXT:      %3 = stablehlo.dynamic_update_slice %iterArg_12, %2, %iterArg, %c_7 : (tensor<3x3xf32>, tensor<1x3xf32>, tensor<i64>, tensor<i64>) -> tensor<3x3xf32>
// CHECK-NEXT:      %4 = stablehlo.subtract %c_9, %iterArg_13 : tensor<i64>
// CHECK-NEXT:      %5 = stablehlo.subtract %c_6, %iterArg : tensor<i64>
// CHECK-DAG:       %[[ns1:.+]] = stablehlo.compare EQ, %iterArg_13, %c : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-DAG:       %[[b1:.+]] = stablehlo.compare EQ, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      %[[or1:.+]] = stablehlo.or %[[ns1]], %[[b1]] : tensor<i1>
// CHECK-NEXT:      %9 = "stablehlo.if"(%[[or1]]) ({
// CHECK-NEXT:        stablehlo.return %c_8 : tensor<i64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        %15:2 = stablehlo.while(%iterArg_15 = %c_8, %iterArg_16 = %5) : tensor<i64>, tensor<i64>
// CHECK-NEXT:        cond {
// CHECK-NEXT:          %19 = stablehlo.compare LT, %iterArg_16, %4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:          stablehlo.return %19 : tensor<i1>
// CHECK-NEXT:        } do {
// CHECK-NEXT:          %19 = stablehlo.add %iterArg_15, %c_8 : tensor<i64>
// CHECK-NEXT:          %20 = stablehlo.add %19, %5 : tensor<i64>
// CHECK-NEXT:          %21 = stablehlo.subtract %20, %c_8 : tensor<i64>
// CHECK-NEXT:          %22 = stablehlo.multiply %iterArg_16, %21 : tensor<i64>
// CHECK-NEXT:          %23 = stablehlo.divide %22, %19 : tensor<i64>
// CHECK-NEXT:          stablehlo.return %19, %23 : tensor<i64>, tensor<i64>
// CHECK-NEXT:        }
// CHECK-DAG:         %[[sub3:.+]] = stablehlo.subtract %15#0, %c_8 : tensor<i64>
// CHECK-DAG:         %[[cmp3:.+]] = stablehlo.compare EQ, %15#1, %4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        %18 = stablehlo.select %[[cmp3]], %15#0, %[[sub3]] : tensor<i1>, tensor<i64>
// CHECK-NEXT:        stablehlo.return %18 : tensor<i64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<i64>
// CHECK-NEXT:      %10 = stablehlo.add %iterArg_13, %9 : tensor<i64>
// CHECK-NEXT:      %11 = stablehlo.reshape %iterArg_13 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:      %12 = stablehlo.dynamic_update_slice %iterArg_14, %11, %iterArg : (tensor<3xi64>, tensor<1xi64>, tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:      %13:2 = stablehlo.while(%iterArg_15 = %c_7, %iterArg_16 = %iterArg_11) : tensor<i64>, tensor<3xf32> attributes {enzyme.disable_mincut}
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %15 = stablehlo.compare LT, %iterArg_15, %9 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %15 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %15 = stablehlo.add %iterArg_13, %iterArg_15 : tensor<i64>
// CHECK-NEXT:        %16 = stablehlo.multiply %15, %c_6 : tensor<i64>
// CHECK-NEXT:        %17 = stablehlo.add %c_6, %16 : tensor<i64>
// CHECK-NEXT:        %18 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:        %19 = stablehlo.multiply %cst_10, %iterArg_16 : tensor<3xf32>
// CHECK-NEXT:        %20 = stablehlo.cosine %19 : tensor<3xf32>
// CHECK-NEXT:        %21 = stablehlo.convert %18 : (tensor<3xi64>) -> tensor<3xf32>
// CHECK-NEXT:        %22 = stablehlo.multiply %21, %20 : tensor<3xf32>
// CHECK-NEXT:        %23 = stablehlo.add %iterArg_15, %c_8 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %23, %22 : tensor<i64>, tensor<3xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %14 = stablehlo.add %iterArg, %c_8 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %14, %13#1, %3, %10, %12 : tensor<i64>, tensor<3xf32>, tensor<3x3xf32>, tensor<i64>, tensor<3xi64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %1:5 = stablehlo.while(%iterArg = %c_7, %iterArg_11 = %c_6, %iterArg_12 = %0#4, %iterArg_13 = %0#2, %iterArg_14 = %cst_3) : tensor<i64>, tensor<i64>, tensor<3xi64>, tensor<3x3xf32>, tensor<3xf32>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %2 = stablehlo.compare LT, %iterArg, %c_9 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %2 = stablehlo.add %iterArg, %c_8 : tensor<i64>
// CHECK-NEXT:      %3 = stablehlo.subtract %iterArg_11, %c_8 : tensor<i64>
// CHECK-NEXT:      %4 = stablehlo.dynamic_slice %iterArg_13, %3, %c_7, sizes = [1, 3] : (tensor<3x3xf32>, tensor<i64>, tensor<i64>) -> tensor<1x3xf32>
// CHECK-NEXT:      %5 = stablehlo.reshape %4 : (tensor<1x3xf32>) -> tensor<3xf32>
// CHECK-NEXT:      %6 = stablehlo.dynamic_slice %iterArg_12, %3, sizes = [1] : (tensor<3xi64>, tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:      %7 = stablehlo.reshape %6 : (tensor<1xi64>) -> tensor<i64>
// CHECK-NEXT:      %8 = stablehlo.subtract %c_9, %iterArg : tensor<i64>
// CHECK-NEXT:      %9:5 = stablehlo.while(%iterArg_15 = %7, %iterArg_16 = %3, %iterArg_17 = %iterArg_12, %iterArg_18 = %5, %iterArg_19 = %iterArg_13) : tensor<i64>, tensor<i64>, tensor<3xi64>, tensor<3xf32>, tensor<3x3xf32>
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %21 = stablehlo.add %iterArg_15, %c_8 : tensor<i64>
// CHECK-NEXT:        %22 = stablehlo.compare LT, %21, %8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %22 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %21 = stablehlo.subtract %8, %iterArg_15 : tensor<i64>
// CHECK-NEXT:        %22 = stablehlo.subtract %c_6, %iterArg_16 : tensor<i64>
// CHECK-DAG:         %[[ns2:.+]] = stablehlo.compare EQ, %21, %c_8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-DAG:         %[[b2:.+]] = stablehlo.compare EQ, %iterArg_16, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        %[[or2:.+]] = stablehlo.or %[[ns2]], %[[b2]] : tensor<i1>
// CHECK-NEXT:        %26 = "stablehlo.if"(%[[or2]]) ({
// CHECK-NEXT:          stablehlo.return %c_8 : tensor<i64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %37:2 = stablehlo.while(%iterArg_20 = %c_8, %iterArg_21 = %22) : tensor<i64>, tensor<i64>
// CHECK-NEXT:          cond {
// CHECK-NEXT:            %41 = stablehlo.compare LT, %iterArg_21, %21 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:            stablehlo.return %41 : tensor<i1>
// CHECK-NEXT:          } do {
// CHECK-NEXT:            %41 = stablehlo.add %iterArg_20, %c_8 : tensor<i64>
// CHECK-NEXT:            %42 = stablehlo.add %41, %22 : tensor<i64>
// CHECK-NEXT:            %43 = stablehlo.subtract %42, %c_8 : tensor<i64>
// CHECK-NEXT:            %44 = stablehlo.multiply %iterArg_21, %43 : tensor<i64>
// CHECK-NEXT:            %45 = stablehlo.divide %44, %41 : tensor<i64>
// CHECK-NEXT:            stablehlo.return %41, %45 : tensor<i64>, tensor<i64>
// CHECK-NEXT:          }
// CHECK-DAG:           %[[sub4:.+]] = stablehlo.subtract %37#0, %c_8 : tensor<i64>
// CHECK-DAG:           %[[cmp4:.+]] = stablehlo.compare EQ, %37#1, %21 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:          %40 = stablehlo.select %[[cmp4]], %37#0, %[[sub4]] : tensor<i1>, tensor<i64>
// CHECK-NEXT:          stablehlo.return %40 : tensor<i64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<i64>
// CHECK-NEXT:        %27 = stablehlo.reshape %iterArg_18 : (tensor<3xf32>) -> tensor<1x3xf32>
// CHECK-NEXT:        %28 = stablehlo.dynamic_update_slice %iterArg_19, %27, %iterArg_16, %c_7 : (tensor<3x3xf32>, tensor<1x3xf32>, tensor<i64>, tensor<i64>) -> tensor<3x3xf32>
// CHECK-NEXT:        %29 = stablehlo.reshape %iterArg_15 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:        %30 = stablehlo.dynamic_update_slice %iterArg_17, %29, %iterArg_16 : (tensor<3xi64>, tensor<1xi64>, tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:        %31 = stablehlo.add %iterArg_15, %26 : tensor<i64>
// CHECK-NEXT:        %32 = stablehlo.compare EQ, %31, %8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        %33 = stablehlo.convert %32 : (tensor<i1>) -> tensor<i64>
// CHECK-NEXT:        %34 = stablehlo.subtract %31, %33 : tensor<i64>
// CHECK-NEXT:        %35:2 = stablehlo.while(%iterArg_20 = %iterArg_15, %iterArg_21 = %iterArg_18) : tensor<i64>, tensor<3xf32> attributes {enzyme.disable_mincut}
// CHECK-NEXT:        cond {
// CHECK-NEXT:          %37 = stablehlo.compare LT, %iterArg_20, %34 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:          stablehlo.return %37 : tensor<i1>
// CHECK-NEXT:        } do {
// CHECK-NEXT:          %37 = stablehlo.multiply %iterArg_20, %c_6 : tensor<i64>
// CHECK-NEXT:          %38 = stablehlo.add %c_6, %37 : tensor<i64>
// CHECK-NEXT:          %39 = stablehlo.broadcast_in_dim %38, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:          %40 = stablehlo.multiply %cst_10, %iterArg_21 : tensor<3xf32>
// CHECK-NEXT:          %41 = stablehlo.cosine %40 : tensor<3xf32>
// CHECK-NEXT:          %42 = stablehlo.convert %39 : (tensor<3xi64>) -> tensor<3xf32>
// CHECK-NEXT:          %43 = stablehlo.multiply %42, %41 : tensor<3xf32>
// CHECK-NEXT:          %44 = stablehlo.add %iterArg_20, %c_8 : tensor<i64>
// CHECK-NEXT:          stablehlo.return %44, %43 : tensor<i64>, tensor<3xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        %36 = stablehlo.add %iterArg_16, %c_8 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %31, %36, %30, %35#1, %28 : tensor<i64>, tensor<i64>, tensor<3xi64>, tensor<3xf32>, tensor<3x3xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %10 = stablehlo.subtract %8, %c_8 : tensor<i64>
// CHECK-NEXT:      %11 = stablehlo.multiply %10, %c_6 : tensor<i64>
// CHECK-NEXT:      %12 = stablehlo.add %c_6, %11 : tensor<i64>
// CHECK-NEXT:      %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:      %14 = stablehlo.multiply %cst_10, %9#3 : tensor<3xf32>
// CHECK-NEXT:      %15 = stablehlo.convert %13 : (tensor<3xi64>) -> tensor<3xf32>
// CHECK-NEXT:      %16 = stablehlo.multiply %iterArg_14, %15 : tensor<3xf32>
// CHECK-NEXT:      %17 = stablehlo.sine %14 : tensor<3xf32>
// CHECK-NEXT:      %18 = stablehlo.negate %17 : tensor<3xf32>
// CHECK-NEXT:      %19 = stablehlo.multiply %16, %18 : tensor<3xf32>
// CHECK-NEXT:      %20 = stablehlo.multiply %19, %cst_10 : tensor<3xf32>
// CHECK-NEXT:      stablehlo.return %2, %9#1, %9#2, %9#4, %20 : tensor<i64>, tensor<i64>, tensor<3xi64>, tensor<3x3xf32>, tensor<3xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    check.expect_close %0#1, %cst, max_ulp_difference = 10 : tensor<3xf32>, tensor<3xf32>
// CHECK-NEXT:    check.expect_close %1#4, %cst_1, max_ulp_difference = 10 : tensor<3xf32>, tensor<3xf32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

