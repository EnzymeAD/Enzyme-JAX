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
  func.func private @"*_broadcast_scalar_1"(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>) {
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<f32>
    return %0, %arg0, %arg1 : tensor<f32>, tensor<f32>, tensor<f32>
  }
  func.func private @"Const{typeof(myf)}(Main.myf)_autodiff"(%arg0: tensor<3xf32>) -> tensor<3xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    %cst = stablehlo.constant dense<6.28318548> : tensor<f32>
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_0 = stablehlo.constant dense<5> : tensor<i64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %1:6 = stablehlo.while(%iterArg = %c_2, %iterArg_3 = %c, %iterArg_4 = %c_1, %iterArg_5 = %c_0, %iterArg_6 = %cst, %iterArg_7 = %0) : tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<f32>, tensor<3xf32> attributes {enzyme.disable_mincut, enzymexla.binomial_checkpointing, enzymexla.checkpoint_period = 2 : i64, enzymexla.enable_checkpointing = true}
    cond {
      %3 = stablehlo.subtract %iterArg_5, %iterArg_3 : tensor<i64>
      %4 = stablehlo.divide %3, %iterArg_4 : tensor<i64>
      %c_8 = stablehlo.constant dense<1> : tensor<i64>
      %5 = stablehlo.add %4, %c_8 : tensor<i64>
      %6 = stablehlo.compare LT, %iterArg, %5 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %6 : tensor<i1>
    } do {
      %3 = stablehlo.multiply %iterArg, %iterArg_4 : tensor<i64>
      %4 = stablehlo.add %iterArg_3, %3 : tensor<i64>
      %c_8 = stablehlo.constant dense<1> : tensor<i64>
      %5 = stablehlo.add %iterArg, %c_8 : tensor<i64>
      %6 = stablehlo.broadcast_in_dim %iterArg_7, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
      %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<3xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<3xf32>
      %8 = stablehlo.broadcast_in_dim %iterArg_6, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %9 = stablehlo.broadcast_in_dim %iterArg_7, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
      %10 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
      %11:3 = enzyme.batch @"*_broadcast_scalar"(%8, %10) {batch_shape = array<i64: 3>} : (tensor<3xf32>, tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>, tensor<3xf32>)
      %12 = stablehlo.broadcast_in_dim %11#0, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
      %13:2 = enzyme.batch @cos_broadcast_scalar(%12) {batch_shape = array<i64: 3>} : (tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>)
      %14 = stablehlo.broadcast_in_dim %13#0, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
      %15:3 = enzyme.batch @"*_broadcast_scalar_1"(%7, %14) {batch_shape = array<i64: 3>} : (tensor<3xf32>, tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>, tensor<3xf32>)
      stablehlo.return %5, %iterArg_3, %iterArg_4, %iterArg_5, %iterArg_6, %15#0 : tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<f32>, tensor<3xf32>
    }
    %2 = stablehlo.transpose %1#5, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    return %2 : tensor<3xf32>
  }
  func.func private @fwd(%arg0: tensor<3xf32> {tf.aliasing_output = 2 : i32}) -> (tensor<3xf32>, tensor<3xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<3xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
    %2 = stablehlo.add %cst_2, %1 : tensor<3xf32>
    %3 = stablehlo.transpose %0, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    %4 = stablehlo.transpose %2, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    %5:2 = enzyme.autodiff @"Const{typeof(myf)}(Main.myf)_autodiff"(%3, %4) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_active>]} : (tensor<3xf32>, tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>)
    %6 = stablehlo.transpose %5#0, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    %7 = stablehlo.transpose %5#1, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    %8 = stablehlo.transpose %6, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    %9 = stablehlo.transpose %7, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    return %8, %9 : tensor<3xf32>, tensor<3xf32>
  }
  func.func @main() {
    %expected_y = stablehlo.constant dense<[0.0, 0.0585665, 0.0650583]> : tensor<3xf32>
    %expected_dx = stablehlo.constant dense<[1.0, -0.870949, 0.75936]> : tensor<3xf32>
    %cst = stablehlo.constant dense<[0.0, 0.7853982, 0.3926991]> : tensor<3xf32>
    %y, %dx = func.call @fwd(%cst) : (tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>)
    check.expect_close %y, %expected_y, max_ulp_difference = 10, min_ulp_difference = 0 : tensor<3xf32>, tensor<3xf32>
    check.expect_close %dx, %expected_dx, max_ulp_difference = 10, min_ulp_difference = 0 : tensor<3xf32>, tensor<3xf32>
    return
  }
}


// CHECK:  func.func @main() {
// CHECK-NEXT:    %c = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<[0.000000e+00, 5.856650e-02, 6.505830e-02]> : tensor<3xf32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<[1.000000e+00, -0.87094897, 7.593600e-01]> : tensor<3xf32>
// CHECK-NEXT:    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
// CHECK-NEXT:    %cst_2 = stablehlo.constant dense<[0.000000e+00, 0.785398185, 0.392699093]> : tensor<3xf32>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<2xi64>
// CHECK-NEXT:    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<2x3xf32>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<5> : tensor<i64>
// CHECK-NEXT:    %c_8 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %cst_9 = stablehlo.constant dense<6.28318548> : tensor<3xf32>
// CHECK-NEXT:    %0:5 = stablehlo.while(%iterArg = %c_8, %iterArg_10 = %cst_2, %iterArg_11 = %cst_5, %iterArg_12 = %c_8, %iterArg_13 = %c_4) : tensor<i64>, tensor<3xf32>, tensor<2x3xf32>, tensor<i64>, tensor<2xi64>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %2 = stablehlo.compare LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %2 = stablehlo.reshape %iterArg_10 : (tensor<3xf32>) -> tensor<1x3xf32>
// CHECK-NEXT:      %3 = stablehlo.dynamic_update_slice %iterArg_11, %2, %iterArg, %c_8 : (tensor<2x3xf32>, tensor<1x3xf32>, tensor<i64>, tensor<i64>) -> tensor<2x3xf32>
// CHECK-NEXT:      %4 = stablehlo.subtract %c_7, %iterArg_12 : tensor<i64>
// CHECK-NEXT:      %5 = stablehlo.subtract %c_3, %iterArg : tensor<i64>
// CHECK-DAG:       %[[ns1:.+]] = stablehlo.compare EQ, %iterArg_12, %c : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-DAG:       %[[b1:.+]] = stablehlo.compare EQ, %iterArg, %c_6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      %[[or1:.+]] = stablehlo.or %[[ns1]], %[[b1]] : tensor<i1>
// CHECK-NEXT:      %9 = "stablehlo.if"(%[[or1]]) ({
// CHECK-NEXT:        stablehlo.return %c_6 : tensor<i64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        %15:2 = stablehlo.while(%iterArg_14 = %c_6, %iterArg_15 = %5) : tensor<i64>, tensor<i64>
// CHECK-NEXT:        cond {
// CHECK-NEXT:          %19 = stablehlo.compare LT, %iterArg_15, %4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:          stablehlo.return %19 : tensor<i1>
// CHECK-NEXT:        } do {
// CHECK-NEXT:          %19 = stablehlo.add %iterArg_14, %c_6 : tensor<i64>
// CHECK-NEXT:          %20 = stablehlo.add %19, %5 : tensor<i64>
// CHECK-NEXT:          %21 = stablehlo.subtract %20, %c_6 : tensor<i64>
// CHECK-NEXT:          %22 = stablehlo.multiply %iterArg_15, %21 : tensor<i64>
// CHECK-NEXT:          %23 = stablehlo.divide %22, %19 : tensor<i64>
// CHECK-NEXT:          stablehlo.return %19, %23 : tensor<i64>, tensor<i64>
// CHECK-NEXT:        }
// CHECK-DAG:         %[[sub3:.+]] = stablehlo.subtract %15#0, %c_6 : tensor<i64>
// CHECK-DAG:         %[[cmp3:.+]] = stablehlo.compare EQ, %15#1, %4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        %18 = stablehlo.select %[[cmp3]], %15#0, %[[sub3]] : tensor<i1>, tensor<i64>
// CHECK-NEXT:        stablehlo.return %18 : tensor<i64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<i64>
// CHECK-NEXT:      %10 = stablehlo.add %iterArg_12, %9 : tensor<i64>
// CHECK-NEXT:      %11 = stablehlo.reshape %iterArg_12 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:      %12 = stablehlo.dynamic_update_slice %iterArg_13, %11, %iterArg : (tensor<2xi64>, tensor<1xi64>, tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:      %13:2 = stablehlo.while(%iterArg_14 = %c_8, %iterArg_15 = %iterArg_10) : tensor<i64>, tensor<3xf32> attributes {enzyme.disable_mincut}
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %15 = stablehlo.compare LT, %iterArg_14, %9 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %15 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %15 = stablehlo.multiply %cst_9, %iterArg_15 : tensor<3xf32>
// CHECK-NEXT:        %16 = stablehlo.cosine %15 : tensor<3xf32>
// CHECK-NEXT:        %17 = stablehlo.multiply %iterArg_15, %16 : tensor<3xf32>
// CHECK-NEXT:        %18 = stablehlo.add %iterArg_14, %c_6 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %18, %17 : tensor<i64>, tensor<3xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %14 = stablehlo.add %iterArg, %c_6 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %14, %13#1, %3, %10, %12 : tensor<i64>, tensor<3xf32>, tensor<2x3xf32>, tensor<i64>, tensor<2xi64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %1:5 = stablehlo.while(%iterArg = %c_8, %iterArg_10 = %c_3, %iterArg_11 = %0#4, %iterArg_12 = %0#2, %iterArg_13 = %cst_1) : tensor<i64>, tensor<i64>, tensor<2xi64>, tensor<2x3xf32>, tensor<3xf32>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %2 = stablehlo.compare LT, %iterArg, %c_7 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %2 = stablehlo.add %iterArg, %c_6 : tensor<i64>
// CHECK-NEXT:      %3 = stablehlo.subtract %iterArg_10, %c_6 : tensor<i64>
// CHECK-NEXT:      %4 = stablehlo.dynamic_slice %iterArg_12, %3, %c_8, sizes = [1, 3] : (tensor<2x3xf32>, tensor<i64>, tensor<i64>) -> tensor<1x3xf32>
// CHECK-NEXT:      %5 = stablehlo.reshape %4 : (tensor<1x3xf32>) -> tensor<3xf32>
// CHECK-NEXT:      %6 = stablehlo.dynamic_slice %iterArg_11, %3, sizes = [1] : (tensor<2xi64>, tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:      %7 = stablehlo.reshape %6 : (tensor<1xi64>) -> tensor<i64>
// CHECK-NEXT:      %8 = stablehlo.subtract %c_7, %iterArg : tensor<i64>
// CHECK-NEXT:      %9:5 = stablehlo.while(%iterArg_14 = %7, %iterArg_15 = %3, %iterArg_16 = %iterArg_11, %iterArg_17 = %5, %iterArg_18 = %iterArg_12) : tensor<i64>, tensor<i64>, tensor<2xi64>, tensor<3xf32>, tensor<2x3xf32>
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %19 = stablehlo.add %iterArg_14, %c_6 : tensor<i64>
// CHECK-NEXT:        %20 = stablehlo.compare LT, %19, %8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %20 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %19 = stablehlo.subtract %8, %iterArg_14 : tensor<i64>
// CHECK-NEXT:        %20 = stablehlo.subtract %c_3, %iterArg_15 : tensor<i64>
// CHECK-DAG:         %[[ns2:.+]] = stablehlo.compare EQ, %19, %c_6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-DAG:         %[[b2:.+]] = stablehlo.compare EQ, %iterArg_15, %c_6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        %[[or2:.+]] = stablehlo.or %[[ns2]], %[[b2]] : tensor<i1>
// CHECK-NEXT:        %24 = "stablehlo.if"(%[[or2]]) ({
// CHECK-NEXT:          stablehlo.return %c_6 : tensor<i64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %35:2 = stablehlo.while(%iterArg_19 = %c_6, %iterArg_20 = %20) : tensor<i64>, tensor<i64>
// CHECK-NEXT:          cond {
// CHECK-NEXT:            %39 = stablehlo.compare LT, %iterArg_20, %19 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:            stablehlo.return %39 : tensor<i1>
// CHECK-NEXT:          } do {
// CHECK-NEXT:            %39 = stablehlo.add %iterArg_19, %c_6 : tensor<i64>
// CHECK-NEXT:            %40 = stablehlo.add %39, %20 : tensor<i64>
// CHECK-NEXT:            %41 = stablehlo.subtract %40, %c_6 : tensor<i64>
// CHECK-NEXT:            %42 = stablehlo.multiply %iterArg_20, %41 : tensor<i64>
// CHECK-NEXT:            %43 = stablehlo.divide %42, %39 : tensor<i64>
// CHECK-NEXT:            stablehlo.return %39, %43 : tensor<i64>, tensor<i64>
// CHECK-NEXT:          }
// CHECK-DAG:           %[[sub4:.+]] = stablehlo.subtract %35#0, %c_6 : tensor<i64>
// CHECK-DAG:           %[[cmp4:.+]] = stablehlo.compare EQ, %35#1, %19 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:          %38 = stablehlo.select %[[cmp4]], %35#0, %[[sub4]] : tensor<i1>, tensor<i64>
// CHECK-NEXT:          stablehlo.return %38 : tensor<i64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<i64>
// CHECK-NEXT:        %25 = stablehlo.reshape %iterArg_17 : (tensor<3xf32>) -> tensor<1x3xf32>
// CHECK-NEXT:        %26 = stablehlo.dynamic_update_slice %iterArg_18, %25, %iterArg_15, %c_8 : (tensor<2x3xf32>, tensor<1x3xf32>, tensor<i64>, tensor<i64>) -> tensor<2x3xf32>
// CHECK-NEXT:        %27 = stablehlo.reshape %iterArg_14 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:        %28 = stablehlo.dynamic_update_slice %iterArg_16, %27, %iterArg_15 : (tensor<2xi64>, tensor<1xi64>, tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:        %29 = stablehlo.add %iterArg_14, %24 : tensor<i64>
// CHECK-NEXT:        %30 = stablehlo.compare EQ, %29, %8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        %31 = stablehlo.convert %30 : (tensor<i1>) -> tensor<i64>
// CHECK-NEXT:        %32 = stablehlo.subtract %29, %31 : tensor<i64>
// CHECK-NEXT:        %33:2 = stablehlo.while(%iterArg_19 = %iterArg_14, %iterArg_20 = %iterArg_17) : tensor<i64>, tensor<3xf32> attributes {enzyme.disable_mincut}
// CHECK-NEXT:        cond {
// CHECK-NEXT:          %35 = stablehlo.compare LT, %iterArg_19, %32 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:          stablehlo.return %35 : tensor<i1>
// CHECK-NEXT:        } do {
// CHECK-NEXT:          %35 = stablehlo.multiply %cst_9, %iterArg_20 : tensor<3xf32>
// CHECK-NEXT:          %36 = stablehlo.cosine %35 : tensor<3xf32>
// CHECK-NEXT:          %37 = stablehlo.multiply %iterArg_20, %36 : tensor<3xf32>
// CHECK-NEXT:          %38 = stablehlo.add %iterArg_19, %c_6 : tensor<i64>
// CHECK-NEXT:          stablehlo.return %38, %37 : tensor<i64>, tensor<3xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        %34 = stablehlo.add %iterArg_15, %c_6 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %29, %34, %28, %33#1, %26 : tensor<i64>, tensor<i64>, tensor<2xi64>, tensor<3xf32>, tensor<2x3xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %10 = stablehlo.multiply %cst_9, %9#3 : tensor<3xf32>
// CHECK-NEXT:      %11 = stablehlo.cosine %10 : tensor<3xf32>
// CHECK-NEXT:      %12 = stablehlo.multiply %iterArg_13, %11 : tensor<3xf32>
// CHECK-NEXT:      %13 = stablehlo.multiply %iterArg_13, %9#3 : tensor<3xf32>
// CHECK-NEXT:      %14 = stablehlo.sine %10 : tensor<3xf32>
// CHECK-NEXT:      %15 = stablehlo.negate %14 : tensor<3xf32>
// CHECK-NEXT:      %16 = stablehlo.multiply %13, %15 : tensor<3xf32>
// CHECK-NEXT:      %17 = stablehlo.multiply %16, %cst_9 : tensor<3xf32>
// CHECK-NEXT:      %18 = stablehlo.add %12, %17 : tensor<3xf32>
// CHECK-NEXT:      stablehlo.return %2, %9#1, %9#2, %9#4, %18 : tensor<i64>, tensor<i64>, tensor<2xi64>, tensor<2x3xf32>, tensor<3xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    check.expect_close %0#1, %cst, max_ulp_difference = 10 : tensor<3xf32>, tensor<3xf32>
// CHECK-NEXT:    check.expect_close %1#4, %cst_0, max_ulp_difference = 10 : tensor<3xf32>, tensor<3xf32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
