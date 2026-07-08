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
  func.func private @"Const{typeof(myf)}(Main.myf)_autodiff"(%arg0: tensor<3xf32>, %arg1: tensor<i64>, %arg2: tensor<i64>, %arg3: tensor<i64>) -> (tensor<3xf32>, tensor<i64>, tensor<i64>, tensor<i64>) {
    %0 = stablehlo.transpose %arg0, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    %cst = stablehlo.constant dense<6.28318548> : tensor<f32>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %1 = stablehlo.subtract %arg2, %arg1 : tensor<i64>
    %2 = stablehlo.divide %1, %arg3 : tensor<i64>
    %c_0 = stablehlo.constant dense<1> : tensor<i64>
    %3 = stablehlo.add %2, %c_0 : tensor<i64>
    %4:6 = stablehlo.while(%iterArg = %c, %iterArg_1 = %3, %iterArg_2 = %arg3, %iterArg_3 = %0, %iterArg_4 = %cst, %iterArg_5 = %arg1) : tensor<i64>, tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<f32>, tensor<i64> attributes {enzyme.disable_mincut, enzymexla.binomial_checkpointing, enzymexla.checkpoint_period = 3 : i64, enzymexla.enable_checkpointing = true}
    cond {
      %6 = stablehlo.compare LT, %iterArg, %iterArg_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %6 : tensor<i1>
    } do {
      %6 = stablehlo.multiply %iterArg, %iterArg_2 : tensor<i64>
      %7 = stablehlo.add %iterArg_5, %6 : tensor<i64>
      %c_6 = stablehlo.constant dense<1> : tensor<i64>
      %8 = stablehlo.add %iterArg, %c_6 : tensor<i64>
      %9 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<i64>) -> tensor<3xi64>
      %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<3xf32>
      %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<3xf32>
      %10 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %11 = stablehlo.broadcast_in_dim %iterArg_3, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
      %12 = stablehlo.broadcast_in_dim %11, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
      %13:3 = enzyme.batch @"*_broadcast_scalar"(%10, %12) {batch_shape = array<i64: 3>} : (tensor<3xf32>, tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>, tensor<3xf32>)
      %14 = stablehlo.broadcast_in_dim %13#0, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
      %15:2 = enzyme.batch @cos_broadcast_scalar(%14) {batch_shape = array<i64: 3>} : (tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>)
      %16 = stablehlo.broadcast_in_dim %15#0, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
      %17:3 = enzyme.batch @"*_broadcast_scalar_1"(%9, %16) {batch_shape = array<i64: 3>} : (tensor<3xi64>, tensor<3xf32>) -> (tensor<3xf32>, tensor<3xi64>, tensor<3xf32>)
      stablehlo.return %8, %iterArg_1, %iterArg_2, %17#0, %iterArg_4, %iterArg_5 : tensor<i64>, tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<f32>, tensor<i64>
    }
    %5 = stablehlo.transpose %4#3, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    return %5, %4#5, %arg2, %4#2 : tensor<3xf32>, tensor<i64>, tensor<i64>, tensor<i64>
  }
  func.func private @f(%arg0: tensor<3xf32> {tf.aliasing_output = 2 : i32},
                       %arg1: tensor<i64> {tf.aliasing_output = 3 : i32},
                       %arg2: tensor<i64> {tf.aliasing_output = 4 : i32},
                       %arg3: tensor<i64> {tf.aliasing_output = 5 : i32}) -> (tensor<3xf32>,
                                                                              tensor<3xf32>,
                                                                              tensor<3xf32>, tensor<i64>, tensor<i64>, tensor<i64>) {
    %0 = stablehlo.transpose %arg0, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<3xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
    %2 = stablehlo.add %cst_2, %1 : tensor<3xf32>
    %3 = stablehlo.transpose %0, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    %4 = stablehlo.transpose %2, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    %5:5 = enzyme.autodiff @"Const{typeof(myf)}(Main.myf)_autodiff"(%3, %arg1, %arg2, %arg3, %4) {activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_const>], ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_const>]} : (tensor<3xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<3xf32>) -> (tensor<3xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<3xf32>)
    %6 = stablehlo.transpose %5#0, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    %7 = stablehlo.transpose %5#4, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    %8 = stablehlo.transpose %6, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    %9 = stablehlo.transpose %7, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    %10 = stablehlo.transpose %0, dims = [0] : (tensor<3xf32>) -> tensor<3xf32>
    return %5#0, %5#4, %arg0, %5#1, %5#2, %5#3 : tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<i64>, tensor<i64>, tensor<i64>
  }
  func.func @main() {
    %expected_y = stablehlo.constant dense<[30.0, -23.808477, 26.87121]> : tensor<3xf32>
    %expected_dx = stablehlo.constant dense<[0.0, 3.2691625e18, -3.634229e17]> : tensor<3xf32>
    %x = stablehlo.constant dense<[0.0, 0.7853982, 0.3926991]> : tensor<3xf32>

    %start = stablehlo.constant dense<3> : tensor<i64>
    %limit = stablehlo.constant dense<30> : tensor<i64>
    %step = stablehlo.constant dense<3> : tensor<i64>

    %result:6 = func.call @f(%x, %start, %limit, %step) : (tensor<3xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> (tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<i64>, tensor<i64>, tensor<i64>)

    check.expect_close %result#0, %expected_y, max_ulp_difference = 10, min_ulp_difference = 0 : tensor<3xf32>, tensor<3xf32>
    check.expect_close %result#1, %expected_dx, max_ulp_difference = 10, min_ulp_difference = 0 : tensor<3xf32>, tensor<3xf32>
    return
  }
}

// CHECK:  func.func @main() {
// CHECK-DAG:    %[[nine:.+]] = stablehlo.constant dense<9> : tensor<i64>
// CHECK-DAG:    %[[two:.+]] = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<[3.000000e+01, -23.8084774, 26.8712101]> : tensor<3xf32>
// CHECK-NEXT:    %cst_1 = stablehlo.constant dense<[0.000000e+00, 3.26916253E+18, -3.63422884E+17]> : tensor<3xf32>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
// CHECK-NEXT:    %cst_4 = stablehlo.constant dense<[0.000000e+00, 0.785398185, 0.392699093]> : tensor<3xf32>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<0> : tensor<3xi64>
// CHECK-NEXT:    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<3x3xf32>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_8 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %cst_9 = stablehlo.constant dense<6.28318548> : tensor<3xf32>
// CHECK-NEXT:    %c_10 = stablehlo.constant dense<10> : tensor<i64>
// CHECK-NEXT:    %0:7 = stablehlo.while(%iterArg = %c_7, %iterArg_11 = %c_10, %iterArg_12 = %cst_4, %iterArg_13 = %c_5, %iterArg_14 = %cst_6, %iterArg_15 = %c_7, %iterArg_16 = %c_5) : tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<3xi64>, tensor<3x3xf32>, tensor<i64>, tensor<3xi64>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %2 = stablehlo.compare LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %2 = stablehlo.reshape %iterArg_11 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:      %3 = stablehlo.dynamic_update_slice %iterArg_13, %2, %iterArg : (tensor<3xi64>, tensor<1xi64>, tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:      %4 = stablehlo.reshape %iterArg_12 : (tensor<3xf32>) -> tensor<1x3xf32>
// CHECK-NEXT:      %5 = stablehlo.dynamic_update_slice %iterArg_14, %4, %iterArg, %c_7 : (tensor<3x3xf32>, tensor<1x3xf32>, tensor<i64>, tensor<i64>) -> tensor<3x3xf32>
// CHECK-NEXT:      %6 = stablehlo.subtract %c_10, %iterArg_15 : tensor<i64>
// CHECK-NEXT:      %7 = stablehlo.subtract %c_2, %iterArg : tensor<i64>
// CHECK-DAG:       %[[ns1:.+]] = stablehlo.compare EQ, %iterArg_15, %[[nine]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-DAG:       %[[b1:.+]] = stablehlo.compare EQ, %iterArg, %[[two]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      %[[or1:.+]] = stablehlo.or %[[ns1]], %[[b1]] : tensor<i1>
// CHECK-NEXT:      %11 = "stablehlo.if"(%[[or1]]) ({
// CHECK-NEXT:        stablehlo.return %c_8 : tensor<i64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        %17:2 = stablehlo.while(%iterArg_17 = %c_8, %iterArg_18 = %7) : tensor<i64>, tensor<i64>
// CHECK-NEXT:        cond {
// CHECK-NEXT:          %21 = stablehlo.compare LT, %iterArg_18, %6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:          stablehlo.return %21 : tensor<i1>
// CHECK-NEXT:        } do {
// CHECK-NEXT:          %21 = stablehlo.add %iterArg_17, %c_8 : tensor<i64>
// CHECK-NEXT:          %22 = stablehlo.add %21, %7 : tensor<i64>
// CHECK-NEXT:          %23 = stablehlo.subtract %22, %c_8 : tensor<i64>
// CHECK-NEXT:          %24 = stablehlo.multiply %iterArg_18, %23 : tensor<i64>
// CHECK-NEXT:          %25 = stablehlo.divide %24, %21 : tensor<i64>
// CHECK-NEXT:          stablehlo.return %21, %25 : tensor<i64>, tensor<i64>
// CHECK-NEXT:        }
// CHECK-DAG:         %[[sub3:.+]] = stablehlo.subtract %17#0, %c_8 : tensor<i64>
// CHECK-DAG:         %[[cmp3:.+]] = stablehlo.compare EQ, %17#1, %6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        %20 = stablehlo.select %[[cmp3]], %17#0, %[[sub3]] : tensor<i1>, tensor<i64>
// CHECK-NEXT:        stablehlo.return %20 : tensor<i64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<i64>
// CHECK-NEXT:      %12 = stablehlo.add %iterArg_15, %11 : tensor<i64>
// CHECK-NEXT:      %13 = stablehlo.reshape %iterArg_15 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:      %14 = stablehlo.dynamic_update_slice %iterArg_16, %13, %iterArg : (tensor<3xi64>, tensor<1xi64>, tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:      %15:3 = stablehlo.while(%iterArg_17 = %c_7, %iterArg_18 = %iterArg_11, %iterArg_19 = %iterArg_12) : tensor<i64>, tensor<i64>, tensor<3xf32> attributes {enzyme.disable_mincut}
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %17 = stablehlo.compare LT, %iterArg_17, %11 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %17 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %17 = stablehlo.add %iterArg_15, %iterArg_17 : tensor<i64>
// CHECK-NEXT:        %18 = stablehlo.multiply %17, %c_2 : tensor<i64>
// CHECK-NEXT:        %19 = stablehlo.add %c_2, %18 : tensor<i64>
// CHECK-NEXT:        %20 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:        %21 = stablehlo.multiply %cst_9, %iterArg_19 : tensor<3xf32>
// CHECK-NEXT:        %22 = stablehlo.cosine %21 : tensor<3xf32>
// CHECK-NEXT:        %23 = stablehlo.convert %20 : (tensor<3xi64>) -> tensor<3xf32>
// CHECK-NEXT:        %24 = stablehlo.multiply %23, %22 : tensor<3xf32>
// CHECK-NEXT:        %25 = stablehlo.add %iterArg_17, %c_8 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %25, %iterArg_18, %24 : tensor<i64>, tensor<i64>, tensor<3xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %16 = stablehlo.add %iterArg, %c_8 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %16, %15#1, %15#2, %3, %5, %12, %14 : tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<3xi64>, tensor<3x3xf32>, tensor<i64>, tensor<3xi64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %1:6 = stablehlo.while(%iterArg = %c_7, %iterArg_11 = %c_2, %iterArg_12 = %0#6, %iterArg_13 = %0#3, %iterArg_14 = %0#4, %iterArg_15 = %cst_3) : tensor<i64>, tensor<i64>, tensor<3xi64>, tensor<3xi64>, tensor<3x3xf32>, tensor<3xf32>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %2 = stablehlo.compare LT, %iterArg, %c_10 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %2 = stablehlo.add %iterArg, %c_8 : tensor<i64>
// CHECK-NEXT:      %3 = stablehlo.subtract %iterArg_11, %c_8 : tensor<i64>
// CHECK-NEXT:      %4 = stablehlo.dynamic_slice %iterArg_13, %3, sizes = [1] : (tensor<3xi64>, tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:      %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
// CHECK-NEXT:      %6 = stablehlo.dynamic_slice %iterArg_14, %3, %c_7, sizes = [1, 3] : (tensor<3x3xf32>, tensor<i64>, tensor<i64>) -> tensor<1x3xf32>
// CHECK-NEXT:      %7 = stablehlo.reshape %6 : (tensor<1x3xf32>) -> tensor<3xf32>
// CHECK-NEXT:      %8 = stablehlo.dynamic_slice %iterArg_12, %3, sizes = [1] : (tensor<3xi64>, tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:      %9 = stablehlo.reshape %8 : (tensor<1xi64>) -> tensor<i64>
// CHECK-NEXT:      %10 = stablehlo.subtract %c_10, %iterArg : tensor<i64>
// CHECK-NEXT:      %11:7 = stablehlo.while(%iterArg_16 = %9, %iterArg_17 = %3, %iterArg_18 = %iterArg_12, %iterArg_19 = %5, %iterArg_20 = %7, %iterArg_21 = %iterArg_13, %iterArg_22 = %iterArg_14) : tensor<i64>, tensor<i64>, tensor<3xi64>, tensor<i64>, tensor<3xf32>, tensor<3xi64>, tensor<3x3xf32>
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %23 = stablehlo.add %iterArg_16, %c_8 : tensor<i64>
// CHECK-NEXT:        %24 = stablehlo.compare LT, %23, %10 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %24 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %23 = stablehlo.subtract %10, %iterArg_16 : tensor<i64>
// CHECK-NEXT:        %24 = stablehlo.subtract %c_2, %iterArg_17 : tensor<i64>
// CHECK-DAG:         %[[ns2:.+]] = stablehlo.compare EQ, %23, %c_8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-DAG:         %[[b2:.+]] = stablehlo.compare EQ, %iterArg_17, %[[two]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        %[[or2:.+]] = stablehlo.or %[[ns2]], %[[b2]] : tensor<i1>
// CHECK-NEXT:        %28 = "stablehlo.if"(%[[or2]]) ({
// CHECK-NEXT:          stablehlo.return %c_8 : tensor<i64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %41:2 = stablehlo.while(%iterArg_23 = %c_8, %iterArg_24 = %24) : tensor<i64>, tensor<i64>
// CHECK-NEXT:          cond {
// CHECK-NEXT:            %45 = stablehlo.compare LT, %iterArg_24, %23 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:            stablehlo.return %45 : tensor<i1>
// CHECK-NEXT:          } do {
// CHECK-NEXT:            %45 = stablehlo.add %iterArg_23, %c_8 : tensor<i64>
// CHECK-NEXT:            %46 = stablehlo.add %45, %24 : tensor<i64>
// CHECK-NEXT:            %47 = stablehlo.subtract %46, %c_8 : tensor<i64>
// CHECK-NEXT:            %48 = stablehlo.multiply %iterArg_24, %47 : tensor<i64>
// CHECK-NEXT:            %49 = stablehlo.divide %48, %45 : tensor<i64>
// CHECK-NEXT:            stablehlo.return %45, %49 : tensor<i64>, tensor<i64>
// CHECK-NEXT:          }
// CHECK-DAG:           %[[sub4:.+]] = stablehlo.subtract %41#0, %c_8 : tensor<i64>
// CHECK-DAG:           %[[cmp4:.+]] = stablehlo.compare EQ, %41#1, %23 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:          %44 = stablehlo.select %[[cmp4]], %41#0, %[[sub4]] : tensor<i1>, tensor<i64>
// CHECK-NEXT:          stablehlo.return %44 : tensor<i64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<i64>
// CHECK-NEXT:        %29 = stablehlo.reshape %iterArg_19 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:        %30 = stablehlo.dynamic_update_slice %iterArg_21, %29, %iterArg_17 : (tensor<3xi64>, tensor<1xi64>, tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:        %31 = stablehlo.reshape %iterArg_20 : (tensor<3xf32>) -> tensor<1x3xf32>
// CHECK-NEXT:        %32 = stablehlo.dynamic_update_slice %iterArg_22, %31, %iterArg_17, %c_7 : (tensor<3x3xf32>, tensor<1x3xf32>, tensor<i64>, tensor<i64>) -> tensor<3x3xf32>
// CHECK-NEXT:        %33 = stablehlo.reshape %iterArg_16 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:        %34 = stablehlo.dynamic_update_slice %iterArg_18, %33, %iterArg_17 : (tensor<3xi64>, tensor<1xi64>, tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:        %35 = stablehlo.add %iterArg_16, %28 : tensor<i64>
// CHECK-NEXT:        %36 = stablehlo.compare EQ, %35, %10 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        %37 = stablehlo.convert %36 : (tensor<i1>) -> tensor<i64>
// CHECK-NEXT:        %38 = stablehlo.subtract %35, %37 : tensor<i64>
// CHECK-NEXT:        %39:3 = stablehlo.while(%iterArg_23 = %iterArg_16, %iterArg_24 = %iterArg_19, %iterArg_25 = %iterArg_20) : tensor<i64>, tensor<i64>, tensor<3xf32> attributes {enzyme.disable_mincut}
// CHECK-NEXT:        cond {
// CHECK-NEXT:          %41 = stablehlo.compare LT, %iterArg_23, %38 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:          stablehlo.return %41 : tensor<i1>
// CHECK-NEXT:        } do {
// CHECK-NEXT:          %41 = stablehlo.multiply %iterArg_23, %c_2 : tensor<i64>
// CHECK-NEXT:          %42 = stablehlo.add %c_2, %41 : tensor<i64>
// CHECK-NEXT:          %43 = stablehlo.broadcast_in_dim %42, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:          %44 = stablehlo.multiply %cst_9, %iterArg_25 : tensor<3xf32>
// CHECK-NEXT:          %45 = stablehlo.cosine %44 : tensor<3xf32>
// CHECK-NEXT:          %46 = stablehlo.convert %43 : (tensor<3xi64>) -> tensor<3xf32>
// CHECK-NEXT:          %47 = stablehlo.multiply %46, %45 : tensor<3xf32>
// CHECK-NEXT:          %48 = stablehlo.add %iterArg_23, %c_8 : tensor<i64>
// CHECK-NEXT:          stablehlo.return %48, %iterArg_24, %47 : tensor<i64>, tensor<i64>, tensor<3xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        %40 = stablehlo.add %iterArg_17, %c_8 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %35, %40, %34, %39#1, %39#2, %30, %32 : tensor<i64>, tensor<i64>, tensor<3xi64>, tensor<i64>, tensor<3xf32>, tensor<3xi64>, tensor<3x3xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %12 = stablehlo.subtract %10, %c_8 : tensor<i64>
// CHECK-NEXT:      %13 = stablehlo.multiply %12, %c_2 : tensor<i64>
// CHECK-NEXT:      %14 = stablehlo.add %c_2, %13 : tensor<i64>
// CHECK-NEXT:      %15 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:      %16 = stablehlo.multiply %cst_9, %11#4 : tensor<3xf32>
// CHECK-NEXT:      %17 = stablehlo.convert %15 : (tensor<3xi64>) -> tensor<3xf32>
// CHECK-NEXT:      %18 = stablehlo.multiply %iterArg_15, %17 : tensor<3xf32>
// CHECK-NEXT:      %19 = stablehlo.sine %16 : tensor<3xf32>
// CHECK-NEXT:      %20 = stablehlo.negate %19 : tensor<3xf32>
// CHECK-NEXT:      %21 = stablehlo.multiply %18, %20 : tensor<3xf32>
// CHECK-NEXT:      %22 = stablehlo.multiply %21, %cst_9 : tensor<3xf32>
// CHECK-NEXT:      stablehlo.return %2, %11#1, %11#2, %11#5, %11#6, %22 : tensor<i64>, tensor<i64>, tensor<3xi64>, tensor<3xi64>, tensor<3x3xf32>, tensor<3xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    check.expect_close %0#2, %cst, max_ulp_difference = 10 : tensor<3xf32>, tensor<3xf32>
// CHECK-NEXT:    check.expect_close %1#5, %cst_1, max_ulp_difference = 10 : tensor<3xf32>, tensor<3xf32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
