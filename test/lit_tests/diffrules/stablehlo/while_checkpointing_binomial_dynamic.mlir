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
// CHECK-NEXT:    %c = stablehlo.constant dense<9> : tensor<i64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<[3.000000e+01, -23.8084774, 26.8712101]> : tensor<3xf32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<[0.000000e+00, 3.26916253E+18, -3.63422884E+17]> : tensor<3xf32>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
// CHECK-NEXT:    %cst_3 = stablehlo.constant dense<[0.000000e+00, 0.785398185, 0.392699093]> : tensor<3xf32>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<3x3xf32>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<0> : tensor<3xi64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_8 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %cst_9 = stablehlo.constant dense<6.28318548> : tensor<3xf32>
// CHECK-NEXT:    %c_10 = stablehlo.constant dense<10> : tensor<i64>
// CHECK-NEXT:    %0:7 = stablehlo.while(%iterArg = %c_7, %iterArg_11 = %c_7, %iterArg_12 = %c_10, %iterArg_13 = %cst_3, %iterArg_14 = %c_6, %iterArg_15 = %cst_5, %iterArg_16 = %c_6) : tensor<i64>, tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<3xi64>, tensor<3x3xf32>, tensor<3xi64> attributes {enzyme.disable_mincut}
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %2 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %2 = stablehlo.reshape %iterArg_12 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:      %3 = stablehlo.dynamic_update_slice %iterArg_14, %2, %iterArg : (tensor<3xi64>, tensor<1xi64>, tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:      %4 = stablehlo.reshape %iterArg_13 : (tensor<3xf32>) -> tensor<1x3xf32>
// CHECK-NEXT:      %5 = stablehlo.dynamic_update_slice %iterArg_15, %4, %iterArg, %c_7 : (tensor<3x3xf32>, tensor<1x3xf32>, tensor<i64>, tensor<i64>) -> tensor<3x3xf32>
// CHECK-NEXT:      %6 = stablehlo.reshape %iterArg_11 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:      %7 = stablehlo.dynamic_update_slice %iterArg_16, %6, %iterArg : (tensor<3xi64>, tensor<1xi64>, tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:      %8 = stablehlo.subtract %c_10, %iterArg_11 : tensor<i64>
// CHECK-NEXT:      %9 = stablehlo.subtract %c_1, %iterArg : tensor<i64>
// CHECK-NEXT:      %10 = stablehlo.minimum %9, %8 : tensor<i64>
// CHECK-NEXT:      %11 = stablehlo.compare GE, %iterArg_11, %c : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      %12 = stablehlo.compare LE, %10, %c_8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      %13 = stablehlo.or %11, %12 : tensor<i1>
// CHECK-NEXT:      %14 = "stablehlo.if"(%13) ({
// CHECK-NEXT:        stablehlo.return %8 : tensor<i64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        %18:2 = stablehlo.while(%iterArg_17 = %c_7, %iterArg_18 = %c_8) : tensor<i64>, tensor<i64>
// CHECK-NEXT:        cond {
// CHECK-NEXT:          %34 = stablehlo.compare LT, %iterArg_18, %8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:          stablehlo.return %34 : tensor<i1>
// CHECK-NEXT:        } do {
// CHECK-NEXT:          %34 = stablehlo.add %iterArg_17, %c_8 : tensor<i64>
// CHECK-NEXT:          %35 = stablehlo.add %10, %34 : tensor<i64>
// CHECK-NEXT:          %36 = stablehlo.multiply %iterArg_18, %35 : tensor<i64>
// CHECK-NEXT:          %37 = stablehlo.divide %36, %34 : tensor<i64>
// CHECK-NEXT:          stablehlo.return %34, %37 : tensor<i64>, tensor<i64>
// CHECK-NEXT:        }
// CHECK-NEXT:        %19 = stablehlo.add %10, %18#0 : tensor<i64>
// CHECK-NEXT:        %20 = stablehlo.multiply %18#1, %10 : tensor<i64>
// CHECK-NEXT:        %21 = stablehlo.divide %20, %19 : tensor<i64>
// CHECK-NEXT:        %22 = stablehlo.subtract %8, %21 : tensor<i64>
// CHECK-NEXT:        %23 = stablehlo.multiply %18#1, %18#0 : tensor<i64>
// CHECK-NEXT:        %24 = stablehlo.divide %23, %19 : tensor<i64>
// CHECK-NEXT:        %25 = stablehlo.maximum %22, %c_8 : tensor<i64>
// CHECK-NEXT:        %26 = stablehlo.subtract %8, %c_8 : tensor<i64>
// CHECK-NEXT:        %27 = stablehlo.minimum %24, %26 : tensor<i64>
// CHECK-NEXT:        %28 = stablehlo.add %25, %27 : tensor<i64>
// CHECK-NEXT:        %29 = stablehlo.divide %28, %c_4 : tensor<i64>
// CHECK-NEXT:        %30 = stablehlo.subtract %10, %c_8 : tensor<i64>
// CHECK-NEXT:        %31 = stablehlo.subtract %8, %30 : tensor<i64>
// CHECK-NEXT:        %32 = stablehlo.minimum %29, %31 : tensor<i64>
// CHECK-NEXT:        %33 = stablehlo.maximum %32, %c_8 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %33 : tensor<i64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<i64>
// CHECK-NEXT:      %15:3 = stablehlo.while(%iterArg_17 = %c_7, %iterArg_18 = %iterArg_12, %iterArg_19 = %iterArg_13) : tensor<i64>, tensor<i64>, tensor<3xf32> attributes {enzyme.disable_mincut}
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %18 = stablehlo.compare LT, %iterArg_17, %14 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %18 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %18 = stablehlo.add %iterArg_11, %iterArg_17 : tensor<i64>
// CHECK-NEXT:        %19 = stablehlo.multiply %18, %c_1 : tensor<i64>
// CHECK-NEXT:        %20 = stablehlo.add %c_1, %19 : tensor<i64>
// CHECK-NEXT:        %21 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:        %22 = stablehlo.multiply %cst_9, %iterArg_19 : tensor<3xf32>
// CHECK-NEXT:        %23 = stablehlo.cosine %22 : tensor<3xf32>
// CHECK-NEXT:        %24 = stablehlo.convert %21 : (tensor<3xi64>) -> tensor<3xf32>
// CHECK-NEXT:        %25 = stablehlo.multiply %24, %23 : tensor<3xf32>
// CHECK-NEXT:        %26 = stablehlo.add %iterArg_17, %c_8 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %26, %iterArg_18, %25 : tensor<i64>, tensor<i64>, tensor<3xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %16 = stablehlo.add %iterArg_11, %14 : tensor<i64>
// CHECK-NEXT:      %17 = stablehlo.add %iterArg, %c_8 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %17, %16, %15#1, %15#2, %3, %5, %7 : tensor<i64>, tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<3xi64>, tensor<3x3xf32>, tensor<3xi64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %1:6 = stablehlo.while(%iterArg = %c_7, %iterArg_11 = %c_1, %iterArg_12 = %cst_2, %iterArg_13 = %0#4, %iterArg_14 = %0#5, %iterArg_15 = %0#6) : tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<3xi64>, tensor<3x3xf32>, tensor<3xi64> attributes {enzyme.disable_mincut}
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %2 = stablehlo.compare LT, %iterArg, %c_10 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %2 = stablehlo.subtract %iterArg_11, %c_8 : tensor<i64>
// CHECK-NEXT:      %3 = stablehlo.subtract %c_10, %iterArg : tensor<i64>
// CHECK-NEXT:      %4 = stablehlo.dynamic_slice %iterArg_13, %2, sizes = [1] : (tensor<3xi64>, tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:      %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
// CHECK-NEXT:      %6 = stablehlo.dynamic_slice %iterArg_14, %2, %c_7, sizes = [1, 3] : (tensor<3x3xf32>, tensor<i64>, tensor<i64>) -> tensor<1x3xf32>
// CHECK-NEXT:      %7 = stablehlo.reshape %6 : (tensor<1x3xf32>) -> tensor<3xf32>
// CHECK-NEXT:      %8 = stablehlo.dynamic_slice %iterArg_15, %2, sizes = [1] : (tensor<3xi64>, tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:      %9 = stablehlo.reshape %8 : (tensor<1xi64>) -> tensor<i64>
// CHECK-NEXT:      %10:7 = stablehlo.while(%iterArg_16 = %9, %iterArg_17 = %2, %iterArg_18 = %5, %iterArg_19 = %7, %iterArg_20 = %iterArg_13, %iterArg_21 = %iterArg_14, %iterArg_22 = %iterArg_15) : tensor<i64>, tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<3xi64>, tensor<3x3xf32>, tensor<3xi64>
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %23 = stablehlo.add %iterArg_16, %c_8 : tensor<i64>
// CHECK-NEXT:        %24 = stablehlo.compare LT, %23, %3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %24 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %23 = stablehlo.subtract %3, %iterArg_16 : tensor<i64>
// CHECK-NEXT:        %24 = stablehlo.subtract %c_1, %iterArg_17 : tensor<i64>
// CHECK-NEXT:        %25 = stablehlo.minimum %24, %23 : tensor<i64>
// CHECK-NEXT:        %26 = stablehlo.compare LE, %23, %c_8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        %27 = stablehlo.compare LE, %25, %c_8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        %28 = stablehlo.or %26, %27 : tensor<i1>
// CHECK-NEXT:        %29 = "stablehlo.if"(%28) ({
// CHECK-NEXT:          stablehlo.return %23 : tensor<i64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %42:2 = stablehlo.while(%iterArg_23 = %c_7, %iterArg_24 = %c_8) : tensor<i64>, tensor<i64>
// CHECK-NEXT:          cond {
// CHECK-NEXT:            %58 = stablehlo.compare LT, %iterArg_24, %23 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:            stablehlo.return %58 : tensor<i1>
// CHECK-NEXT:          } do {
// CHECK-NEXT:            %58 = stablehlo.add %iterArg_23, %c_8 : tensor<i64>
// CHECK-NEXT:            %59 = stablehlo.add %25, %58 : tensor<i64>
// CHECK-NEXT:            %60 = stablehlo.multiply %iterArg_24, %59 : tensor<i64>
// CHECK-NEXT:            %61 = stablehlo.divide %60, %58 : tensor<i64>
// CHECK-NEXT:            stablehlo.return %58, %61 : tensor<i64>, tensor<i64>
// CHECK-NEXT:          }
// CHECK-NEXT:          %43 = stablehlo.add %25, %42#0 : tensor<i64>
// CHECK-NEXT:          %44 = stablehlo.multiply %42#1, %25 : tensor<i64>
// CHECK-NEXT:          %45 = stablehlo.divide %44, %43 : tensor<i64>
// CHECK-NEXT:          %46 = stablehlo.subtract %23, %45 : tensor<i64>
// CHECK-NEXT:          %47 = stablehlo.multiply %42#1, %42#0 : tensor<i64>
// CHECK-NEXT:          %48 = stablehlo.divide %47, %43 : tensor<i64>
// CHECK-NEXT:          %49 = stablehlo.maximum %46, %c_8 : tensor<i64>
// CHECK-NEXT:          %50 = stablehlo.subtract %23, %c_8 : tensor<i64>
// CHECK-NEXT:          %51 = stablehlo.minimum %48, %50 : tensor<i64>
// CHECK-NEXT:          %52 = stablehlo.add %49, %51 : tensor<i64>
// CHECK-NEXT:          %53 = stablehlo.divide %52, %c_4 : tensor<i64>
// CHECK-NEXT:          %54 = stablehlo.subtract %25, %c_8 : tensor<i64>
// CHECK-NEXT:          %55 = stablehlo.subtract %23, %54 : tensor<i64>
// CHECK-NEXT:          %56 = stablehlo.minimum %53, %55 : tensor<i64>
// CHECK-NEXT:          %57 = stablehlo.maximum %56, %c_8 : tensor<i64>
// CHECK-NEXT:          stablehlo.return %57 : tensor<i64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<i64>
// CHECK-NEXT:        %30 = stablehlo.reshape %iterArg_18 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:        %31 = stablehlo.dynamic_update_slice %iterArg_20, %30, %iterArg_17 : (tensor<3xi64>, tensor<1xi64>, tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:        %32 = stablehlo.reshape %iterArg_19 : (tensor<3xf32>) -> tensor<1x3xf32>
// CHECK-NEXT:        %33 = stablehlo.dynamic_update_slice %iterArg_21, %32, %iterArg_17, %c_7 : (tensor<3x3xf32>, tensor<1x3xf32>, tensor<i64>, tensor<i64>) -> tensor<3x3xf32>
// CHECK-NEXT:        %34 = stablehlo.reshape %iterArg_16 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:        %35 = stablehlo.dynamic_update_slice %iterArg_22, %34, %iterArg_17 : (tensor<3xi64>, tensor<1xi64>, tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:        %36 = stablehlo.add %iterArg_16, %29 : tensor<i64>
// CHECK-NEXT:        %37 = stablehlo.compare EQ, %36, %3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        %38 = stablehlo.subtract %36, %c_8 : tensor<i64>
// CHECK-NEXT:        %39 = stablehlo.select %37, %38, %36 : tensor<i1>, tensor<i64>
// CHECK-NEXT:        %40:3 = stablehlo.while(%iterArg_23 = %iterArg_16, %iterArg_24 = %iterArg_18, %iterArg_25 = %iterArg_19) : tensor<i64>, tensor<i64>, tensor<3xf32> attributes {enzyme.disable_mincut}
// CHECK-NEXT:        cond {
// CHECK-NEXT:          %42 = stablehlo.compare LT, %iterArg_23, %39 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:          stablehlo.return %42 : tensor<i1>
// CHECK-NEXT:        } do {
// CHECK-NEXT:          %42 = stablehlo.multiply %iterArg_23, %c_1 : tensor<i64>
// CHECK-NEXT:          %43 = stablehlo.add %c_1, %42 : tensor<i64>
// CHECK-NEXT:          %44 = stablehlo.broadcast_in_dim %43, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:          %45 = stablehlo.multiply %cst_9, %iterArg_25 : tensor<3xf32>
// CHECK-NEXT:          %46 = stablehlo.cosine %45 : tensor<3xf32>
// CHECK-NEXT:          %47 = stablehlo.convert %44 : (tensor<3xi64>) -> tensor<3xf32>
// CHECK-NEXT:          %48 = stablehlo.multiply %47, %46 : tensor<3xf32>
// CHECK-NEXT:          %49 = stablehlo.add %iterArg_23, %c_8 : tensor<i64>
// CHECK-NEXT:          stablehlo.return %49, %iterArg_24, %48 : tensor<i64>, tensor<i64>, tensor<3xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        %41 = stablehlo.add %iterArg_17, %c_8 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %36, %41, %40#1, %40#2, %31, %33, %35 : tensor<i64>, tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<3xi64>, tensor<3x3xf32>, tensor<3xi64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %11 = stablehlo.subtract %3, %c_8 : tensor<i64>
// CHECK-NEXT:      %12 = stablehlo.multiply %11, %c_1 : tensor<i64>
// CHECK-NEXT:      %13 = stablehlo.add %c_1, %12 : tensor<i64>
// CHECK-NEXT:      %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:      %15 = stablehlo.multiply %cst_9, %10#3 : tensor<3xf32>
// CHECK-NEXT:      %16 = stablehlo.convert %14 : (tensor<3xi64>) -> tensor<3xf32>
// CHECK-NEXT:      %17 = stablehlo.multiply %iterArg_12, %16 : tensor<3xf32>
// CHECK-NEXT:      %18 = stablehlo.sine %15 : tensor<3xf32>
// CHECK-NEXT:      %19 = stablehlo.negate %18 : tensor<3xf32>
// CHECK-NEXT:      %20 = stablehlo.multiply %17, %19 : tensor<3xf32>
// CHECK-NEXT:      %21 = stablehlo.multiply %20, %cst_9 : tensor<3xf32>
// CHECK-NEXT:      %22 = stablehlo.add %iterArg, %c_8 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %22, %10#1, %21, %10#4, %10#5, %10#6 : tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<3xi64>, tensor<3x3xf32>, tensor<3xi64>
// CHECK-NEXT:    }
// CHECK-NEXT:    check.expect_close %0#3, %cst, max_ulp_difference = 10 : tensor<3xf32>, tensor<3xf32>
// CHECK-NEXT:    check.expect_close %1#2, %cst_0, max_ulp_difference = 10 : tensor<3xf32>, tensor<3xf32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
