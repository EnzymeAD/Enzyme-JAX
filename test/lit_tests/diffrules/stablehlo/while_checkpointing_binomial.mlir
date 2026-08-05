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
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<0> : tensor<2xi64>
// CHECK-NEXT:    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<2x3xf32>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<5> : tensor<i64>
// CHECK-NEXT:    %c_8 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %cst_9 = stablehlo.constant dense<6.28318548> : tensor<3xf32>
// CHECK-NEXT:    %0:5 = stablehlo.while(%iterArg = %c_8, %iterArg_10 = %c_8, %iterArg_11 = %cst_2, %iterArg_12 = %cst_4, %iterArg_13 = %c_3) : tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<2x3xf32>, tensor<2xi64> attributes {enzyme.disable_mincut}
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %2 = stablehlo.compare LT, %iterArg, %c_5 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %2 = stablehlo.reshape %iterArg_11 : (tensor<3xf32>) -> tensor<1x3xf32>
// CHECK-NEXT:      %3 = stablehlo.dynamic_update_slice %iterArg_12, %2, %iterArg, %c_8 : (tensor<2x3xf32>, tensor<1x3xf32>, tensor<i64>, tensor<i64>) -> tensor<2x3xf32>
// CHECK-NEXT:      %4 = stablehlo.reshape %iterArg_10 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:      %5 = stablehlo.dynamic_update_slice %iterArg_13, %4, %iterArg : (tensor<2xi64>, tensor<1xi64>, tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:      %6 = stablehlo.subtract %c_7, %iterArg_10 : tensor<i64>
// CHECK-NEXT:      %7 = stablehlo.subtract %c_5, %iterArg : tensor<i64>
// CHECK-NEXT:      %8 = stablehlo.minimum %7, %6 : tensor<i64>
// CHECK-NEXT:      %9 = stablehlo.compare GE, %iterArg_10, %c : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      %10 = stablehlo.compare LE, %8, %c_6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      %11 = stablehlo.or %9, %10 : tensor<i1>
// CHECK-NEXT:      %12 = "stablehlo.if"(%11) ({
// CHECK-NEXT:        stablehlo.return %6 : tensor<i64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        %16:2 = stablehlo.while(%iterArg_14 = %c_8, %iterArg_15 = %c_6) : tensor<i64>, tensor<i64>
// CHECK-NEXT:        cond {
// CHECK-NEXT:          %32 = stablehlo.compare LT, %iterArg_15, %6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:          stablehlo.return %32 : tensor<i1>
// CHECK-NEXT:        } do {
// CHECK-NEXT:          %32 = stablehlo.add %iterArg_14, %c_6 : tensor<i64>
// CHECK-NEXT:          %33 = stablehlo.add %8, %32 : tensor<i64>
// CHECK-NEXT:          %34 = stablehlo.multiply %iterArg_15, %33 : tensor<i64>
// CHECK-NEXT:          %35 = stablehlo.divide %34, %32 : tensor<i64>
// CHECK-NEXT:          stablehlo.return %32, %35 : tensor<i64>, tensor<i64>
// CHECK-NEXT:        }
// CHECK-NEXT:        %17 = stablehlo.add %8, %16#0 : tensor<i64>
// CHECK-NEXT:        %18 = stablehlo.multiply %16#1, %8 : tensor<i64>
// CHECK-NEXT:        %19 = stablehlo.divide %18, %17 : tensor<i64>
// CHECK-NEXT:        %20 = stablehlo.subtract %6, %19 : tensor<i64>
// CHECK-NEXT:        %21 = stablehlo.multiply %16#1, %16#0 : tensor<i64>
// CHECK-NEXT:        %22 = stablehlo.divide %21, %17 : tensor<i64>
// CHECK-NEXT:        %23 = stablehlo.maximum %20, %c_6 : tensor<i64>
// CHECK-NEXT:        %24 = stablehlo.subtract %6, %c_6 : tensor<i64>
// CHECK-NEXT:        %25 = stablehlo.minimum %22, %24 : tensor<i64>
// CHECK-NEXT:        %26 = stablehlo.add %23, %25 : tensor<i64>
// CHECK-NEXT:        %27 = stablehlo.divide %26, %c_5 : tensor<i64>
// CHECK-NEXT:        %28 = stablehlo.subtract %8, %c_6 : tensor<i64>
// CHECK-NEXT:        %29 = stablehlo.subtract %6, %28 : tensor<i64>
// CHECK-NEXT:        %30 = stablehlo.minimum %27, %29 : tensor<i64>
// CHECK-NEXT:        %31 = stablehlo.maximum %30, %c_6 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %31 : tensor<i64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<i64>
// CHECK-NEXT:      %13:2 = stablehlo.while(%iterArg_14 = %c_8, %iterArg_15 = %iterArg_11) : tensor<i64>, tensor<3xf32> attributes {enzyme.disable_mincut}
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %16 = stablehlo.compare LT, %iterArg_14, %12 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %16 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %16 = stablehlo.multiply %cst_9, %iterArg_15 : tensor<3xf32>
// CHECK-NEXT:        %17 = stablehlo.cosine %16 : tensor<3xf32>
// CHECK-NEXT:        %18 = stablehlo.multiply %iterArg_15, %17 : tensor<3xf32>
// CHECK-NEXT:        %19 = stablehlo.add %iterArg_14, %c_6 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %19, %18 : tensor<i64>, tensor<3xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %14 = stablehlo.add %iterArg_10, %12 : tensor<i64>
// CHECK-NEXT:      %15 = stablehlo.add %iterArg, %c_6 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %15, %14, %13#1, %3, %5 : tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<2x3xf32>, tensor<2xi64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %1:5 = stablehlo.while(%iterArg = %c_8, %iterArg_10 = %c_5, %iterArg_11 = %cst_1, %iterArg_12 = %0#3, %iterArg_13 = %0#4) : tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<2x3xf32>, tensor<2xi64> attributes {enzyme.disable_mincut}
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %2 = stablehlo.compare LT, %iterArg, %c_7 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %2 = stablehlo.subtract %iterArg_10, %c_6 : tensor<i64>
// CHECK-NEXT:      %3 = stablehlo.subtract %c_7, %iterArg : tensor<i64>
// CHECK-NEXT:      %4 = stablehlo.dynamic_slice %iterArg_12, %2, %c_8, sizes = [1, 3] : (tensor<2x3xf32>, tensor<i64>, tensor<i64>) -> tensor<1x3xf32>
// CHECK-NEXT:      %5 = stablehlo.reshape %4 : (tensor<1x3xf32>) -> tensor<3xf32>
// CHECK-NEXT:      %6 = stablehlo.dynamic_slice %iterArg_13, %2, sizes = [1] : (tensor<2xi64>, tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:      %7 = stablehlo.reshape %6 : (tensor<1xi64>) -> tensor<i64>
// CHECK-NEXT:      %8:5 = stablehlo.while(%iterArg_14 = %7, %iterArg_15 = %2, %iterArg_16 = %5, %iterArg_17 = %iterArg_12, %iterArg_18 = %iterArg_13) : tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<2x3xf32>, tensor<2xi64>
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %19 = stablehlo.add %iterArg_14, %c_6 : tensor<i64>
// CHECK-NEXT:        %20 = stablehlo.compare LT, %19, %3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %20 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %19 = stablehlo.subtract %3, %iterArg_14 : tensor<i64>
// CHECK-NEXT:        %20 = stablehlo.subtract %c_5, %iterArg_15 : tensor<i64>
// CHECK-NEXT:        %21 = stablehlo.minimum %20, %19 : tensor<i64>
// CHECK-NEXT:        %22 = stablehlo.compare LE, %19, %c_6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        %23 = stablehlo.compare LE, %21, %c_6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        %24 = stablehlo.or %22, %23 : tensor<i1>
// CHECK-NEXT:        %25 = "stablehlo.if"(%24) ({
// CHECK-NEXT:          stablehlo.return %19 : tensor<i64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %36:2 = stablehlo.while(%iterArg_19 = %c_8, %iterArg_20 = %c_6) : tensor<i64>, tensor<i64>
// CHECK-NEXT:          cond {
// CHECK-NEXT:            %52 = stablehlo.compare LT, %iterArg_20, %19 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:            stablehlo.return %52 : tensor<i1>
// CHECK-NEXT:          } do {
// CHECK-NEXT:            %52 = stablehlo.add %iterArg_19, %c_6 : tensor<i64>
// CHECK-NEXT:            %53 = stablehlo.add %21, %52 : tensor<i64>
// CHECK-NEXT:            %54 = stablehlo.multiply %iterArg_20, %53 : tensor<i64>
// CHECK-NEXT:            %55 = stablehlo.divide %54, %52 : tensor<i64>
// CHECK-NEXT:            stablehlo.return %52, %55 : tensor<i64>, tensor<i64>
// CHECK-NEXT:          }
// CHECK-NEXT:          %37 = stablehlo.add %21, %36#0 : tensor<i64>
// CHECK-NEXT:          %38 = stablehlo.multiply %36#1, %21 : tensor<i64>
// CHECK-NEXT:          %39 = stablehlo.divide %38, %37 : tensor<i64>
// CHECK-NEXT:          %40 = stablehlo.subtract %19, %39 : tensor<i64>
// CHECK-NEXT:          %41 = stablehlo.multiply %36#1, %36#0 : tensor<i64>
// CHECK-NEXT:          %42 = stablehlo.divide %41, %37 : tensor<i64>
// CHECK-NEXT:          %43 = stablehlo.maximum %40, %c_6 : tensor<i64>
// CHECK-NEXT:          %44 = stablehlo.subtract %19, %c_6 : tensor<i64>
// CHECK-NEXT:          %45 = stablehlo.minimum %42, %44 : tensor<i64>
// CHECK-NEXT:          %46 = stablehlo.add %43, %45 : tensor<i64>
// CHECK-NEXT:          %47 = stablehlo.divide %46, %c_5 : tensor<i64>
// CHECK-NEXT:          %48 = stablehlo.subtract %21, %c_6 : tensor<i64>
// CHECK-NEXT:          %49 = stablehlo.subtract %19, %48 : tensor<i64>
// CHECK-NEXT:          %50 = stablehlo.minimum %47, %49 : tensor<i64>
// CHECK-NEXT:          %51 = stablehlo.maximum %50, %c_6 : tensor<i64>
// CHECK-NEXT:          stablehlo.return %51 : tensor<i64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<i64>
// CHECK-NEXT:        %26 = stablehlo.reshape %iterArg_16 : (tensor<3xf32>) -> tensor<1x3xf32>
// CHECK-NEXT:        %27 = stablehlo.dynamic_update_slice %iterArg_17, %26, %iterArg_15, %c_8 : (tensor<2x3xf32>, tensor<1x3xf32>, tensor<i64>, tensor<i64>) -> tensor<2x3xf32>
// CHECK-NEXT:        %28 = stablehlo.reshape %iterArg_14 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:        %29 = stablehlo.dynamic_update_slice %iterArg_18, %28, %iterArg_15 : (tensor<2xi64>, tensor<1xi64>, tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:        %30 = stablehlo.add %iterArg_14, %25 : tensor<i64>
// CHECK-NEXT:        %31 = stablehlo.compare EQ, %30, %3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        %32 = stablehlo.subtract %30, %c_6 : tensor<i64>
// CHECK-NEXT:        %33 = stablehlo.select %31, %32, %30 : tensor<i1>, tensor<i64>
// CHECK-NEXT:        %34:2 = stablehlo.while(%iterArg_19 = %iterArg_14, %iterArg_20 = %iterArg_16) : tensor<i64>, tensor<3xf32> attributes {enzyme.disable_mincut}
// CHECK-NEXT:        cond {
// CHECK-NEXT:          %36 = stablehlo.compare LT, %iterArg_19, %33 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:          stablehlo.return %36 : tensor<i1>
// CHECK-NEXT:        } do {
// CHECK-NEXT:          %36 = stablehlo.multiply %cst_9, %iterArg_20 : tensor<3xf32>
// CHECK-NEXT:          %37 = stablehlo.cosine %36 : tensor<3xf32>
// CHECK-NEXT:          %38 = stablehlo.multiply %iterArg_20, %37 : tensor<3xf32>
// CHECK-NEXT:          %39 = stablehlo.add %iterArg_19, %c_6 : tensor<i64>
// CHECK-NEXT:          stablehlo.return %39, %38 : tensor<i64>, tensor<3xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        %35 = stablehlo.add %iterArg_15, %c_6 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %30, %35, %34#1, %27, %29 : tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<2x3xf32>, tensor<2xi64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %9 = stablehlo.multiply %cst_9, %8#2 : tensor<3xf32>
// CHECK-NEXT:      %10 = stablehlo.cosine %9 : tensor<3xf32>
// CHECK-NEXT:      %11 = stablehlo.multiply %iterArg_11, %10 : tensor<3xf32>
// CHECK-NEXT:      %12 = stablehlo.multiply %iterArg_11, %8#2 : tensor<3xf32>
// CHECK-NEXT:      %13 = stablehlo.sine %9 : tensor<3xf32>
// CHECK-NEXT:      %14 = stablehlo.negate %13 : tensor<3xf32>
// CHECK-NEXT:      %15 = stablehlo.multiply %12, %14 : tensor<3xf32>
// CHECK-NEXT:      %16 = stablehlo.multiply %15, %cst_9 : tensor<3xf32>
// CHECK-NEXT:      %17 = stablehlo.add %11, %16 : tensor<3xf32>
// CHECK-NEXT:      %18 = stablehlo.add %iterArg, %c_6 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %18, %8#1, %17, %8#3, %8#4 : tensor<i64>, tensor<i64>, tensor<3xf32>, tensor<2x3xf32>, tensor<2xi64>
// CHECK-NEXT:    }
// CHECK-NEXT:    check.expect_close %0#2, %cst, max_ulp_difference = 10 : tensor<3xf32>, tensor<3xf32>
// CHECK-NEXT:    check.expect_close %1#2, %cst_0, max_ulp_difference = 10 : tensor<3xf32>, tensor<3xf32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
