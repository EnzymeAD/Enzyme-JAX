// RUN: enzymexlamlir-opt %s --enzyme-batch --enzyme-hlo-opt '--enzyme=postpasses=arith-raise' --canonicalize | FileCheck %s
module @reactant_gradient attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func private @identity_broadcast_scalar(%arg0: tensor<f64>) -> tensor<f64> {
    return %arg0 : tensor<f64>
  }
  func.func private @"*_broadcast_scalar"(%arg0: tensor<f64>, %arg1: tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<f64>) {
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<f64>
    return %0, %arg0, %arg1 : tensor<f64>, tensor<f64>, tensor<f64>
  }
  func.func private @"+_broadcast_scalar"(%arg0: tensor<f64>, %arg1: tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<f64>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f64>
    return %0, %arg0, %arg1 : tensor<f64>, tensor<f64>, tensor<f64>
  }
  func.func private @identity_broadcast_scalar_1(%arg0: tensor<f64>) -> tensor<f64> {
    return %arg0 : tensor<f64>
  }
  func.func private @"Const{typeof(loop_rng_grad)}_autodiff"(%arg0: tensor<1xf64>, %arg1: tensor<2xui64>) -> (tensor<f64>, tensor<1xf64>, tensor<2xui64>) {
    %0 = stablehlo.transpose %arg0, dims = [0] : (tensor<1xf64>) -> tensor<1xf64>
    %1 = stablehlo.transpose %arg1, dims = [0] : (tensor<2xui64>) -> tensor<2xui64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<1xf64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %2 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_2 = stablehlo.constant dense<5> : tensor<i64>
    %c_3 = stablehlo.constant dense<1> : tensor<i64>
    %c_4 = stablehlo.constant dense<0> : tensor<i64>
    %3 = stablehlo.subtract %c_2, %c : tensor<i64>
    %4 = stablehlo.divide %3, %c_3 : tensor<i64>
    %c_5 = stablehlo.constant dense<1> : tensor<i64>
    %5 = stablehlo.add %4, %c_5 : tensor<i64>
    %6:7 = stablehlo.while(%iterArg = %c_4, %iterArg_9 = %5, %iterArg_10 = %2, %iterArg_11 = %c, %iterArg_12 = %1, %iterArg_13 = %c_3, %iterArg_14 = %0) : tensor<i64>, tensor<i64>, tensor<1xf64>, tensor<i64>, tensor<2xui64>, tensor<i64>, tensor<1xf64> attributes {enzyme.disable_mincut, enzymexla.checkpoint_period = 2 : i64, enzymexla.enable_checkpointing = true}
    cond {
      %11 = stablehlo.compare LT, %iterArg, %iterArg_9 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %11 : tensor<i1>
    } do {
      %11 = stablehlo.multiply %iterArg, %iterArg_13 : tensor<i64>
      %12 = stablehlo.add %iterArg_11, %11 : tensor<i64>
      %c_15 = stablehlo.constant dense<1> : tensor<i64>
      %13 = stablehlo.add %iterArg, %c_15 : tensor<i64>
      %output_state, %output = stablehlo.rng_bit_generator %iterArg_12, algorithm = DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<64xui64>)
      %c_16 = stablehlo.constant dense<12> : tensor<64xui64>
      %14 = stablehlo.shift_right_logical %output, %c_16 : tensor<64xui64>
      %c_17 = stablehlo.constant dense<4607182418800017408> : tensor<64xui64>
      %15 = stablehlo.or %14, %c_17 : tensor<64xui64>
      %16 = stablehlo.bitcast_convert %15 : (tensor<64xui64>) -> tensor<64xf64>
      %cst_18 = stablehlo.constant dense<1.000000e+00> : tensor<64xf64>
      %17 = stablehlo.subtract %16, %cst_18 : tensor<64xf64>
      %cst_19 = stablehlo.constant dense<-0.99999999999999988> : tensor<64xf64>
      %cst_20 = stablehlo.constant dense<0.99999999999999988> : tensor<64xf64>
      %18 = stablehlo.subtract %cst_20, %cst_19 : tensor<64xf64>
      %19 = stablehlo.multiply %17, %18 : tensor<64xf64>
      %20 = stablehlo.add %19, %cst_19 : tensor<64xf64>
      %21 = stablehlo.clamp %cst_19, %20, %cst_20 : tensor<64xf64>
      %22 = chlo.erf_inv %21 : tensor<64xf64> -> tensor<64xf64>
      %cst_21 = stablehlo.constant dense<1.4142135623730951> : tensor<64xf64>
      %23 = stablehlo.multiply %22, %cst_21 : tensor<64xf64>
      %24 = stablehlo.slice %output_state [0:2] : (tensor<2xui64>) -> tensor<2xui64>
      %25 = stablehlo.transpose %24, dims = [0] : (tensor<2xui64>) -> tensor<2xui64>
      %26 = stablehlo.reshape %25 : (tensor<2xui64>) -> tensor<2xui64>
      %27 = stablehlo.transpose %26, dims = [0] : (tensor<2xui64>) -> tensor<2xui64>
      %cst_22 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %28 = enzyme.batch @identity_broadcast_scalar(%23) {batch_shape = array<i64: 64>} : (tensor<64xf64>) -> tensor<64xf64>
      %cst_23 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %cst_24 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %29 = stablehlo.reduce(%28 init: %cst_22) applies stablehlo.add across dimensions = [0] : (tensor<64xf64>, tensor<f64>) -> tensor<f64>
      %cst_25 = stablehlo.constant dense<0.000000e+00> : tensor<1xf64>
      %30 = stablehlo.broadcast_in_dim %iterArg_10, dims = [0] : (tensor<1xf64>) -> tensor<1xf64>
      %31 = stablehlo.broadcast_in_dim %30, dims = [0] : (tensor<1xf64>) -> tensor<1xf64>
      %cst_26 = stablehlo.constant dense<0.000000e+00> : tensor<1xf64>
      %32 = stablehlo.broadcast_in_dim %iterArg_14, dims = [0] : (tensor<1xf64>) -> tensor<1xf64>
      %33 = stablehlo.broadcast_in_dim %32, dims = [0] : (tensor<1xf64>) -> tensor<1xf64>
      %34 = stablehlo.broadcast_in_dim %29, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %35:3 = enzyme.batch @"*_broadcast_scalar"(%33, %34) {batch_shape = array<i64: 1>} : (tensor<1xf64>, tensor<1xf64>) -> (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>)
      %36 = stablehlo.broadcast_in_dim %35#0, dims = [0] : (tensor<1xf64>) -> tensor<1xf64>
      %37:3 = enzyme.batch @"+_broadcast_scalar"(%31, %36) {batch_shape = array<i64: 1>} : (tensor<1xf64>, tensor<1xf64>) -> (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>)
      stablehlo.return %13, %iterArg_9, %37#0, %iterArg_11, %27, %iterArg_13, %iterArg_14 : tensor<i64>, tensor<i64>, tensor<1xf64>, tensor<i64>, tensor<2xui64>, tensor<i64>, tensor<1xf64>
    }
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %7 = enzyme.batch @identity_broadcast_scalar_1(%6#2) {batch_shape = array<i64: 1>} : (tensor<1xf64>) -> tensor<1xf64>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %8 = stablehlo.reduce(%7 init: %cst_6) applies stablehlo.add across dimensions = [0] : (tensor<1xf64>, tensor<f64>) -> tensor<f64>
    %9 = stablehlo.transpose %6#6, dims = [0] : (tensor<1xf64>) -> tensor<1xf64>
    %10 = stablehlo.transpose %6#4, dims = [0] : (tensor<2xui64>) -> tensor<2xui64>
    return %8, %9, %10 : tensor<f64>, tensor<1xf64>, tensor<2xui64>
  }
  func.func @main(%arg0: tensor<1xf64> {tf.aliasing_output = 1 : i32}, %arg1: tensor<2xui64> {tf.aliasing_output = 2 : i32}) -> (tensor<1xf64>, tensor<1xf64>, tensor<2xui64>) {
    %0 = stablehlo.transpose %arg0, dims = [0] : (tensor<1xf64>) -> tensor<1xf64>
    %1 = stablehlo.transpose %arg1, dims = [0] : (tensor<2xui64>) -> tensor<2xui64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<1xf64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %2 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %3 = stablehlo.transpose %0, dims = [0] : (tensor<1xf64>) -> tensor<1xf64>
    %4 = stablehlo.transpose %1, dims = [0] : (tensor<2xui64>) -> tensor<2xui64>
    %5 = stablehlo.transpose %2, dims = [0] : (tensor<1xf64>) -> tensor<1xf64>
    %6:3 = enzyme.autodiff @"Const{typeof(loop_rng_grad)}_autodiff"(%3, %4, %cst_2, %5) {activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>], ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>]} : (tensor<1xf64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>) -> (tensor<1xf64>, tensor<2xui64>, tensor<1xf64>)
    %7 = stablehlo.transpose %6#0, dims = [0] : (tensor<1xf64>) -> tensor<1xf64>
    %8 = stablehlo.transpose %6#1, dims = [0] : (tensor<2xui64>) -> tensor<2xui64>
    %9 = stablehlo.transpose %6#2, dims = [0] : (tensor<1xf64>) -> tensor<1xf64>
    %10 = stablehlo.transpose %9, dims = [0] : (tensor<1xf64>) -> tensor<1xf64>
    %11 = stablehlo.transpose %7, dims = [0] : (tensor<1xf64>) -> tensor<1xf64>
    %12 = stablehlo.transpose %8, dims = [0] : (tensor<2xui64>) -> tensor<2xui64>
    return %10, %11, %12 : tensor<1xf64>, tensor<1xf64>, tensor<2xui64>
  }
}

// CHECK: func.func private @"diffeConst{typeof(loop_rng_grad)}_autodiff"(%arg0: tensor<1xf64>, %arg1: tensor<2xui64>, %arg2: tensor<f64>, %arg3: tensor<1xf64>) -> (tensor<1xf64>, tensor<2xui64>, tensor<1xf64>) {
// CHECK:    %{{.+}} = stablehlo.constant dense<12> : tensor<64xui64>
