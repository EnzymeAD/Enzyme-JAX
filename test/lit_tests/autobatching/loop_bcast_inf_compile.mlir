// RUN: enzymexlamlir-opt --pass-pipeline="any(inline,enzyme-hlo-generate-td{patterns=broadcast_in_dim_simplify<16>(1024);iota_simplify<16>(1024);while_simplify<1>(1);while_deadresult;while_op_induction_replacement;greedy_while_loop_batch_fission;add_const_prop;mul_const_prop;div_const_prop;sub_const_prop},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

module {
  func.func private @"*_broadcast_scalar"(%arg0: tensor<f32>, %arg1: tensor<i64>) -> (tensor<f32>, tensor<f32>, tensor<i64>) {
    %0 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<f32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<f32>
    return %1, %arg0, %arg1 : tensor<f32>, tensor<f32>, tensor<i64>
  }
  func.func private @identity_broadcast_scalar(%arg0: tensor<f32>) -> tensor<f32> {
    return %arg0 : tensor<f32>
  }
  func.func private @"/_broadcast_scalar"(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>) {
    %0 = stablehlo.divide %arg0, %arg1 : tensor<f32>
    return %0, %arg0, %arg1 : tensor<f32>, tensor<f32>, tensor<f32>
  }
  func.func @nnorm(%arg0: tensor<10xf32>) -> tensor<10xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c_0 = stablehlo.constant dense<10> : tensor<i64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.transpose %arg0, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    // CHECK: stablehlo.while
    %1:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %0) : tensor<i64>, tensor<10xf32> attributes {enzyme.disable_mincut}
    cond {
      %3 = stablehlo.subtract %c_0, %c_1 : tensor<i64>
      %4 = stablehlo.divide %3, %c_1 : tensor<i64>
      %5 = stablehlo.add %4, %c_1 : tensor<i64>
      %6 = stablehlo.compare  LT, %iterArg, %5 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %6 : tensor<i1>
    } do {
      %3 = stablehlo.multiply %iterArg, %c_1 : tensor<i64>
      %4 = stablehlo.add %c_1, %3 : tensor<i64>
      %5 = stablehlo.add %iterArg, %c_1 : tensor<i64>
      %7 = stablehlo.broadcast_in_dim %iterArg_2, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    //   %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
      %8 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i64>) -> tensor<10xi64>
      %9:3 = enzyme.batch @"*_broadcast_scalar"(%7, %8) {batch_shape = array<i64: 10>} : (tensor<10xf32>, tensor<10xi64>) -> (tensor<10xf32>, tensor<10xf32>, tensor<10xi64>)
      %10 = enzyme.batch @identity_broadcast_scalar(%iterArg_2) {batch_shape = array<i64: 10>} : (tensor<10xf32>) -> tensor<10xf32>
      %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<10xf32>, tensor<f32>) -> tensor<f32>
      %12 = stablehlo.broadcast_in_dim %9#0, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
      %13 = stablehlo.broadcast_in_dim %12, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
      %14 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<f32>) -> tensor<10xf32>
      %15:3 = enzyme.batch @"/_broadcast_scalar"(%13, %14) {batch_shape = array<i64: 10>} : (tensor<10xf32>, tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>, tensor<10xf32>)
      stablehlo.return %5, %15#0 : tensor<i64>, tensor<10xf32>
    }
    %2 = stablehlo.transpose %1#1, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    return %2 : tensor<10xf32>
  }
}
