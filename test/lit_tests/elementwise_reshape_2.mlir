// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=elementwise_reshape_like;reshape_elementwise(0)},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<32x64xf32>, %arg3: tensor<64x32xf32>) -> tensor<64x64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<32> : tensor<i64>
    %c_1 = stablehlo.constant dense<64> : tensor<i64>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<64x64xf32>
    %c_3 = stablehlo.constant dense<1> : tensor<i32>
    %c_4 = stablehlo.constant dense<0> : tensor<i64>
    %c_5 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.transpose %arg2, dims = [1, 0] : (tensor<32x64xf32>) -> tensor<64x32xf32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<64x32xf32>
    %2 = stablehlo.multiply %1, %0 : tensor<64x32xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<64x32xf32>) -> tensor<64x32x64x1xf32>
    %4 = stablehlo.broadcast_in_dim %arg3, dims = [2, 1] : (tensor<64x32xf32>) -> tensor<64x32x64x1xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x32x64x1xf32>
    %6 = stablehlo.multiply %cst, %arg1 : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<64x1xf32>
    %8:2 = stablehlo.while(%iterArg = %c_4, %iterArg_6 = %cst_2) : tensor<i64>, tensor<64x64xf32>
    cond {
      %9 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %9 : tensor<i1>
    } do {
      %9 = stablehlo.add %c_5, %iterArg : tensor<i64>
      %10 = stablehlo.convert %9 : (tensor<i64>) -> tensor<i32>
      %11 = stablehlo.subtract %10, %c_3 : tensor<i32>
      %12:2 = stablehlo.while(%iterArg_7 = %c_4, %iterArg_8 = %7) : tensor<i64>, tensor<64x1xf32> attributes {enzyme.disable_mincut}
      cond {
        %14 = stablehlo.compare  LT, %iterArg_7, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %14 : tensor<i1>
      } do {
        %14 = stablehlo.add %c_5, %iterArg_7 : tensor<i64>
        %15 = stablehlo.reshape %iterArg_8 : (tensor<64x1xf32>) -> tensor<64x1x1xf32>
        %16 = stablehlo.dynamic_slice %5, %iterArg, %iterArg_7, %c_4, %c_4, sizes = [1, 1, 64, 1] : (tensor<64x32x64x1xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x1x64x1xf32>
        %17 = stablehlo.reshape %16 : (tensor<1x1x64x1xf32>) -> tensor<64x1x1xf32>
        %18 = stablehlo.add %15, %17 : tensor<64x1x1xf32>
        %19 = stablehlo.reshape %18 : (tensor<64x1x1xf32>) -> tensor<64x1xf32>
        stablehlo.return %14, %19 : tensor<i64>, tensor<64x1xf32>
      }
      %13 = stablehlo.dynamic_update_slice %iterArg_6, %12#1, %c, %11 : (tensor<64x64xf32>, tensor<64x1xf32>, tensor<i32>, tensor<i32>) -> tensor<64x64xf32>
      stablehlo.return %9, %13 : tensor<i64>, tensor<64x64xf32>
    }
    return %8#1 : tensor<64x64xf32>
  }
}

// CHECK:   %14 = stablehlo.compare  LT, %iterArg_7, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:   stablehlo.return %14 : tensor<i1>
// CHECK-NEXT: } do {
// CHECK-NEXT:   %14 = stablehlo.add %c_5, %iterArg_7 : tensor<i64>
// CHECK-NEXT:   %15 = stablehlo.dynamic_slice %5, %iterArg, %iterArg_7, %c_4, %c_4, sizes = [1, 1, 64, 1] : (tensor<64x32x64x1xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x1x64x1xf32>
// CHECK-NEXT:   %16 = stablehlo.reshape %15 : (tensor<1x1x64x1xf32>) -> tensor<64x1xf32>
// CHECK-NEXT:   %17 = stablehlo.add %iterArg_8, %16 : tensor<64x1xf32>
// CHECK-NEXT:   stablehlo.return %14, %17 : tensor<i64>, tensor<64x1xf32>
// CHECK-NEXT: }
