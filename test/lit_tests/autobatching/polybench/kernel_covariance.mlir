// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<32x64xf32>) -> (tensor<32x32xf32>, tensor<32x64xf32>) {
    %cst = stablehlo.constant dense<0.015873015873015872> : tensor<32x32x1x1xf64>
    %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<32x1xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<32x32xf32>
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %c_4 = stablehlo.constant dense<32> : tensor<i64>
    %c_5 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.reshape %arg0 : (tensor<32x64xf32>) -> tensor<32x64x1xf32>
    %1 = stablehlo.reduce(%0 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<32x64x1xf32>, tensor<f32>) -> tensor<32x1xf32>
    %2 = stablehlo.multiply %1, %cst_0 : tensor<32x1xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [1, 0] : (tensor<32x64xf32>) -> tensor<64x32x1x1xf32>
    %4 = stablehlo.broadcast_in_dim %2, dims = [1, 2] : (tensor<32x1xf32>) -> tensor<64x32x1x1xf32>
    %5 = stablehlo.subtract %3, %4 : tensor<64x32x1x1xf32>
    %6 = stablehlo.transpose %5, dims = [1, 0, 2, 3] : (tensor<64x32x1x1xf32>) -> tensor<32x64x1x1xf32>
    %7 = stablehlo.reshape %6 : (tensor<32x64x1x1xf32>) -> tensor<32x64xf32>
    %8 = stablehlo.reshape %5 : (tensor<64x32x1x1xf32>) -> tensor<64x32x1xf32>
    %9 = stablehlo.dot_general %8, %8, batching_dims = [2] x [2], contracting_dims = [0] x [0] : (tensor<64x32x1xf32>, tensor<64x32x1xf32>) -> tensor<1x32x32xf32>
    %10 = stablehlo.convert %9 : (tensor<1x32x32xf32>) -> tensor<1x32x32xf64>
    %11 = stablehlo.reshape %10 : (tensor<1x32x32xf64>) -> tensor<32x32x1x1xf64>
    %12 = stablehlo.multiply %11, %cst : tensor<32x32x1x1xf64>
    %13 = stablehlo.convert %12 : (tensor<32x32x1x1xf64>) -> tensor<32x32x1x1xf32>
    %14:2 = stablehlo.while(%iterArg = %c_3, %iterArg_6 = %cst_2) : tensor<i64>, tensor<32x32xf32>
    cond {
      %15 = stablehlo.compare  LT, %iterArg, %c_4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %15 : tensor<i1>
    } do {
      %15 = stablehlo.add %c_5, %iterArg {enzymexla.bounds = [[1, 32]]} : tensor<i64>
      %16 = stablehlo.convert %15 {enzymexla.bounds = [[1, 32]]} : (tensor<i64>) -> tensor<i32>
      %17 = stablehlo.subtract %16, %c {enzymexla.bounds = [[0, 31]]} : tensor<i32>
      %18:2 = stablehlo.while(%iterArg_7 = %c_3, %iterArg_8 = %iterArg_6) : tensor<i64>, tensor<32x32xf32> attributes {enzyme.disable_mincut}
      cond {
        %19 = stablehlo.compare  LT, %iterArg_7, %c_4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %19 : tensor<i1>
      } do {
        %19 = stablehlo.transpose %iterArg_8, dims = [1, 0] : (tensor<32x32xf32>) -> tensor<32x32xf32>
        %20 = stablehlo.add %c_5, %iterArg_7 {enzymexla.bounds = [[1, 32]]} : tensor<i64>
        %21 = stablehlo.convert %20 {enzymexla.bounds = [[1, 32]]} : (tensor<i64>) -> tensor<i32>
        %22 = stablehlo.subtract %21, %c {enzymexla.bounds = [[0, 31]]} : tensor<i32>
        %23 = stablehlo.dynamic_slice %13, %iterArg, %iterArg_7, %c_3, %c_3, sizes = [1, 1, 1, 1] : (tensor<32x32x1x1xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x1x1x1xf32>
        %24 = stablehlo.reshape %23 : (tensor<1x1x1x1xf32>) -> tensor<1x1xf32>
        %25 = stablehlo.dynamic_update_slice %19, %24, %17, %22 : (tensor<32x32xf32>, tensor<1x1xf32>, tensor<i32>, tensor<i32>) -> tensor<32x32xf32>
        %26 = stablehlo.dynamic_slice %25, %22, %17, sizes = [1, 1] : (tensor<32x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
        %27 = stablehlo.dynamic_update_slice %iterArg_8, %26, %22, %17 : (tensor<32x32xf32>, tensor<1x1xf32>, tensor<i32>, tensor<i32>) -> tensor<32x32xf32>
        stablehlo.return %20, %27 : tensor<i64>, tensor<32x32xf32>
      }
      stablehlo.return %15, %18#1 : tensor<i64>, tensor<32x32xf32>
    }
    return %14#1, %7 : tensor<32x32xf32>, tensor<32x64xf32>
  }
}

// CHECK: %19 = stablehlo.add %c_5, %iterArg_7 {enzymexla.bounds = {{.*}}} : tensor<i64>
// CHECK-NEXT: %20 = stablehlo.convert %19 {enzymexla.bounds = {{.*}}} : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT: %21 = stablehlo.subtract %20, %c {enzymexla.bounds = {{.*}}} : tensor<i32>
// CHECK-NEXT: %22 = stablehlo.dynamic_slice %13, %iterArg, %iterArg_7, %c_3, %c_3, sizes = [1, 1, 1, 1] : (tensor<32x32x1x1xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x1x1x1xf32>
// CHECK-NEXT: %23 = stablehlo.reshape %22 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<1x1x1x1xf32>) -> tensor<1x1xf32>
// CHECK-NEXT: %24 = stablehlo.dynamic_update_slice %iterArg_8, %23, %21, %17 : (tensor<32x32xf32>, tensor<1x1xf32>, tensor<i32>, tensor<i32>) -> tensor<32x32xf32>
// CHECK-NEXT: %25 = stablehlo.dynamic_slice %24, %17, %21, sizes = [1, 1] {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<32x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
// CHECK-NEXT: %26 = stablehlo.dynamic_update_slice %iterArg_8, %25, %21, %17 : (tensor<32x32xf32>, tensor<1x1xf32>, tensor<i32>, tensor<i32>) -> tensor<32x32xf32>
// CHECK-NEXT: stablehlo.return %19, %26 : tensor<i64>, tensor<32x32xf32>
// CHECK-NEXT: }
