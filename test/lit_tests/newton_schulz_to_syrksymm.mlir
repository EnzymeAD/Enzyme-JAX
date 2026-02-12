// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=dot_general_to_syrk;transpose_syrk_to_syrk;fuse_mul_into_syrk;fuse_add_into_syrk},transform-interpreter,enzyme-hlo-remove-transform,enzyme-hlo-opt)" %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<5x4xf32>) -> tensor<5x4xf32> {
    %cst = stablehlo.constant dense<2.031500e+00> : tensor<4x4xf32>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c_0 = stablehlo.constant dense<5> : tensor<i64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %cst_2 = stablehlo.constant dense<-4.775000e+00> : tensor<4x4xf32>
    %cst_3 = stablehlo.constant dense<3.444500e+00> : tensor<5x4xf32>
    %0:2 = stablehlo.while(%iterArg = %c, %iterArg_4 = %arg0) : tensor<i64>, tensor<5x4xf32>
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c_1 : tensor<i64>
      %2 = stablehlo.dot_general %iterArg_4, %iterArg_4, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<5x4xf32>, tensor<5x4xf32>) -> tensor<4x4xf32>
      %3 = stablehlo.multiply %cst_2, %2 : tensor<4x4xf32>
      %4 = stablehlo.dot_general %2, %2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      %5 = stablehlo.multiply %cst, %4 : tensor<4x4xf32>
      %6 = stablehlo.add %3, %5 : tensor<4x4xf32>
      %7 = stablehlo.multiply %cst_3, %iterArg_4 : tensor<5x4xf32>
      %8 = stablehlo.dot_general %iterArg_4, %6, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<5x4xf32>, tensor<4x4xf32>) -> tensor<5x4xf32>
      %9 = stablehlo.add %7, %8 : tensor<5x4xf32>
      stablehlo.return %1, %9 : tensor<i64>, tensor<5x4xf32>
    }
    return %0#1 : tensor<5x4xf32>
  }
}

// CHECK: module {
// CHECK-NEXT:   func.func @main(%arg0: tensor<5x4xf32>) -> tensor<5x4xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant {enzymexla.finite = [#enzymexla<guaranteed GUARANTEED>], enzymexla.no_nan = [#enzymexla<guaranteed GUARANTEED>]} dense<2.031500e+00> : tensor<f32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf32>
// CHECK-NEXT:     %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-NEXT:     %cst_2 = stablehlo.constant {enzymexla.finite = [#enzymexla<guaranteed GUARANTEED>], enzymexla.no_nan = [#enzymexla<guaranteed GUARANTEED>]} dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %c_3 = stablehlo.constant dense<5> : tensor<i64>
// CHECK-NEXT:     %c_4 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:     %cst_5 = stablehlo.constant dense<-4.775000e+00> : tensor<4x4xf32>
// CHECK-NEXT:     %cst_6 = stablehlo.constant dense<3.444500e+00> : tensor<5x4xf32>
// CHECK-NEXT:     %0:2 = stablehlo.while(%iterArg = %c, %iterArg_7 = %arg0) : tensor<i64>, tensor<5x4xf32>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %1 = stablehlo.compare  LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %1 = stablehlo.add %iterArg, %c_4 {enzymexla.bounds = {{.*}}} : tensor<i64>
// CHECK-NEXT:       %2 = enzymexla.blas.syrk %iterArg_7, %cst_0, %cst_1, %cst_2 {output_uplo = #enzymexla.uplo<F>, transpose = #enzymexla.transpose<transpose>, uplo = #enzymexla.uplo<F>} : (tensor<5x4xf32>, tensor<4x4xf32>, tensor<f32>, tensor<f32>) -> tensor<4x4xf32>
// CHECK-NEXT:       %3 = stablehlo.multiply %cst_5, %2 : tensor<4x4xf32>
// CHECK-NEXT:       %4 = enzymexla.blas.syrk %2, %3, %cst, %cst_1 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed GUARANTEED>], output_uplo = #enzymexla.uplo<F>, uplo = #enzymexla.uplo<F>} : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<f32>, tensor<f32>) -> tensor<4x4xf32>
// CHECK-NEXT:       %5 = stablehlo.multiply %cst_6, %iterArg_7 : tensor<5x4xf32>
// CHECK-NEXT:       %6 = enzymexla.blas.symm %iterArg_7, %4, %5, %cst_1, %cst_1 {side = #enzymexla.side<right>, uplo = #enzymexla.uplo<F>} : (tensor<5x4xf32>, tensor<4x4xf32>, tensor<5x4xf32>, tensor<f32>, tensor<f32>) -> tensor<5x4xf32>
// CHECK-NEXT:       stablehlo.return %1, %6 : tensor<i64>, tensor<5x4xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %0#1 : tensor<5x4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }