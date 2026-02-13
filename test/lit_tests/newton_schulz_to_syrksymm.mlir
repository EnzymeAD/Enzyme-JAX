// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=dot_general_to_syrk;dot_general_to_symm;transpose_syrk_to_syrk;fuse_mul_into_syrk;fuse_add_into_syrk},transform-interpreter,enzyme-hlo-remove-transform,enzyme-hlo-opt)" %s | FileCheck %s

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
// CHECK-DAG:     %[[cstC:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<5x4xf32>
// CHECK-DAG:     %[[cst2:.+]] = stablehlo.constant {enzymexla.finite = [#enzymexla<guaranteed GUARANTEED>], enzymexla.no_nan = [#enzymexla<guaranteed GUARANTEED>]} dense<2.031500e+00> : tensor<f32>
// CHECK-DAG:     %[[cstm0:.+]] = stablehlo.constant {enzymexla.finite = [#enzymexla<guaranteed GUARANTEED>], enzymexla.no_nan = [#enzymexla<guaranteed GUARANTEED>]} dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:     %[[cst1:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:     %[[cst0:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf32>
// CHECK-DAG:     %[[c0:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-DAG:     %[[c5:.+]] = stablehlo.constant dense<5> : tensor<i64>
// CHECK-DAG:     %[[c1:.+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-DAG:     %[[c47:.+]] = stablehlo.constant {enzymexla.symmetric_matrix = [#enzymexla<guaranteed GUARANTEED>]} dense<-4.775000e+00> : tensor<4x4xf32>
// CHECK-DAG:     %[[c34:.+]] = stablehlo.constant dense<3.444500e+00> : tensor<5x4xf32>
// CHECK-NEXT:     %0:2 = stablehlo.while(%iterArg = %[[c0]], %iterArg_8 = %arg0) : tensor<i64>, tensor<5x4xf32>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %1 = stablehlo.compare  LT, %iterArg, %[[c5]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %1 = stablehlo.add %iterArg, %[[c1]] {enzymexla.bounds = {{.*}}} : tensor<i64>
// CHECK-NEXT:       %2 = enzymexla.blas.syrk %iterArg_8, %[[cst0]], %[[cst1]], %[[cstm0]] {output_uplo = #enzymexla.uplo<F>, transpose = #enzymexla.transpose<transpose>, uplo = #enzymexla.uplo<F>} : (tensor<5x4xf32>, tensor<4x4xf32>, tensor<f32>, tensor<f32>) -> tensor<4x4xf32>
// CHECK-NEXT:       %3 = stablehlo.multiply %[[c47]], %2 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed GUARANTEED>]} : tensor<4x4xf32>
// CHECK-NEXT:       %4 = enzymexla.blas.syrk %2, %3, %[[cst2]], %[[cst1]] {enzymexla.symmetric_matrix = [#enzymexla<guaranteed GUARANTEED>], output_uplo = #enzymexla.uplo<F>, uplo = #enzymexla.uplo<F>} : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<f32>, tensor<f32>) -> tensor<4x4xf32>
// CHECK-NEXT:       %5 = stablehlo.multiply %[[c34]], %iterArg_8 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<5x4xf32>
// CHECK-NEXT:       %6 = enzymexla.blas.symm %iterArg_8, %4, %[[cstC]], %[[cst1]], %[[cstm0]] {side = #enzymexla.side<right>, uplo = #enzymexla.uplo<F>} : (tensor<5x4xf32>, tensor<4x4xf32>, tensor<5x4xf32>, tensor<f32>, tensor<f32>) -> tensor<5x4xf32>
// CHECK-NEXT:       %7 = stablehlo.add %5, %6 : tensor<5x4xf32>
// CHECK-NEXT:       stablehlo.return %1, %7 : tensor<i64>, tensor<5x4xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %0#1 : tensor<5x4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
