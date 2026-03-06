// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

func.func @syrk_user_symm(%arg0: tensor<5x4xf32> {enzymexla.memory_effects = []}, %arg1: tensor<7x4xf32> {enzymexla.memory_effects = []}) -> tensor<7x4xf32> attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<7x4xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = blas.syrk %arg0, %cst_1, %cst_2, %cst_3 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed GUARANTEED>], output_uplo = #blas.uplo<any>, transpose = #blas.transpose<transpose>, uplo = #blas.uplo<any>} : (tensor<5x4xf32>, tensor<4x4xf32>, tensor<f32>, tensor<f32>) -> tensor<4x4xf32>
    %1 = blas.symm %0, %arg1, %cst, %cst_0, %cst_3 {side = #blas.side<right>, uplo = #blas.uplo<any>} : (tensor<4x4xf32>, tensor<7x4xf32>, tensor<7x4xf32>, tensor<f32>, tensor<f32>) -> tensor<7x4xf32>
    return %1 : tensor<7x4xf32>
}

// CHECK: %0 = blas.syrk %arg0, %cst_1, %cst_2, %cst_3 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed GUARANTEED>], output_uplo = #blas.uplo<upper>, transpose = #blas.transpose<transpose>, uplo = #blas.uplo<any>} : (tensor<5x4xf32>, tensor<4x4xf32>, tensor<f32>, tensor<f32>) -> tensor<4x4xf32>
// CHECK-NEXT: %1 = blas.symm %0, %arg1, %cst, %cst_0, %cst_3 {side = #blas.side<right>, uplo = #blas.uplo<upper>} : (tensor<4x4xf32>, tensor<7x4xf32>, tensor<7x4xf32>, tensor<f32>, tensor<f32>) -> tensor<7x4xf32>
// CHECK-NEXT: return %1 : tensor<7x4xf32>
