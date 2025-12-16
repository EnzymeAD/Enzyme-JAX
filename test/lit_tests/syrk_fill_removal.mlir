// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-opt{passses=syrk_fill_remove})" | FileCheck %s

func.func @main1(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %cst = stablehlo.constant dense<3.444500e+00> : tensor<32x32xf32>
    %cst_0 = stablehlo.constant dense<2.031500e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<32x32xf32>
    %cst_4 = stablehlo.constant dense<-4.775000e+00> : tensor<32x32xf32>

    // First SYRK: computes C = alpha * A^T * A + beta * C
    %0 = enzymexla.blas.syrk %arg0, %cst_3, %cst_2, %cst_1 {fill, transpose = #enzymexla.transpose<transpose>, uplo = #enzymexla.uplo<F>} : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<f32>, tensor<f32>) -> tensor<32x32xf32>

    // Intermediate elementwise ops feeding into second SYRK's second operand
    %1 = stablehlo.multiply %cst_4, %0 : tensor<32x32xf32>
    %2 = stablehlo.add %1, %cst : tensor<32x32xf32>

    // Second SYRK: uses %0 as input matrix A, and %2 (derived from %0) as input matrix C
    %3 = enzymexla.blas.syrk %0, %2, %cst_0, %cst_2 {fill, uplo = #enzymexla.uplo<F>} : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<f32>, tensor<f32>) -> tensor<32x32xf32>

    return %3 : tensor<32x32xf32>
}

func.func @main2_fail(%arg0: tensor<32x32xf32>) -> (tensor<32x32xf32>, tensor<32x32xf32>) {
    %cst = stablehlo.constant dense<3.444500e+00> : tensor<32x32xf32>
    %cst_0 = stablehlo.constant dense<2.031500e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<32x32xf32>
    %cst_4 = stablehlo.constant dense<-4.775000e+00> : tensor<32x32xf32>

    // First SYRK: computes C = alpha * A^T * A + beta * C
    %0 = enzymexla.blas.syrk %arg0, %cst_3, %cst_2, %cst_1 {fill, transpose = #enzymexla.transpose<transpose>, uplo = #enzymexla.uplo<F>} : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<f32>, tensor<f32>) -> tensor<32x32xf32>

    // Intermediate elementwise ops feeding into second SYRK's second operand
    %1 = stablehlo.multiply %cst_4, %0 : tensor<32x32xf32>
    %2 = stablehlo.add %1, %cst : tensor<32x32xf32>

    // Second SYRK: uses %0 as input matrix A, and %2 (derived from %0) as input matrix C
    %3 = enzymexla.blas.syrk %0, %2, %cst_0, %cst_2 {fill, uplo = #enzymexla.uplo<F>} : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<f32>, tensor<f32>) -> tensor<32x32xf32>

    return %3, %2 : tensor<32x32xf32>, tensor<32x32xf32>
}

func.func @main3(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %cst = stablehlo.constant dense<3.444500e+00> : tensor<32x32xf32>
    %cst_0 = stablehlo.constant dense<2.031500e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<32x32xf32>
    %cst_4 = stablehlo.constant dense<-4.775000e+00> : tensor<32x32xf32>

    // First SYRK: computes C = alpha * A^T * A + beta * C
    %0 = enzymexla.blas.syrk %arg0, %cst_3, %cst_2, %cst_1 {fill, transpose = #enzymexla.transpose<transpose>, uplo = #enzymexla.uplo<F>} : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<f32>, tensor<f32>) -> tensor<32x32xf32>

    // Intermediate elementwise ops feeding into second SYRK's second operand
    %1 = stablehlo.multiply %cst_4, %0 : tensor<32x32xf32>
    %2 = stablehlo.add %1, %cst : tensor<32x32xf32>

    // Second SYRK: uses %0 as input matrix A, and %2 (derived from %0) as input matrix C
    %3 = enzymexla.blas.syrk %0, %2, %cst_0, %cst_2 {fill, uplo = #enzymexla.uplo<L>} : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<f32>, tensor<f32>) -> tensor<32x32xf32>

    return %3 : tensor<32x32xf32>
}
