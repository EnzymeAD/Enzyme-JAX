// RUN: enzymexlamlir-opt %s | FileCheck %s
// RUN: enzymexlamlir-opt --lower-enzymexla-blas="backend=cpu" %s | FileCheck %s --check-prefix=LOWERCPU

func.func @main1(%arg0: tensor<64x32xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %alpha = stablehlo.constant dense<2.0> : tensor<f32>
    %beta = stablehlo.constant dense<3.0> : tensor<f32>
    %0 = enzymexla.blas.syrk %arg0, %arg1, %alpha, %beta {transpose = #enzymexla.transpose<none>, uplo = #enzymexla.uplo<U>} : (tensor<64x32xf32>, tensor<64x64xf32>, tensor<f32>, tensor<f32>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
}

// CHECK: enzymexla.blas.syrk

func.func @main2(%arg0: tensor<64x32xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %alpha = stablehlo.constant dense<2.0> : tensor<f32>
    %beta = stablehlo.constant dense<3.0> : tensor<f32>
    %0 = enzymexla.blas.syrk %arg0, %arg1, %alpha, %beta {transpose = #enzymexla.transpose<none>, uplo = #enzymexla.uplo<L>} : (tensor<64x32xf32>, tensor<64x64xf32>, tensor<f32>, tensor<f32>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
}

func.func @main3(%arg0: tensor<5x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %cst = stablehlo.constant dense<5.000000e+00> : tensor<4x4xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = enzymexla.blas.syrk %arg0, %cst_0, %cst_1, %cst_2 {fill, transpose = #enzymexla.transpose<transpose>, uplo = #enzymexla.uplo<U>} : (tensor<5x4xf32>, tensor<4x4xf32>, tensor<f32>, tensor<f32>) -> tensor<4x4xf32>
    %1 = stablehlo.multiply %cst, %arg1 : tensor<4x4xf32>
    %2 = stablehlo.add %0, %1 : tensor<4x4xf32>
    return %2 : tensor<4x4xf32>
}
