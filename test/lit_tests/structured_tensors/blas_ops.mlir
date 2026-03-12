// RUN: enzymexlamlir-opt %s | FileCheck %s

func.func @trmm(%arg0: tensor<64x64xf32>, %arg1: tensor<64x32xf32>) -> tensor<64x32xf32> {
    %alpha = stablehlo.constant dense<2.0> : tensor<f32>
    %0 = enzymexla.blas.trmm %arg0, %arg1, %alpha {side = #enzymexla.side<left>, uplo = #enzymexla.uplo<U>, transpose = #enzymexla.transpose<none>, diag = #enzymexla.diag<nonunit>} : (tensor<64x64xf32>, tensor<64x32xf32>, tensor<f32>) -> tensor<64x32xf32>
    return %0 : tensor<64x32xf32>
}

// CHECK: enzymexla.blas.trmm

func.func @trmv(%arg0: tensor<64x64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %0 = enzymexla.blas.trmv %arg0, %arg1 {uplo = #enzymexla.uplo<U>, transpose = #enzymexla.transpose<none>, diag = #enzymexla.diag<nonunit>} : (tensor<64x64xf32>, tensor<64xf32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
}

// CHECK: enzymexla.blas.trmv
