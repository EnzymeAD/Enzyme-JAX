// RUN: enzymexlamlir-opt %s | FileCheck %s

func.func @trmm(%arg0: tensor<64x64xf32>, %arg1: tensor<64x32xf32>) -> tensor<64x32xf32> {
    %alpha = stablehlo.constant dense<2.0> : tensor<f32>
    %0 = enzymexla.blas.trmm %arg0, %arg1, %alpha {side = #enzymexla.side<left>, uplo = #enzymexla.uplo<U>, transpose = #enzymexla.transpose<none>, unit_diagonal} : (tensor<64x64xf32>, tensor<64x32xf32>, tensor<f32>) -> tensor<64x32xf32>
    return %0 : tensor<64x32xf32>
}

// CHECK: enzymexla.blas.trmm
