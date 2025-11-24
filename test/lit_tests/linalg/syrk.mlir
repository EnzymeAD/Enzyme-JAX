// RUN: enzymexlamlir-opt %s | FileCheck %s

func.func @main(%arg0: tensor<64x32xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %alpha = stablehlo.constant dense<2.0> : tensor<f32>
    %beta = stablehlo.constant dense<3.0> : tensor<f32>
    %0 = enzymexla.blas.syrk %arg0, %arg1, %alpha, %beta {transpose = #enzymexla.transpose<none>} : (tensor<64x32xf32>, tensor<64x64xf32>, tensor<f32>, tensor<f32>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
}

// CHECK: enzymexla.blas.syrk
