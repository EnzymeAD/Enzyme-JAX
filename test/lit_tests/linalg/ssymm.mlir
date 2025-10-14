// RUN: enzymexlamlir-opt %s -split-input-file | FileCheck %s
module {
    func.func @main(%arg0: tensor<64x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %alpha = stablehlo.constant dense<2.0> : tensor<f64>
    %beta = stablehlo.constant dense<3.0> : tensor<f64>
    %0 = enzymexla.lapack.ssymm %arg0, %arg1, %arg2, %alpha, %beta {side = #enzymexla.side<left>, uplo = #enzymexla.uplo<U>} : (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<f64>, tensor<f64>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
    }
}

// CHECK: enzymexla.ssymm
