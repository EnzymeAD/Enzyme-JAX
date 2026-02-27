// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cpu blas_int_width=64},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CPU

func.func @main(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<i64>) {
  %0:2 = enzymexla.lapack.potrf %arg0 {uplo = #enzymexla.uplo<U>} : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<i64>)
  return %0#0, %0#1 : tensor<64x64xf32>, tensor<i64>
}

