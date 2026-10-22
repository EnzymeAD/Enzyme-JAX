// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=tpu blas_int_width=64},enzyme-hlo-opt,drop-unsupported-attributes)" %s | FileCheck %s --check-prefix=TPU

// @SVD returns V, so Vt = adjoint(V) = conj(transpose(V))
func.func @main(%arg0: tensor<32x32xcomplex<f32>>) -> (tensor<32x32xcomplex<f32>>, tensor<32xf32>, tensor<32x32xcomplex<f32>>, tensor<i64>) {
  %0:4 = enzymexla.lapack.gesvj %arg0 : (tensor<32x32xcomplex<f32>>) -> (tensor<32x32xcomplex<f32>>, tensor<32xf32>, tensor<32x32xcomplex<f32>>, tensor<i64>)
  return %0#0, %0#1, %0#2, %0#3 : tensor<32x32xcomplex<f32>>, tensor<32xf32>, tensor<32x32xcomplex<f32>>, tensor<i64>
}

// TPU-LABEL: func.func @main
// TPU:         %[[SVD:.+]]:3 = stablehlo.custom_call @SVD(%arg0)
// enzyme-hlo-opt moves the conjugate ahead of the transpose
// TPU:         %[[CV:.+]] = chlo.conj %[[SVD]]#2
// TPU:         %[[CVT:.+]] = stablehlo.transpose %[[CV]], dims = [1, 0]
// TPU:         return %[[SVD]]#0, %[[SVD]]#1, %[[CVT]],
