// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cuda})" %s | FileCheck %s --check-prefix=CUDA

// gesvd returns V^H directly, so it is used as-is
func.func @gesvd(%arg0: tensor<64x32xcomplex<f32>>) -> (tensor<64x32xcomplex<f32>>, tensor<32xf32>, tensor<32x32xcomplex<f32>>, tensor<i64>) {
  %0:4 = enzymexla.lapack.gesvd %arg0 : (tensor<64x32xcomplex<f32>>) -> (tensor<64x32xcomplex<f32>>, tensor<32xf32>, tensor<32x32xcomplex<f32>>, tensor<i64>)
  return %0#0, %0#1, %0#2, %0#3 : tensor<64x32xcomplex<f32>>, tensor<32xf32>, tensor<32x32xcomplex<f32>>, tensor<i64>
}

// CUDA-LABEL: func.func @gesvd
// CUDA:         %[[SVD:.+]]:5 = stablehlo.custom_call @cusolver_gesvd_ffi(%arg0)
// CUDA-SAME:      -> (tensor<64x32xcomplex<f32>>, tensor<32xf32>, tensor<64x32xcomplex<f32>>, tensor<32x32xcomplex<f32>>, tensor<i32>)
// CUDA-NOT:     stablehlo.transpose
// CUDA-NOT:     chlo.conj
// CUDA:         return %[[SVD]]#2, %[[SVD]]#1, %[[SVD]]#3,

// gesvdj returns V, so Vt = adjoint(V) = conj(transpose(V))
func.func @gesvj(%arg0: tensor<32x64xcomplex<f32>>) -> (tensor<32x32xcomplex<f32>>, tensor<32xf32>, tensor<32x64xcomplex<f32>>, tensor<i64>) {
  %0:4 = enzymexla.lapack.gesvj %arg0 : (tensor<32x64xcomplex<f32>>) -> (tensor<32x32xcomplex<f32>>, tensor<32xf32>, tensor<32x64xcomplex<f32>>, tensor<i64>)
  return %0#0, %0#1, %0#2, %0#3 : tensor<32x32xcomplex<f32>>, tensor<32xf32>, tensor<32x64xcomplex<f32>>, tensor<i64>
}

// CUDA-LABEL: func.func @gesvj
// CUDA:         %[[SVD:.+]]:5 = stablehlo.custom_call @cusolver_gesvdj_ffi(%arg0)
// CUDA-SAME:      -> (tensor<32x64xcomplex<f32>>, tensor<32xf32>, tensor<32x32xcomplex<f32>>, tensor<64x32xcomplex<f32>>, tensor<i32>)
// CUDA:         %[[VT:.+]] = stablehlo.transpose %[[SVD]]#3, dims = [1, 0]
// CUDA:         %[[CVT:.+]] = chlo.conj %[[VT]]
// CUDA:         return %[[SVD]]#2, %[[SVD]]#1, %[[CVT]],
