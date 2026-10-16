// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cpu blas_int_width=64})" %s | FileCheck %s --check-prefix=CPU

// The job char is baked into the LAPACK wrapper, so an op that only computes
// singular values must not share a wrapper with one that also computes U and Vt

func.func @main(%arg0: tensor<6x4xf64>) -> (tensor<4xf64>, tensor<6x4xf64>, tensor<4xf64>, tensor<4x4xf64>) {
  %a:4 = enzymexla.lapack.gesvd %arg0 <{compute_uv = false}> : (tensor<6x4xf64>) -> (tensor<6x4xf64>, tensor<4xf64>, tensor<4x4xf64>, tensor<i64>)
  %b:4 = enzymexla.lapack.gesvd %arg0 : (tensor<6x4xf64>) -> (tensor<6x4xf64>, tensor<4xf64>, tensor<4x4xf64>, tensor<i64>)
  return %a#1, %b#0, %b#1, %b#2 : tensor<4xf64>, tensor<6x4xf64>, tensor<4xf64>, tensor<4x4xf64>
}

// CPU-DAG: llvm.func private @enzymexla_wrapper_lapack_dgesvd__nouv(
// CPU-DAG: llvm.func private @enzymexla_wrapper_lapack_dgesvd_(
// CPU-DAG: enzymexla.jit_call @enzymexla_wrapper_lapack_dgesvd__nouv
// CPU-DAG: enzymexla.jit_call @enzymexla_wrapper_lapack_dgesvd_ (
