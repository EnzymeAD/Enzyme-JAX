// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cpu blas_int_width=64},enzyme-hlo-opt)" %s | FileCheck %s
// To print the lowered IR, run this command without the trailing `| FileCheck %s`.

module {
  func.func @lapack_chain(%a: tensor<4x4xf64>) -> (tensor<4x4xf64>, tensor<i64>) {
    %0:3 = enzymexla.lapack.geqrf %a : (tensor<4x4xf64>) -> (tensor<4x4xf64>, tensor<4xf64>, tensor<i64>)
    %1 = enzymexla.lapack.orgqr %0#0, %0#1 : (tensor<4x4xf64>, tensor<4xf64>) -> tensor<4x4xf64>
    %2:2 = enzymexla.lapack.potrf %1 <{uplo = #enzymexla.uplo<U>}> : (tensor<4x4xf64>) -> (tensor<4x4xf64>, tensor<i64>)
    return %2#0, %2#1 : tensor<4x4xf64>, tensor<i64>
  }
}

// CHECK-COUNT-3: enzymexla.jit_call
