// RUN: enzymexlamlir-opt --lower-enzymexla-lapack={backend=cpu,blas_int_width=64} --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @batched_potrf(%arg0: tensor<3x8x8xf64>) -> (tensor<3x8x8xf64>, tensor<3xi32>) {
    %0:2 = enzymexla.lapack.potrf %arg0 <{uplo = #enzymexla.uplo<L>}> : (tensor<3x8x8xf64>) -> (tensor<3x8x8xf64>, tensor<3xi32>)
    return %0#0, %0#1 : tensor<3x8x8xf64>, tensor<3xi32>
  }

  func.func @shlo_cholesky(%arg0: tensor<16x16xf64>) -> tensor<16x16xf64> {
    %0 = stablehlo.cholesky %arg0, lower = true : tensor<16x16xf64>
    return %0 : tensor<16x16xf64>
  }

  func.func @shlo_cholesky_batched(%arg0: tensor<2x8x8xcomplex<f64>>) -> tensor<2x8x8xcomplex<f64>> {
    %0 = stablehlo.cholesky %arg0, lower = false : tensor<2x8x8xcomplex<f64>>
    return %0 : tensor<2x8x8xcomplex<f64>>
  }
}

// CHECK-DAG: llvm.func @enzymexla_lapack_dpotrf_
// CHECK-DAG: llvm.func @enzymexla_lapack_zpotrf_
// CHECK-LABEL: func.func @batched_potrf(
// CHECK-NOT: enzymexla.lapack.potrf
// CHECK: call @batched_enzymexla_wrapper_lapack_dpotrf_
// CHECK-LABEL: func.func @shlo_cholesky(
// CHECK-NOT: stablehlo.cholesky
// CHECK: enzymexla.jit_call @enzymexla_wrapper_lapack_dpotrf_
// CHECK: stablehlo.select
// CHECK-LABEL: func.func @shlo_cholesky_batched(
// CHECK-NOT: stablehlo.cholesky
// CHECK: call @batched_enzymexla_wrapper_lapack_zpotrf_
// CHECK: stablehlo.select
