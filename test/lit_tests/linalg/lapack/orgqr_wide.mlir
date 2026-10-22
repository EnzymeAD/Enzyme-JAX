// RUN: not enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cpu blas_int_width=64},enzyme-hlo-opt)" %s 2>&1 | FileCheck %s --check-prefix=CPU
// RUN: not enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cuda},enzyme-hlo-opt)" %s 2>&1 | FileCheck %s --check-prefix=CUDA
// RUN: not enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=tpu},enzyme-hlo-opt)" %s 2>&1 | FileCheck %s --check-prefix=TPU

// LAPACK's ?orgqr/?ungqr need m >= n, so a wide input is rejected on CPU
// CPU: error: 'enzymexla.lapack.orgqr' op `enzymexla.lapack.orgqr` requires at least as many rows as columns
// CUDA: error: 'enzymexla.lapack.orgqr' op `enzymexla.lapack.orgqr` requires at least as many rows as columns
// TPU: error: 'enzymexla.lapack.orgqr' op `enzymexla.lapack.orgqr` requires at least as many rows as columns

func.func @main(%arg0: tensor<32x64xf32>, %arg1: tensor<32xf32>) -> tensor<32x64xf32> {
    %0 = enzymexla.lapack.orgqr %arg0, %arg1 : (tensor<32x64xf32>, tensor<32xf32>) -> tensor<32x64xf32>
    return %0 : tensor<32x64xf32>
}
