// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-linalg{backend=cpu blas_int_width=64},enzyme-hlo-opt)" %s | FileCheck %s

func.func @main(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xi32>, tensor<64xi32>, tensor<i32>) {
  %0:4 = enzymexla.linalg.lu %arg0 : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xi32>, tensor<64xi32>, tensor<i32>)
  return %0#0, %0#1, %0#2, %0#3 : tensor<64x64xf32>, tensor<64xi32>, tensor<64xi32>, tensor<i32>
}

// CHECK: enzymexla.lapack.getrf
