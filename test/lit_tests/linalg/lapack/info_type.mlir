// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cpu blas_int_width=32},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CPU

// With 32-bit LAPACK ints, `info` comes back as i32 and has to be converted to
// the type the op declares

func.func @geqrf(%arg0: tensor<64x64xf32>) -> tensor<i64> {
  %0:3 = enzymexla.lapack.geqrf %arg0 : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<i64>)
  return %0#2 : tensor<i64>
}

// CPU-LABEL: func.func @geqrf
// CPU:         %[[CALL:.+]]:3 = enzymexla.jit_call
// CPU:         %[[INFO:.+]] = stablehlo.convert %[[CALL]]#2 : (tensor<i32>) -> tensor<i64>
// CPU:         return %[[INFO]] : tensor<i64>

func.func @geqrt(%arg0: tensor<64x64xf32>) -> tensor<i64> {
  %0:3 = enzymexla.lapack.geqrt %arg0 : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<i64>)
  return %0#2 : tensor<i64>
}

// CPU-LABEL: func.func @geqrt
// CPU:         %[[CALL:.+]]:3 = enzymexla.jit_call
// CPU:         %[[INFO:.+]] = stablehlo.convert %[[CALL]]#2 : (tensor<i32>) -> tensor<i64>
// CPU:         return %[[INFO]] : tensor<i64>

func.func @potrf(%arg0: tensor<64x64xf32>) -> tensor<i64> {
  %0:2 = enzymexla.lapack.potrf %arg0 <{uplo = #enzymexla.uplo<U>}> : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<i64>)
  return %0#1 : tensor<i64>
}

// CPU-LABEL: func.func @potrf
// CPU:         %[[CALL:.+]]:2 = enzymexla.jit_call
// CPU:         %[[INFO:.+]] = stablehlo.convert %[[CALL]]#1 : (tensor<i32>) -> tensor<i64>
// CPU:         return %[[INFO]] : tensor<i64>
