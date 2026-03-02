// RUN: enzymexlamlir-opt --lower-enzymexla-blas={backend=cpu,blas_int_width=64} --enzyme-hlo-opt %s | FileCheck %s --check-prefix=CHECK

func.func @main(%alpha: tensor<f32>, %a: tensor<64x64xf32>, %b: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = enzymexla.blas.trsm %alpha, %a, %b {side = #enzymexla.side<left>, uplo = #enzymexla.uplo<U>, transa = #enzymexla.transpose<none>, diag = #enzymexla.diag<nonunit>} : (tensor<f32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// CHECK:  func.func @main(%arg0: tensor<f32>, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
// CHECK-NEXT:    %1 = stablehlo.multiply %arg2, %0 : tensor<64x64xf32>
// CHECK-NEXT:    %2 = "stablehlo.triangular_solve"(%arg1, %1) <{left_side = true, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:    return %2 : tensor<64x64xf32>
// CHECK-NEXT:  }
