// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active,enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --enzyme-hlo-opt | FileCheck %s --check-prefix=REVERSE

func.func @main(%x : tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  %Q, %R, %info = enzymexla.linalg.qr %x : (tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<i32>)
  func.return %Q, %R : tensor<4x4xf32>, tensor<4x4xf32>
}

// REVERSE: func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
// REVERSE-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// REVERSE-NEXT:   %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// REVERSE-NEXT:   %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf32>
// REVERSE-NEXT:   %Q, %R, %info = enzymexla.linalg.qr %arg0 : (tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<i32>)
// REVERSE-NEXT:   %0 = enzymexla.blas.trsm %cst_0, %R, %arg2 {side = #enzymexla.side<right>, transa = #enzymexla.transpose<adjoint>, uplo = #enzymexla.uplo<U>} : (tensor<f32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// REVERSE-NEXT:   %1 = enzymexla.blas.gemm %cst_0, %arg1, %Q, %cst, %cst_1 {transa = #enzymexla.transpose<adjoint>} : (tensor<f32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<f32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// REVERSE-NEXT:   %2 = stablehlo.subtract %0, %1 : tensor<4x4xf32>
// REVERSE-NEXT:   %3 = enzymexla.blas.symm %2, %Q, %arg1, %cst_0, %cst_0 {side = #enzymexla.side<right>, uplo = #enzymexla.uplo<L>} : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<f32>, tensor<f32>) -> tensor<4x4xf32>
// REVERSE-NEXT:   %4 = enzymexla.blas.trsm %cst_0, %R, %3 {side = #enzymexla.side<right>, transa = #enzymexla.transpose<adjoint>, uplo = #enzymexla.uplo<U>} : (tensor<f32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// REVERSE-NEXT:   return %4 : tensor<4x4xf32>
// REVERSE-NEXT: }
