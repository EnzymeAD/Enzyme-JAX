// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=qr_real outfn= retTys=enzyme_active,enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --enzyme-hlo-opt --drop-unsupported-attributes --verify-each=0 | FileCheck %s --check-prefix=REVERSE-REAL
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=qr_complex outfn= retTys=enzyme_active,enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --enzyme-hlo-opt --drop-unsupported-attributes --verify-each=0 | FileCheck %s --check-prefix=REVERSE-COMPLEX

func.func @qr_real(%x : tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  %Q, %R, %info = enzymexla.linalg.qr %x : (tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<i32>)
  func.return %Q, %R : tensor<4x4xf32>, tensor<4x4xf32>
}

// REVERSE-REAL: func.func @qr_real(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
// REVERSE-REAL-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// REVERSE-REAL-NEXT:   %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// REVERSE-REAL-NEXT:   %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf32>
// REVERSE-REAL-NEXT:   %Q, %R, %info = enzymexla.linalg.qr %arg0 : (tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<i32>)
// REVERSE-REAL-NEXT:   %0 = enzymexla.blas.trsm %cst_0, %R, %arg2 {side = #enzymexla.side<right>, transa = #enzymexla.transpose<adjoint>, uplo = #enzymexla.uplo<U>} : (tensor<f32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// REVERSE-REAL-NEXT:   %1 = enzymexla.blas.gemm %cst_0, %arg1, %Q, %cst, %cst_1 {transa = #enzymexla.transpose<adjoint>} : (tensor<f32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<f32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// REVERSE-REAL-NEXT:   %2 = stablehlo.subtract %0, %1 : tensor<4x4xf32>
// REVERSE-REAL-NEXT:   %3 = enzymexla.blas.symm %2, %Q, %arg1, %cst_0, %cst_0 {side = #enzymexla.side<right>, uplo = #enzymexla.uplo<L>} : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<f32>, tensor<f32>) -> tensor<4x4xf32>
// REVERSE-REAL-NEXT:   %4 = enzymexla.blas.trsm %cst_0, %R, %3 {side = #enzymexla.side<right>, transa = #enzymexla.transpose<adjoint>, uplo = #enzymexla.uplo<U>} : (tensor<f32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// REVERSE-REAL-NEXT:   return %4 : tensor<4x4xf32>
// REVERSE-REAL-NEXT: }

func.func @qr_complex(%x : tensor<4x4xcomplex<f32>>) -> (tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>) {
  %Q, %R, %info = enzymexla.linalg.qr %x : (tensor<4x4xcomplex<f32>>) -> (tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<i32>)
  func.return %Q, %R : tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>
}

// REVERSE-COMPLEX: func.func @qr_complex(%arg0: tensor<4x4xcomplex<f32>>, %arg1: tensor<4x4xcomplex<f32>>, %arg2: tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>> {
// REVERSE-COMPLEX-NEXT:   %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
// REVERSE-COMPLEX-NEXT:   %cst_0 = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
// REVERSE-COMPLEX-NEXT:   %cst_1 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<4x4xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:   %Q, %R, %info = enzymexla.linalg.qr %arg0 : (tensor<4x4xcomplex<f32>>) -> (tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<i32>)
// REVERSE-COMPLEX-NEXT:   %0 = enzymexla.blas.trsm %cst_0, %R, %arg2 {side = #enzymexla.side<right>, transa = #enzymexla.transpose<adjoint>, uplo = #enzymexla.uplo<U>} : (tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:   %1 = enzymexla.blas.gemm %cst_0, %arg1, %Q, %cst, %cst_1 {transa = #enzymexla.transpose<adjoint>} : (tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:   %2 = stablehlo.subtract %0, %1 : tensor<4x4xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:   %3 = enzymexla.blas.symm %2, %Q, %arg1, %cst_0, %cst_0 {side = #enzymexla.side<right>, uplo = #enzymexla.uplo<L>} : (tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<complex<f32>>, tensor<complex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:   %4 = enzymexla.blas.trsm %cst_0, %R, %3 {side = #enzymexla.side<right>, transa = #enzymexla.transpose<adjoint>, uplo = #enzymexla.uplo<U>} : (tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:   return %4 : tensor<4x4xcomplex<f32>>
// REVERSE-COMPLEX-NEXT: }
