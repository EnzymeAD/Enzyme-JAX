// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active,enzyme_active,enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --enzyme-hlo-opt --drop-unsupported-attributes --verify-each=0 | FileCheck %s --check-prefix=REVERSE

func.func @main(%x : tensor<4x4xcomplex<f32>>) -> (tensor<4x4xcomplex<f32>>, tensor<4xcomplex<f32>>, tensor<4x4xcomplex<f32>>) {
  %U, %S, %Vt, %info = enzymexla.linalg.svd %x : (tensor<4x4xcomplex<f32>>) -> (tensor<4x4xcomplex<f32>>, tensor<4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<i32>)
  func.return %U, %S, %Vt : tensor<4x4xcomplex<f32>>, tensor<4xcomplex<f32>>, tensor<4x4xcomplex<f32>>
}

// REVERSE: func.func @main(%arg0: tensor<4x4xcomplex<f32>>, %arg1: tensor<4x4xcomplex<f32>>, %arg2: tensor<4xcomplex<f32>>, %arg3: tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>> {
// REVERSE-NEXT:   %cst = stablehlo.constant
// REVERSE-NEXT:   %c = stablehlo.constant
// REVERSE-NEXT:   %cst_0 = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<4xcomplex<f32>>
// REVERSE-NEXT:   %cst_1 = stablehlo.constant dense<(5.000000e-01,0.000000e+00)> : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %cst_2 = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %cst_3 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
// REVERSE-NEXT:   %cst_4 = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
// REVERSE-NEXT:   %cst_5 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %U, %S, %Vt, %info = enzymexla.linalg.svd %arg0 : (tensor<4x4xcomplex<f32>>) -> (tensor<4x4xcomplex<f32>>, tensor<4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<i32>)
// REVERSE-NEXT:   %0 = chlo.conj %arg1 : tensor<4x4xcomplex<f32>> -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %2 = enzymexla.blas.gemm %cst_4, %1, %U, %cst_3, %cst_5 : (tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %3 = chlo.conj %U : tensor<4x4xcomplex<f32>> -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %4 = stablehlo.transpose %3, dims = [1, 0] : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %5 = enzymexla.blas.gemm %cst_4, %4, %arg1, %cst_3, %cst_5 : (tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %6 = stablehlo.subtract %5, %2 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %7 = chlo.conj %Vt : tensor<4x4xcomplex<f32>> -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %8 = stablehlo.transpose %7, dims = [1, 0] : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %9 = chlo.conj %arg3 : tensor<4x4xcomplex<f32>> -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %10 = stablehlo.transpose %9, dims = [1, 0] : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %11 = enzymexla.blas.gemm %cst_4, %arg3, %8, %cst_3, %cst_5 : (tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %12 = enzymexla.blas.gemm %cst_4, %Vt, %10, %cst_3, %cst_5 : (tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %13 = stablehlo.subtract %12, %11 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %14 = stablehlo.broadcast_in_dim %S, dims = [1] : (tensor<4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %15 = stablehlo.broadcast_in_dim %S, dims = [0] : (tensor<4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %16 = stablehlo.subtract %14, %15 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %17 = stablehlo.add %14, %15 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %18 = stablehlo.divide %cst_2, %16 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %19 = stablehlo.divide %cst_2, %17 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %20 = stablehlo.add %18, %19 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %21 = stablehlo.select %c, %cst_5, %20 : tensor<4x4xi1>, tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %22 = stablehlo.subtract %18, %19 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %23 = stablehlo.select %c, %cst_5, %22 : tensor<4x4xi1>, tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %24 = stablehlo.multiply %23, %13 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %25 = stablehlo.multiply %21, %6 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %26 = stablehlo.add %25, %24 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %27 = stablehlo.multiply %cst_1, %U : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %28 = enzymexla.blas.gemm %cst_4, %27, %26, %cst_3, %cst_5 : (tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %29 = enzymexla.blas.gemm %cst_4, %28, %Vt, %cst_3, %cst_5 : (tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %30 = stablehlo.dot_general %U, %arg2, batching_dims = [1] x [0], contracting_dims = [] x [] : (tensor<4x4xcomplex<f32>>, tensor<4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %31 = enzymexla.blas.gemm %cst_4, %30, %Vt, %cst_3, %cst_5 : (tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %32 = stablehlo.add %29, %31 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %33 = stablehlo.divide %cst_0, %S : tensor<4xcomplex<f32>>
// REVERSE-NEXT:   %34 = stablehlo.dot_general %arg1, %33, batching_dims = [1] x [0], contracting_dims = [] x [] : (tensor<4x4xcomplex<f32>>, tensor<4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %35 = enzymexla.blas.gemm %cst_4, %U, %4, %cst_3, %cst_5 : (tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %36 = stablehlo.subtract %cst, %35 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %37 = enzymexla.blas.gemm %cst_4, %36, %34, %cst_3, %cst_5 : (tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %38 = enzymexla.blas.gemm %cst_4, %37, %Vt, %cst_3, %cst_5 : (tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %39 = stablehlo.add %32, %38 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %40 = stablehlo.dot_general %U, %33, batching_dims = [1] x [0], contracting_dims = [] x [] : (tensor<4x4xcomplex<f32>>, tensor<4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %41 = enzymexla.blas.gemm %cst_4, %8, %Vt, %cst_3, %cst_5 : (tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %42 = stablehlo.subtract %cst, %41 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %43 = enzymexla.blas.gemm %cst_4, %arg3, %42, %cst_3, %cst_5 : (tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %44 = enzymexla.blas.gemm %cst_4, %40, %43, %cst_3, %cst_5 : (tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %45 = stablehlo.add %39, %44 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   return %45 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT: }
