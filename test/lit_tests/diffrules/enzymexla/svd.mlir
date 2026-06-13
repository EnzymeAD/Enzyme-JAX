// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active,enzyme_active,enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --enzyme-hlo-opt --drop-unsupported-attributes --verify-each=0 | FileCheck %s --check-prefix=REVERSE

func.func @main(%x : tensor<4x4xcomplex<f32>>) -> (tensor<4x4xcomplex<f32>>, tensor<4xcomplex<f32>>, tensor<4x4xcomplex<f32>>) {
  %U, %S, %Vt, %info = enzymexla.linalg.svd %x : (tensor<4x4xcomplex<f32>>) -> (tensor<4x4xcomplex<f32>>, tensor<4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<i32>)
  func.return %U, %S, %Vt : tensor<4x4xcomplex<f32>>, tensor<4xcomplex<f32>>, tensor<4x4xcomplex<f32>>
}

// REVERSE: func.func @main(%arg0: tensor<4x4xcomplex<f32>>, %arg1: tensor<4x4xcomplex<f32>>, %arg2: tensor<4xcomplex<f32>>, %arg3: tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>> {
// REVERSE-NEXT:   %cst = stablehlo.constant dense<[[(1.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)], [(0.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)], [(0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)], [(0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00)]]> : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %c = stablehlo.constant dense<[[true, false, false, false], [false, true, false, false], [false, false, true, false], [false, false, false, true]]> : tensor<4x4xi1>
// REVERSE-NEXT:   %cst_0 = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<4xcomplex<f32>>
// REVERSE-NEXT:   %cst_1 = stablehlo.constant dense<(5.000000e-01,0.000000e+00)> : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %cst_2 = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %cst_3 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
// REVERSE-NEXT:   %cst_4 = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
// REVERSE-NEXT:   %cst_5 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %U, %S, %Vt, %info = enzymexla.linalg.svd %arg0 : (tensor<4x4xcomplex<f32>>) -> (tensor<4x4xcomplex<f32>>, tensor<4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<i32>)
// REVERSE-NEXT:   %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %1 = enzymexla.blas.gemm %cst_4, %0, %U, %cst_3, %cst_5 : (tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %2 = stablehlo.transpose %U, dims = [1, 0] : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %3 = enzymexla.blas.gemm %cst_4, %2, %arg1, %cst_3, %cst_5 : (tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %4 = stablehlo.subtract %3, %1 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %5 = stablehlo.transpose %Vt, dims = [1, 0] : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %6 = stablehlo.transpose %arg3, dims = [1, 0] : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %7 = enzymexla.blas.gemm %cst_4, %arg3, %5, %cst_3, %cst_5 : (tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %8 = enzymexla.blas.gemm %cst_4, %Vt, %6, %cst_3, %cst_5 : (tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %9 = stablehlo.subtract %8, %7 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %10 = stablehlo.broadcast_in_dim %S, dims = [1] : (tensor<4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %11 = stablehlo.broadcast_in_dim %S, dims = [0] : (tensor<4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %12 = stablehlo.subtract %10, %11 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %13 = stablehlo.add %10, %11 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %14 = stablehlo.divide %cst_2, %12 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %15 = stablehlo.divide %cst_2, %13 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %16 = stablehlo.add %14, %15 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %17 = stablehlo.select %c, %cst_5, %16 : tensor<4x4xi1>, tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %18 = stablehlo.subtract %14, %15 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %19 = stablehlo.select %c, %cst_5, %18 : tensor<4x4xi1>, tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %20 = stablehlo.multiply %19, %9 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %21 = stablehlo.multiply %17, %4 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %22 = stablehlo.add %21, %20 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %23 = stablehlo.multiply %cst_1, %U : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %24 = enzymexla.blas.gemm %cst_4, %23, %22, %cst_3, %cst_5 : (tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %25 = enzymexla.blas.gemm %cst_4, %24, %Vt, %cst_3, %cst_5 : (tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %26 = stablehlo.dot_general %U, %arg2, batching_dims = [1] x [0], contracting_dims = [] x [] : (tensor<4x4xcomplex<f32>>, tensor<4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %27 = enzymexla.blas.gemm %cst_4, %26, %Vt, %cst_3, %cst_5 : (tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %28 = stablehlo.add %25, %27 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %29 = stablehlo.divide %cst_0, %S : tensor<4xcomplex<f32>>
// REVERSE-NEXT:   %30 = stablehlo.dot_general %arg1, %29, batching_dims = [1] x [0], contracting_dims = [] x [] : (tensor<4x4xcomplex<f32>>, tensor<4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %31 = enzymexla.blas.gemm %cst_4, %U, %2, %cst_3, %cst_5 : (tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %32 = stablehlo.subtract %cst, %31 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %33 = enzymexla.blas.gemm %cst_4, %32, %30, %cst_3, %cst_5 : (tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %34 = enzymexla.blas.gemm %cst_4, %33, %Vt, %cst_3, %cst_5 : (tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %35 = stablehlo.add %28, %34 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %36 = stablehlo.dot_general %U, %29, batching_dims = [1] x [0], contracting_dims = [] x [] : (tensor<4x4xcomplex<f32>>, tensor<4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %37 = enzymexla.blas.gemm %cst_4, %5, %Vt, %cst_3, %cst_5 : (tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %38 = stablehlo.subtract %cst, %37 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %39 = enzymexla.blas.gemm %cst_4, %arg3, %38, %cst_3, %cst_5 : (tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %40 = enzymexla.blas.gemm %cst_4, %36, %39, %cst_3, %cst_5 : (tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   %41 = stablehlo.add %35, %40 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT:   return %41 : tensor<4x4xcomplex<f32>>
// REVERSE-NEXT: }
