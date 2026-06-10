// RUN: enzymexlamlir-opt --lower-enzymexla-blas="backend=cpu" --enzyme-hlo-opt --drop-unsupported-attributes %s | FileCheck %s

func.func @fused_matmul_add(%alpha: tensor<f32>, %A: tensor<2x4xf32>, %B: tensor<4x3xf32>, %beta: tensor<f32>, %C: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = enzymexla.blas.gemm %alpha, %A, %B, %beta, %C : (tensor<f32>, tensor<2x4xf32>, tensor<4x3xf32>, tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
}

// CHECK: func.func @fused_matmul_add(%arg0: tensor<f32>, %arg1: tensor<2x4xf32>, %arg2: tensor<4x3xf32>, %arg3: tensor<f32>, %arg4: tensor<2x3xf32>) -> tensor<2x3xf32> {
// CHECK-NEXT:   %0 = stablehlo.dot_general %arg1, %arg2, contracting_dims = [1] x [0] : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:   %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x3xf32>
// CHECK-NEXT:   %2 = stablehlo.multiply %1, %0 : tensor<2x3xf32>
// CHECK-NEXT:   %3 = stablehlo.broadcast_in_dim %arg3, dims = [] : (tensor<f32>) -> tensor<2x3xf32>
// CHECK-NEXT:   %4 = stablehlo.multiply %3, %arg4 : tensor<2x3xf32>
// CHECK-NEXT:   %5 = stablehlo.add %2, %4 : tensor<2x3xf32>
// CHECK-NEXT:   return %5 : tensor<2x3xf32>
// CHECK-NEXT: }

func.func @matmul(%alpha: tensor<f32>, %A: tensor<2x4xf32>, %B: tensor<4x3xf32>) -> tensor<2x3xf32> {
    %beta = stablehlo.constant dense<3.0> : tensor<f32>
    %C = stablehlo.constant dense<0.0> : tensor<2x3xf32>
    %0 = enzymexla.blas.gemm %alpha, %A, %B, %beta, %C : (tensor<f32>, tensor<2x4xf32>, tensor<4x3xf32>, tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
}

// CHECK: func.func @matmul(%arg0: tensor<f32>, %arg1: tensor<2x4xf32>, %arg2: tensor<4x3xf32>) -> tensor<2x3xf32> {
// CHECK-NEXT:   %0 = stablehlo.dot_general %arg1, %arg2, contracting_dims = [1] x [0] : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:   %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x3xf32>
// CHECK-NEXT:   %2 = stablehlo.multiply %1, %0 : tensor<2x3xf32>
// CHECK-NEXT:   return %2 : tensor<2x3xf32>
// CHECK-NEXT: }

func.func @batch_matmul(%alpha: tensor<f32>, %A: tensor<8x10x2x4xf32>, %B: tensor<8x10x4x3xf32>) -> tensor<8x10x2x3xf32> {
    %beta = stablehlo.constant dense<3.0> : tensor<f32>
    %C = stablehlo.constant dense<0.0> : tensor<8x10x2x3xf32>
    %0 = enzymexla.blas.gemm %alpha, %A, %B, %beta, %C : (tensor<f32>, tensor<8x10x2x4xf32>, tensor<8x10x4x3xf32>, tensor<f32>, tensor<8x10x2x3xf32>) -> tensor<8x10x2x3xf32>
    return %0 : tensor<8x10x2x3xf32>
}

// CHECK: func.func @batch_matmul(%arg0: tensor<f32>, %arg1: tensor<8x10x2x4xf32>, %arg2: tensor<8x10x4x3xf32>) -> tensor<8x10x2x3xf32> {
// CHECK-NEXT:   %0 = stablehlo.dot_general %arg1, %arg2, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<8x10x2x4xf32>, tensor<8x10x4x3xf32>) -> tensor<8x10x2x3xf32>
// CHECK-NEXT:   %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<8x10x2x3xf32>
// CHECK-NEXT:   %2 = stablehlo.multiply %1, %0 : tensor<8x10x2x3xf32>
// CHECK-NEXT:   return %2 : tensor<8x10x2x3xf32>
// CHECK-NEXT: }

func.func @matmul_transpose_transpose(%A: tensor<4x2xf32>, %B: tensor<3x4xf32>) -> tensor<2x3xf32> {
    %alpha = stablehlo.constant dense<1.0> : tensor<f32>
    %beta = stablehlo.constant dense<0.0> : tensor<f32>
    %C = stablehlo.constant dense<0.0> : tensor<2x3xf32>
    %0 = enzymexla.blas.gemm %alpha, %A, %B, %beta, %C {transa = #enzymexla.transpose<transpose>, transb = #enzymexla.transpose<transpose>} : (tensor<f32>, tensor<4x2xf32>, tensor<3x4xf32>, tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
}

// CHECK: func.func @matmul_transpose_transpose(%arg0: tensor<4x2xf32>, %arg1: tensor<3x4xf32>) -> tensor<2x3xf32> {
// CHECK-NEXT:   %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [1] : (tensor<4x2xf32>, tensor<3x4xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:   return %0 : tensor<2x3xf32>
// CHECK-NEXT: }

func.func @matmul_transpose_none(%A: tensor<4x2xf32>, %B: tensor<4x3xf32>) -> tensor<2x3xf32> {
    %alpha = stablehlo.constant dense<1.0> : tensor<f32>
    %beta = stablehlo.constant dense<0.0> : tensor<f32>
    %C = stablehlo.constant dense<0.0> : tensor<2x3xf32>
    %0 = enzymexla.blas.gemm %alpha, %A, %B, %beta, %C {transa = #enzymexla.transpose<transpose>, transb = #enzymexla.transpose<none>} : (tensor<f32>, tensor<4x2xf32>, tensor<4x3xf32>, tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
}

// CHECK: func.func @matmul_transpose_none(%arg0: tensor<4x2xf32>, %arg1: tensor<4x3xf32>) -> tensor<2x3xf32> {
// CHECK-NEXT:   %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [0] : (tensor<4x2xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:   return %0 : tensor<2x3xf32>
// CHECK-NEXT: }

func.func @matmul_none_transpose(%A: tensor<2x4xf32>, %B: tensor<3x4xf32>) -> tensor<2x3xf32> {
    %alpha = stablehlo.constant dense<1.0> : tensor<f32>
    %beta = stablehlo.constant dense<0.0> : tensor<f32>
    %C = stablehlo.constant dense<0.0> : tensor<2x3xf32>
    %0 = enzymexla.blas.gemm %alpha, %A, %B, %beta, %C {transa = #enzymexla.transpose<none>, transb = #enzymexla.transpose<transpose>} : (tensor<f32>, tensor<2x4xf32>, tensor<3x4xf32>, tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
}

// CHECK: func.func @matmul_none_transpose(%arg0: tensor<2x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<2x3xf32> {
// CHECK-NEXT:   %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1] : (tensor<2x4xf32>, tensor<3x4xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:   return %0 : tensor<2x3xf32>
// CHECK-NEXT: }

func.func @matmul_adjoint_adjoint(%A: tensor<4x2xcomplex<f32>>, %B: tensor<3x4xcomplex<f32>>) -> tensor<2x3xcomplex<f32>> {
    %alpha = stablehlo.constant dense<(1.0, 0.0)> : tensor<complex<f32>>
    %beta = stablehlo.constant dense<(0.0, 0.0)> : tensor<complex<f32>>
    %C = stablehlo.constant dense<(0.0, 0.0)> : tensor<2x3xcomplex<f32>>
    %0 = enzymexla.blas.gemm %alpha, %A, %B, %beta, %C {transa = #enzymexla.transpose<adjoint>, transb = #enzymexla.transpose<adjoint>} : (tensor<complex<f32>>, tensor<4x2xcomplex<f32>>, tensor<3x4xcomplex<f32>>, tensor<complex<f32>>, tensor<2x3xcomplex<f32>>) -> tensor<2x3xcomplex<f32>>
    return %0 : tensor<2x3xcomplex<f32>>
}

// CHECK-NEXT: func.func @matmul_adjoint_adjoint(%arg0: tensor<4x2xcomplex<f32>>, %arg1: tensor<3x4xcomplex<f32>>) -> tensor<2x3xcomplex<f32>> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<2x3xcomplex<f32>>
// CHECK-NEXT:   %0 = chlo.conj %arg0 : tensor<4x2xcomplex<f32>> -> tensor<4x2xcomplex<f32>>
// CHECK-NEXT:   %1 = chlo.conj %arg1 : tensor<3x4xcomplex<f32>> -> tensor<3x4xcomplex<f32>>
// CHECK-NEXT:   %2 = stablehlo.dot_general %0, %1, contracting_dims = [0] x [1] : (tensor<4x2xcomplex<f32>>, tensor<3x4xcomplex<f32>>) -> tensor<2x3xcomplex<f32>>
// CHECK-NEXT:   %3 = stablehlo.multiply %cst, %2 : tensor<2x3xcomplex<f32>>
// CHECK-NEXT:   return %3 : tensor<2x3xcomplex<f32>>
// CHECK-NEXT: }
