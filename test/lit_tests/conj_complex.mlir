// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

func.func @conj(%arg0: tensor<5x3x4xf32>) -> tensor<4x5x3xcomplex<f32>> {
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<4x5x3xf32>
    %0 = stablehlo.transpose %arg0, dims = [2, 0, 1] : (tensor<5x3x4xf32>) -> tensor<4x5x3xf32>
    %1 = stablehlo.complex %0, %cst_0 : tensor<4x5x3xcomplex<f32>>
    %2 = chlo.conj %1 : tensor<4x5x3xcomplex<f32>> -> tensor<4x5x3xcomplex<f32>>
    return %2 : tensor<4x5x3xcomplex<f32>>
}

// CHECK: func.func @conj(%arg0: tensor<5x3x4xf32>) -> tensor<4x5x3xcomplex<f32>> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<-1.000000e+00> : tensor<4x5x3xf32>
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [2, 0, 1] : (tensor<5x3x4xf32>) -> tensor<4x5x3xf32>
// CHECK-NEXT:     %1 = stablehlo.complex %0, %cst : tensor<4x5x3xcomplex<f32>>
// CHECK-NEXT:     return %1 : tensor<4x5x3xcomplex<f32>>
// CHECK-NEXT: }
