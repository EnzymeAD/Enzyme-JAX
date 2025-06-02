// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

func.func @conj(%arg0: tensor<4x5x3xcomplex<f32>>) -> tensor<4x5x3xcomplex<f32>> {
    %0 = chlo.conj %arg0 : tensor<4x5x3xcomplex<f32>> -> tensor<4x5x3xcomplex<f32>>
    %1 = chlo.conj %0 : tensor<4x5x3xcomplex<f32>> -> tensor<4x5x3xcomplex<f32>>
    return %1 : tensor<4x5x3xcomplex<f32>>
}

// CHECK: func.func @conj(%arg0: tensor<4x5x3xcomplex<f32>>) -> tensor<4x5x3xcomplex<f32>> {
// CHECK-NEXT:     return %arg0 : tensor<4x5x3xcomplex<f32>>
// CHECK-NEXT: }

func.func @negate(%arg0: tensor<5x3x4xf32>) -> tensor<5x3x4xf32> {
    %0 = stablehlo.negate %arg0 : tensor<5x3x4xf32>
    %1 = stablehlo.negate %0 : tensor<5x3x4xf32>
    return %1 : tensor<5x3x4xf32>
}

// CHECK: func.func @negate(%arg0: tensor<5x3x4xf32>) -> tensor<5x3x4xf32> {
// CHECK-NEXT:     return %arg0 : tensor<5x3x4xf32>
// CHECK-NEXT: }