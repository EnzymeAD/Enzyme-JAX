// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

func.func @real_conj(%arg0: tensor<4x5x3xcomplex<f32>>) -> tensor<4x5x3xf32> {
    %0 = chlo.conj %arg0 : tensor<4x5x3xcomplex<f32>> -> tensor<4x5x3xcomplex<f32>>
    %1 = stablehlo.real %0 : (tensor<4x5x3xcomplex<f32>>) -> tensor<4x5x3xf32>
    return %1 : tensor<4x5x3xf32>
}

// CHECK: func.func @real_conj(%arg0: tensor<4x5x3xcomplex<f32>>) -> tensor<4x5x3xf32> {
// CHECK-NEXT:     %0 = stablehlo.real %arg0 : (tensor<4x5x3xcomplex<f32>>) -> tensor<4x5x3xf32>
// CHECK-NEXT:     return %0 : tensor<4x5x3xf32>
// CHECK-NEXT: }
