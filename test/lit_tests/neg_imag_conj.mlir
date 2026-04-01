// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

func.func @main(%arg0: tensor<2xcomplex<f32>>) -> tensor<2xf32> {
    %0 = chlo.conj %arg0 : tensor<2xcomplex<f32>> -> tensor<2xcomplex<f32>>
    %1 = stablehlo.imag %0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
    %2 = stablehlo.negate %1 : tensor<2xf32>
    return %2 : tensor<2xf32>
}

// CHECK: func.func @main(%arg0: tensor<2xcomplex<f32>>) -> tensor<2xf32> {
// CHECK-NEXT:     %0 = stablehlo.imag %arg0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// CHECK-NEXT:     return %0 : tensor<2xf32>
// CHECK-NEXT: }
