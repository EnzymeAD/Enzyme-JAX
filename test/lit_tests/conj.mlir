// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xcomplex<f32>> {
    %0 = stablehlo.negate %arg1 : tensor<2xf32>
    %1 = stablehlo.complex %arg0, %0 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xcomplex<f32>>
    %2 = chlo.conj %1 : tensor<2xcomplex<f32>> -> tensor<2xcomplex<f32>>
    return %2 : tensor<2xcomplex<f32>>
  }
}

// CHECK:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xcomplex<f32>> {
// CHECK-NEXT:    %0 = stablehlo.complex %arg0, %arg1 : tensor<2xcomplex<f32>>
// CHECK-NEXT:    return %0 : tensor<2xcomplex<f32>>
// CHECK-NEXT:  }
