// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @conj_real(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = chlo.conj %arg0 : tensor<2x2xf32> -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
}

// CHECK: func.func @conj_real(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:   return %arg0 : tensor<2x2xf32>
// CHECK-NEXT: }
