// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=concat_to_onedim_dus"  --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module {
  func.func @main(%5301 : tensor<4x8x80xf64>, %6569 : tensor<1x8x80xf64>) -> tensor<4x8x80xf64> {
    %10860 = stablehlo.slice %5301 [0:3, 0:8, 0:80] : (tensor<4x8x80xf64>) -> tensor<3x8x80xf64>
    %11832 = stablehlo.concatenate %10860, %6569, dim = 0 : (tensor<3x8x80xf64>, tensor<1x8x80xf64>) -> tensor<4x8x80xf64>
    return %11832 : tensor<4x8x80xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<4x8x80xf64>, %arg1: tensor<1x8x80xf64>) -> tensor<4x8x80xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %c, %c, %c : (tensor<4x8x80xf64>, tensor<1x8x80xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<4x8x80xf64>
// CHECK-NEXT:    return %0 : tensor<4x8x80xf64>
// CHECK-NEXT:  }
