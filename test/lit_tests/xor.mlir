// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=xor_simplify" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module {
  func.func @"wrapper!"(%arg0: tensor<8xi1>) -> tensor<8xi1> {
    %c = stablehlo.constant dense<true> : tensor<8xi1>
    %5 = stablehlo.xor %arg0, %c : tensor<8xi1>
    return %5 : tensor<8xi1>
  }
}

// CHECK:  func.func @"wrapper!"(%arg0: tensor<8xi1>) -> tensor<8xi1> {
// CHECK-NEXT:    %0 = stablehlo.not %arg0 : tensor<8xi1>
// CHECK-NEXT:    return %0 : tensor<8xi1>
// CHECK-NEXT:  }

