// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @transpose() -> tensor<2xui64> {
  %c_173 = stablehlo.constant dense<32> : tensor<2xui64>
  %c_174 = stablehlo.constant dense<[0, 1]> : tensor<2xui64>
  %5 = stablehlo.shift_right_logical %c_174, %c_173 : tensor<2xui64>
  return %5 : tensor<2xui64>
}

// CHECK:  func.func @transpose() -> tensor<2xui64> {
// CHECK-NEXT:    %0 = stablehlo.constant dense<0> : tensor<2xui64>
// CHECK-NEXT:    return %0 : tensor<2xui64>
// CHECK-NEXT:  }
