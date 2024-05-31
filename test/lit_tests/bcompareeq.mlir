// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt)" %s | FileCheck %s 

module {

  func.func @main() -> tensor<1000000000xi1> {
    %c_118 = stablehlo.constant dense<0> : tensor<1000000000xi32>
    %c_119 = stablehlo.constant dense<256> : tensor<1000000000xi32>
    %2 = stablehlo.compare  EQ, %c_119, %c_118 : (tensor<1000000000xi32>, tensor<1000000000xi32>) -> tensor<1000000000xi1>
    return %2 : tensor<1000000000xi1>
  }
}

// CHECK:  func.func @main() -> tensor<1000000000xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<false> : tensor<1000000000xi1>
// CHECK-NEXT:    return %c : tensor<1000000000xi1>
// CHECK-NEXT:  }
