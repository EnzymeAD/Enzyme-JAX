// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt)" %s | FileCheck %s 

module {

  func.func @main() -> tensor<i1> {
    %c_118 = stablehlo.constant dense<0> : tensor<i32>
    %c_119 = stablehlo.constant dense<256> : tensor<i32>
    %2 = stablehlo.compare  EQ, %c_119, %c_118 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %2 : tensor<i1>
  }
}

// CHECK:  func.func @main() -> tensor<i1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:    return %c : tensor<i1>
// CHECK-NEXT:  }
