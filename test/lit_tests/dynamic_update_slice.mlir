// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main() -> tensor<3xi64> {
    %init = stablehlo.constant dense<[1,2,3]> : tensor<3xi64>
    %update = stablehlo.constant dense<2> : tensor<1xi64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c2 = stablehlo.constant dense<2> : tensor<i64>
    %1 = stablehlo.dynamic_update_slice %init, %update, %c : (tensor<3xi64>, tensor<1xi64>, tensor<i64>) -> tensor<3xi64>
    %2 = stablehlo.dynamic_update_slice %1, %update, %c2 : (tensor<3xi64>, tensor<1xi64>, tensor<i64>) -> tensor<3xi64>
    return %2 : tensor<3xi64>
  }
}

// CHECK:  func.func @main() -> tensor<3xi64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<2> : tensor<3xi64>
// CHECK-NEXT:    return %c : tensor<3xi64>
// CHECK-NEXT:  }
