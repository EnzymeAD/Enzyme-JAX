// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%arg0: tensor<i64>) -> tensor<i64> {
    %cst = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.divide %arg0, %cst : tensor<i64>
    return %0 : tensor<i64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<i64>) -> tensor<i64> {
// CHECK-NEXT:    return %arg0 : tensor<i64>
// CHECK-NEXT:  }
