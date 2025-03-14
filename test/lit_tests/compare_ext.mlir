// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%arg0: tensor<3xi1>, %arg1: tensor<3xi1>) -> (tensor<3xi1>, tensor<3xi1>, tensor<3xi1>) {
    %c = stablehlo.constant dense<0> : tensor<3xi64>
    %c_1 = stablehlo.constant dense<1> : tensor<3xi64>
    %0 = stablehlo.convert %arg0 : (tensor<3xi1>) -> tensor<3xi64>
    %1 = stablehlo.compare  EQ, %0, %c,  UNSIGNED : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
    %2 = stablehlo.compare  EQ, %0, %c_1,  UNSIGNED : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
    %3 = stablehlo.convert %arg1 : (tensor<3xi1>) -> tensor<3xi64>
    %4 = stablehlo.compare  EQ, %0, %3,  UNSIGNED : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
    return %1, %2, %4 : tensor<3xi1>, tensor<3xi1>, tensor<3xi1>
  }
}

// CHECK:  func.func @main(%arg0: tensor<3xi1>, %arg1: tensor<3xi1>) -> (tensor<3xi1>, tensor<3xi1>, tensor<3xi1>) {
// CHECK-NEXT:    %0 = stablehlo.not %arg0 : tensor<3xi1>
// CHECK-NEXT:    %1 = stablehlo.and %arg0, %arg1 : tensor<3xi1>
// CHECK-NEXT:    return %0, %arg0, %1 : tensor<3xi1>
// CHECK-NEXT:  }
