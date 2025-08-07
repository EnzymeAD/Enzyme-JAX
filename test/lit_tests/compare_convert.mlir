// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%2861:  tensor<3056xi64>) -> tensor<3056xi1> {
    %cst_116 = stablehlo.constant dense<0.000000e+00> : tensor<3056xf64>
    %2863 = stablehlo.convert %2861 : (tensor<3056xi64>) -> tensor<3056xf64>
    %2864 = stablehlo.compare  LT, %cst_116, %2863,  FLOAT : (tensor<3056xf64>, tensor<3056xf64>) -> tensor<3056xi1>
    return %2864 : tensor<3056xi1>
  }
}

// CHECK:  func.func @main(%arg0: tensor<3056xi64>) -> tensor<3056xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<3056xi64>
// CHECK-NEXT:    %0 = stablehlo.compare  LT, %c, %arg0 : (tensor<3056xi64>, tensor<3056xi64>) -> tensor<3056xi1>
// CHECK-NEXT:    return %0 : tensor<3056xi1>
// CHECK-NEXT:  }
