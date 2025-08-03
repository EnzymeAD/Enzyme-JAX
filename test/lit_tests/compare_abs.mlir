// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%2861:  tensor<3056xf64>) -> tensor<3056xi1> {
    %cst_116 = stablehlo.constant dense<0.000000e+00> : tensor<3056xf64>
    %2863 = stablehlo.abs %2861 : tensor<3056xf64>
    %2864 = stablehlo.compare  LT, %cst_116, %2863,  FLOAT : (tensor<3056xf64>, tensor<3056xf64>) -> tensor<3056xi1>
    return %2864 : tensor<3056xi1>
  }
}

// CHECK:  func.func @main(%arg0: tensor<3xi1>, %arg1: tensor<3xi1>) -> (tensor<3xi1>, tensor<3xi1>, tensor<3xi1>) {
// CHECK-NEXT:    %0 = stablehlo.not %arg0 : tensor<3xi1>
// CHECK-NEXT:    %1 = stablehlo.and %arg0, %arg1 : tensor<3xi1>
// CHECK-NEXT:    return %0, %arg0, %1 : tensor<3xi1>
// CHECK-NEXT:  }
