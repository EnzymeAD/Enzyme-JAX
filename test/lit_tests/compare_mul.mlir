// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%2860:  tensor<3056xf64>) -> tensor<3056xi1> {
    %cst_116 = stablehlo.constant dense<0.000000e+00> : tensor<3056xf64>

    %cst_118 = stablehlo.constant dense<0.052356020942397663> : tensor<3056xf64>
      %2861 = stablehlo.multiply %2860, %cst_118 : tensor<3056xf64> 
    %2864 = stablehlo.compare  LT, %cst_116, %2861,  FLOAT : (tensor<3056xf64>, tensor<3056xf64>) -> tensor<3056xi1>
    return %2864 : tensor<3056xi1>
  }
}

// CHECK:  func.func @main(%arg0: tensor<3056xf64>) -> tensor<3056xi1> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<3056xf64>
// CHECK-NEXT:    %0 = stablehlo.compare  GT, %arg0, %cst : (tensor<3056xf64>, tensor<3056xf64>) -> tensor<3056xi1>
// CHECK-NEXT:    return %0 : tensor<3056xi1> 
// CHECK-NEXT:  }
