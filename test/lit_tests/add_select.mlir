// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%58: tensor<185xi1>, %cst_4: tensor<185xf64>, %55:tensor<185xf64>) -> tensor<185xf64> {
    %59 = stablehlo.select %58, %cst_4, %55 : tensor<185xi1>, tensor<185xf64>
    %60 = stablehlo.select %58, %55, %cst_4 : tensor<185xi1>, tensor<185xf64>
    %61 = stablehlo.add %59, %60 : tensor<185xf64>
    return %61 : tensor<185xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<185xi1>, %arg1: tensor<185xf64>, %arg2: tensor<185xf64>) -> tensor<185xf64> {
// CHECK-NEXT:    %0 = stablehlo.add %arg1, %arg2 : tensor<185xf64>
// CHECK-NEXT:    return %0 : tensor<185xf64>
// CHECK-NEXT:  }
