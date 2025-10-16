// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%58: tensor<185xi1>, %cst_4: tensor<185xf64>, %55:tensor<185xf64>) -> tensor<185xf64> {
    %59 = stablehlo.select %58, %cst_4, %55 : tensor<185xi1>, tensor<185xf64>
    %60 = stablehlo.select %58, %55, %cst_4 : tensor<185xi1>, tensor<185xf64>
    %61 = stablehlo.add %59, %60 : tensor<185xf64>
    return %61 : tensor<185xf64>
  }
  func.func @main2(%18: tensor<185xi1>, %cst_5: tensor<185xf64>, %2:tensor<185xf64>, %3 : tensor<185xf64>) -> tensor<185xf64> {
    %19 = stablehlo.select %18, %cst_5, %2 : tensor<185xi1>, tensor<185xf64>
    %20 = stablehlo.select %18, %2, %cst_5 : tensor<185xi1>, tensor<185xf64>
    %21 = stablehlo.subtract %19, %3 : tensor<185xf64>
    %22 = stablehlo.add %20, %21 : tensor<185xf64>
    return %22 : tensor<185xf64>
  }
}


// CHECK:  func.func @main(%arg0: tensor<185xi1>, %arg1: tensor<185xf64>, %arg2: tensor<185xf64>) -> tensor<185xf64> {
// CHECK-NEXT:    %0 = stablehlo.add %arg1, %arg2 : tensor<185xf64>
// CHECK-NEXT:    return %0 : tensor<185xf64>
// CHECK-NEXT:  }

// CHECK:  func.func @main2(%arg0: tensor<185xi1>, %arg1: tensor<185xf64>, %arg2: tensor<185xf64>, %arg3: tensor<185xf64>) -> tensor<185xf64> {
// CHECK-NEXT:    %0 = stablehlo.add %arg1, %arg2 {enzymexla.guaranteed_finite = false} : tensor<185xf64>
// CHECK-NEXT:    %1 = stablehlo.subtract %0, %arg3 {enzymexla.guaranteed_no_nan = false} : tensor<185xf64>
// CHECK-NEXT:    return %1 : tensor<185xf64>
// CHECK-NEXT:  }
