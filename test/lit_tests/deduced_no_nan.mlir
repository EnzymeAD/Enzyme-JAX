// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

func.func @main1() -> tensor<10000xf64> {
    %c = stablehlo.constant dense<-91> : tensor<10000xi64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<10000xf64>
    %0 = stablehlo.iota dim = 0 : tensor<10000xi64>
    %1 = stablehlo.add %0, %c : tensor<10000xi64>
    %2 = stablehlo.convert %1 : (tensor<10000xi64>) -> tensor<10000xf64>
    %3 = stablehlo.multiply %2, %cst : tensor<10000xf64>
    %4 = stablehlo.subtract %2, %2 : tensor<10000xf64>
    %5 = stablehlo.add %3, %4 : tensor<10000xf64>
    return %5 : tensor<10000xf64>
}

// CHECK: func.func @main1() -> tensor<10000xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<10000xf64>
// CHECK-NEXT:     return %cst : tensor<10000xf64>
// CHECK-NEXT: }
