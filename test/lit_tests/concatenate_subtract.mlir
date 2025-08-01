// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

func.func @main1(%arg0: tensor<1520x3056xf64>, %arg1: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
    %0 = stablehlo.slice %arg0 [1:2, 0:3056] : (tensor<1520x3056xf64>) -> tensor<1x3056xf64>
    %1 = stablehlo.slice %arg0 [2:1520, 0:3056] : (tensor<1520x3056xf64>) -> tensor<1518x3056xf64>
    %2 = stablehlo.slice %arg1 [1:1519, 0:3056] : (tensor<1520x3056xf64>) -> tensor<1518x3056xf64>
    %3 = stablehlo.slice %arg1 [1519:1520, 0:3056] : (tensor<1520x3056xf64>) -> tensor<1x3056xf64>
    %4 = stablehlo.subtract %1, %2 : tensor<1518x3056xf64>
    %5 = stablehlo.negate %3 : tensor<1x3056xf64>
    %6 = stablehlo.concatenate %0, %4, %5, dim = 0 : (tensor<1x3056xf64>, tensor<1518x3056xf64>, tensor<1x3056xf64>) -> tensor<1520x3056xf64>
    return %6 : tensor<1520x3056xf64>
}

// CHECK: func.func @main1(%arg0: tensor<1520x3056xf64>, %arg1: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.slice %arg0 [1:1520, 0:3056] : (tensor<1520x3056xf64>) -> tensor<1519x3056xf64>
// CHECK-NEXT:     %1 = stablehlo.pad %0, %cst, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1519x3056xf64>, tensor<f64>) -> tensor<1520x3056xf64>
// CHECK-NEXT:     %2 = stablehlo.slice %arg1 [1:1520, 0:3056] : (tensor<1520x3056xf64>) -> tensor<1519x3056xf64>
// CHECK-NEXT:     %3 = stablehlo.pad %2, %cst, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1519x3056xf64>, tensor<f64>) -> tensor<1520x3056xf64>
// CHECK-NEXT:     %4 = stablehlo.subtract %1, %3 : tensor<1520x3056xf64>
// CHECK-NEXT:     return %4 : tensor<1520x3056xf64>
// CHECK-NEXT: }

// TODO: do the more general case with multiplication in between
