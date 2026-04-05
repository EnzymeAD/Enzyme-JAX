// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%5170: tensor<1520x3056xf64>, %5177: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
    %c_92 = stablehlo.constant dense<0> : tensor<1520xi64>
    %c_43 = stablehlo.constant dense<-1519> : tensor<i64>
    %c_93 = stablehlo.constant dense<1> : tensor<1520xi64>
    %1838 = stablehlo.iota dim = 0 : tensor<1520xi64> // [0, ..., 1520]
    %1839 = stablehlo.add %1838, %c_92 : tensor<1520xi64>
    %1840 = stablehlo.multiply %1839, %c_93 : tensor<1520xi64>
    %5178 = stablehlo.broadcast_in_dim %c_43, dims = [] : (tensor<i64>) -> tensor<1520xi64> // [-1519, ..., -1519]
    %5179 = stablehlo.add %1840, %5178 : tensor<1520xi64>
    %5180 = stablehlo.compare EQ, %5179, %c_92 : (tensor<1520xi64>, tensor<1520xi64>) -> tensor<1520xi1> // [i == 1519, ...]
    %5181 = stablehlo.broadcast_in_dim %5180, dims = [0] : (tensor<1520xi1>) -> tensor<1520x3056xi1>
    %5182 = stablehlo.select %5181, %5170, %5177 : tensor<1520x3056xi1>, tensor<1520x3056xf64>
    return %5182 : tensor<1520x3056xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<1520x3056xf64>, %arg1: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [0:1519, 0:3056] : (tensor<1520x3056xf64>) -> tensor<1519x3056xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [1519:1520, 0:3056] : (tensor<1520x3056xf64>) -> tensor<1x3056xf64>
// CHECK-NEXT:    %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<1519x3056xf64>, tensor<1x3056xf64>) -> tensor<1520x3056xf64>
// CHECK-NEXT:    return %2 : tensor<1520x3056xf64>
// CHECK-NEXT:  }
