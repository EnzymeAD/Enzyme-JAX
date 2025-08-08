// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=select_pad" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<4x1519x3056xf64>, %arg1: tensor<4x1519x3056xf64>, %arg2: tensor<4x1520x3056xi1>) -> tensor<4x1520x3056xf64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 1, 0], interior = [0, 0, 0] : (tensor<4x1519x3056xf64>, tensor<f64>) -> tensor<4x1520x3056xf64>
    %1 = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 1, 0], interior = [0, 0, 0] : (tensor<4x1519x3056xf64>, tensor<f64>) -> tensor<4x1520x3056xf64>
    %2 = stablehlo.select %arg2, %0, %1 : tensor<4x1520x3056xi1>, tensor<4x1520x3056xf64>
    return %2 : tensor<4x1520x3056xf64>
  }
}

// CHECK:      func.func @main(%arg0: tensor<4x1519x3056xf64>, %arg1: tensor<4x1519x3056xf64>, %arg2: tensor<4x1520x3056xi1>) -> tensor<4x1520x3056xf64> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %0 = stablehlo.slice %arg2 [0:4, 0:1519, 0:3056] : (tensor<4x1520x3056xi1>) -> tensor<4x1519x3056xi1>
// CHECK-NEXT:   %1 = stablehlo.select %0, %arg0, %arg1 : tensor<4x1519x3056xi1>, tensor<4x1519x3056xf64>
// CHECK-NEXT:   %2 = stablehlo.pad %1, %cst, low = [0, 0, 0], high = [0, 1, 0], interior = [0, 0, 0] : (tensor<4x1519x3056xf64>, tensor<f64>) -> tensor<4x1520x3056xf64>
// CHECK-NEXT:   return %2 : tensor<4x1520x3056xf64>
// CHECK-NEXT: }
