// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main1(%arg0: tensor<1536xf64>) -> tensor<1519x3056xf64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.slice %arg0 [9:10] : (tensor<1536xf64>) -> tensor<1xf64>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<1xf64>) -> tensor<1x3056xf64>
    %2 = stablehlo.slice %arg0 [10:1528] : (tensor<1536xf64>) -> tensor<1518xf64>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<1518xf64>) -> tensor<1518x3056xf64>
    %4 = stablehlo.concatenate %1, %3, dim = 0 : (tensor<1x3056xf64>, tensor<1518x3056xf64>) -> tensor<1519x3056xf64>
    return %4 : tensor<1519x3056xf64>
}

// CHECK: func.func @main1(%arg0: tensor<1536xf64>) -> tensor<1519x3056xf64> {
// CHECK-NEXT:     %0 = stablehlo.slice %arg0 [9:1528] : (tensor<1536xf64>) -> tensor<1519xf64>
// CHECK-NEXT:     %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<1519xf64>) -> tensor<1519x3056xf64>
// CHECK-NEXT:     return %1 : tensor<1519x3056xf64>
// CHECK-NEXT: }

func.func @main2(%arg0: tensor<2xi1>, %arg1: tensor<2xi1>) -> (tensor<3x4xi1>) {
  %5 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<2xi1>) -> tensor<3x2xi1>
  %b = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2xi1>) -> tensor<3x2xi1>
  %6 = stablehlo.concatenate %5, %b, dim = 1 : (tensor<3x2xi1>, tensor<3x2xi1>) -> tensor<3x4xi1>
  return %6 : tensor<3x4xi1>
}

// CHECK:  func.func @main2(%arg0: tensor<2xi1>, %arg1: tensor<2xi1>) -> tensor<3x4xi1> {
// CHECK-NEXT:    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<2xi1>, tensor<2xi1>) -> tensor<4xi1>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<4xi1>) -> tensor<3x4xi1>
// CHECK-NEXT:    return %1 : tensor<3x4xi1>
// CHECK-NEXT:  }
