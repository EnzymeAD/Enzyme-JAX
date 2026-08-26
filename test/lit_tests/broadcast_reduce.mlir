// RUN:enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=broadcast_reduce" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

module {
  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x4xf32> {
    %cst = stablehlo.constant dense<0.000> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2] : (tensor<2x3xf32>) -> tensor<2x5x3x4xf32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2, 1] : (tensor<2x5x3x4xf32>, tensor<f32>) -> tensor<2x4xf32>
    return %1 : tensor<2x4xf32>
  }
}

// CHECK:  func.func @main(%[[ARG:.+]]: tensor<2x3xf32>) -> tensor<2x4xf32> {
// CHECK-NEXT:    %[[CST5:.+]] = stablehlo.constant dense<5> : tensor<2x4xi64>
// CHECK-NEXT:    %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[BROAD:.+]] = stablehlo.broadcast_in_dim %[[ARG]], dims = [0, 1] : (tensor<2x3xf32>) -> tensor<2x3x4xf32>
// CHECK-NEXT:    %[[REDUC:.+]] = stablehlo.reduce(%[[BROAD]] init: %[[ZERO]]) applies stablehlo.add across dimensions = [1] : (tensor<2x3x4xf32>, tensor<f32>) -> tensor<2x4xf32>
// CHECK-NEXT:    %[[CONVE:.+]] = stablehlo.convert %[[CST5:.+]] : (tensor<2x4xi64>) -> tensor<2x4xf32>
// CHECK-NEXT:    %[[MULTI:.+]] = stablehlo.multiply %[[REDUC]], %[[CONVE]] : tensor<2x4xf32>
// CHECK-NEXT:    return %[[MULTI]] : tensor<2x4xf32>
// CHECK-NEXT:  }
