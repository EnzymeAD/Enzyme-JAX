// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s --mlir-print-ir-after-all | FileCheck %s

module {
   func.func @test(%input: tensor<2xf32>) -> tensor<1xf32> {
    %slice0 = stablehlo.slice %input [0:1] : (tensor<2xf32>) -> tensor<1xf32>
    %slice1 = stablehlo.slice %input [1:2] : (tensor<2xf32>) -> tensor<1xf32>
    %sum = stablehlo.add %slice0, %slice1 : tensor<1xf32>
    %result = stablehlo.add %sum, %slice0 : tensor<1xf32>
    return %result : tensor<1xf32>
  }
}

// CHECK:  func.func @test(%[[ARG0:.+]]: tensor<2xf32>) -> tensor<1xf32> {
// CHECK-NEXT:    %[[CST:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[SLICE0:.+]] = stablehlo.slice %[[ARG0]] [0:1] : (tensor<2xf32>) -> tensor<1xf32>
// CHECK-NEXT:    %[[REDUCE:.+]] = stablehlo.reduce(%[[ARG0]] init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:    %[[RESHAPE:.+]] = stablehlo.reshape %[[REDUCE]] : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:    %[[RESULT:.+]] = stablehlo.add %[[RESHAPE]], %[[SLICE0]] : tensor<1xf32>
// CHECK-NEXT:    return %[[RESULT]] : tensor<1xf32>
// CHECK-NEXT:  }
