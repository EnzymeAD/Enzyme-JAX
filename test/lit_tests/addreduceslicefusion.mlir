// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
   func.func @test(%input: tensor<2xf32>) -> tensor<1xf32> {
    %slice0 = stablehlo.slice %input [0:1] : (tensor<2xf32>) -> tensor<1xf32>
    %slice1 = stablehlo.slice %input [1:2] : (tensor<2xf32>) -> tensor<1xf32>
    %sum = stablehlo.add %slice0, %slice1 : tensor<1xf32>
    %result = stablehlo.add %sum, %slice0 : tensor<1xf32>
    return %result : tensor<1xf32>
  }

  func.func @test_max(%input: tensor<2xf32>) -> tensor<1xf32> {
    %slice0 = stablehlo.slice %input [0:1] : (tensor<2xf32>) -> tensor<1xf32>
    %slice1 = stablehlo.slice %input [1:2] : (tensor<2xf32>) -> tensor<1xf32>
    %max1 = stablehlo.maximum %slice0, %slice1 : tensor<1xf32>
    %result = stablehlo.maximum %max1, %slice0 : tensor<1xf32>
    return %result : tensor<1xf32>
  }

  func.func @test_min(%input: tensor<2xf32>) -> tensor<1xf32> {
    %slice0 = stablehlo.slice %input [0:1] : (tensor<2xf32>) -> tensor<1xf32>
    %slice1 = stablehlo.slice %input [1:2] : (tensor<2xf32>) -> tensor<1xf32>
    %min1 = stablehlo.minimum %slice0, %slice1 : tensor<1xf32>
    %result = stablehlo.minimum %min1, %slice0 : tensor<1xf32>
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

// CHECK:  func.func @test_max(%[[ARG0:.+]]: tensor<2xf32>) -> tensor<1xf32> {
// CHECK-NEXT:    %[[CST:.+]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-NEXT:    %[[REDUCE:.+]] = stablehlo.reduce(%[[ARG0]] init: %[[CST]]) applies stablehlo.maximum across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:    %[[RESHAPE:.+]] = stablehlo.reshape %[[REDUCE]] : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:    return %[[RESHAPE]] : tensor<1xf32>
// CHECK-NEXT:  }

// CHECK:  func.func @test_min(%[[ARG0:.+]]: tensor<2xf32>) -> tensor<1xf32> {
// CHECK-NEXT:    %[[CST:.+]] = stablehlo.constant dense<0x7F800000> : tensor<f32>
// CHECK-NEXT:    %[[REDUCE:.+]] = stablehlo.reduce(%[[ARG0]] init: %[[CST]]) applies stablehlo.minimum across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:    %[[RESHAPE:.+]] = stablehlo.reshape %[[REDUCE]] : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:    return %[[RESHAPE]] : tensor<1xf32>
// CHECK-NEXT:  }
