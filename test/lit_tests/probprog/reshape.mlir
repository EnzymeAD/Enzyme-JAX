// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-probprog-to-stablehlo{backend=cpu})" | FileCheck %s

module {
  // CHECK-LABEL: @test_reshape_expand
  // CHECK:         %[[RESULT:.+]] = stablehlo.reshape %{{.+}} : (tensor<5xf64>) -> tensor<1x5xf64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<1x5xf64>
  func.func @test_reshape_expand(%input: tensor<5xf64>) -> tensor<1x5xf64> {
    %reshaped = enzyme.reshape %input : (tensor<5xf64>) -> tensor<1x5xf64>
    return %reshaped : tensor<1x5xf64>
  }

  // CHECK-LABEL: @test_reshape_flatten
  // CHECK:         %[[RESULT:.+]] = stablehlo.reshape %{{.+}} : (tensor<1x5xf64>) -> tensor<5xf64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<5xf64>
  func.func @test_reshape_flatten(%input: tensor<1x5xf64>) -> tensor<5xf64> {
    %reshaped = enzyme.reshape %input : (tensor<1x5xf64>) -> tensor<5xf64>
    return %reshaped : tensor<5xf64>
  }

  // CHECK-LABEL: @test_reshape_3d
  // CHECK:         %[[RESULT:.+]] = stablehlo.reshape %{{.+}} : (tensor<3x3x3xf64>) -> tensor<27x1xf64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<27x1xf64>
  func.func @test_reshape_3d(%input: tensor<3x3x3xf64>) -> tensor<27x1xf64> {
    %reshaped = enzyme.reshape %input : (tensor<3x3x3xf64>) -> tensor<27x1xf64>
    return %reshaped : tensor<27x1xf64>
  }
}
