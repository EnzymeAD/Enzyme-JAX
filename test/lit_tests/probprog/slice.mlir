// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-probprog-to-stablehlo{backend=cpu})" | FileCheck %s

module {
  // CHECK-LABEL: @test_slice_2d_middle
  // CHECK:         %[[RESULT:.+]] = stablehlo.slice %{{.+}} [0:1, 5:11] : (tensor<1x20xf64>) -> tensor<1x6xf64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<1x6xf64>
  func.func @test_slice_2d_middle(%input: tensor<1x20xf64>) -> tensor<1x6xf64> {
    %sliced = enzyme.slice %input {start_indices = array<i64: 0, 5>, limit_indices = array<i64: 1, 11>, strides = array<i64: 1, 1>} : (tensor<1x20xf64>) -> tensor<1x6xf64>
    return %sliced : tensor<1x6xf64>
  }

  // CHECK-LABEL: @test_slice_2d_start
  // CHECK:         %[[RESULT:.+]] = stablehlo.slice %{{.+}} [0:1, 0:3] : (tensor<1x10xf64>) -> tensor<1x3xf64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<1x3xf64>
  func.func @test_slice_2d_start(%input: tensor<1x10xf64>) -> tensor<1x3xf64> {
    %sliced = enzyme.slice %input {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 1, 3>, strides = array<i64: 1, 1>} : (tensor<1x10xf64>) -> tensor<1x3xf64>
    return %sliced : tensor<1x3xf64>
  }

  // CHECK-LABEL: @test_slice_1d
  // CHECK:         %[[RESULT:.+]] = stablehlo.slice %{{.+}} [1:4] : (tensor<5xf64>) -> tensor<3xf64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<3xf64>
  func.func @test_slice_1d(%input: tensor<5xf64>) -> tensor<3xf64> {
    %sliced = enzyme.slice %input {start_indices = array<i64: 1>, limit_indices = array<i64: 4>, strides = array<i64: 1>} : (tensor<5xf64>) -> tensor<3xf64>
    return %sliced : tensor<3xf64>
  }
}
