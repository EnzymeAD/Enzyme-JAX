// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-probprog-to-stablehlo{backend=cpu})" | FileCheck %s

module {
  // CHECK-LABEL: @test_logistic_scalar
  // CHECK:         %[[RESULT:.+]] = stablehlo.logistic %{{.+}} : tensor<f64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<f64>
  func.func @test_logistic_scalar(%x: tensor<f64>) -> tensor<f64> {
    %result = enzyme.logistic %x : (tensor<f64>) -> tensor<f64>
    return %result : tensor<f64>
  }

  // CHECK-LABEL: @test_logistic_1d
  // CHECK:         %[[RESULT:.+]] = stablehlo.logistic %{{.+}} : tensor<10xf64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<10xf64>
  func.func @test_logistic_1d(%x: tensor<10xf64>) -> tensor<10xf64> {
    %result = enzyme.logistic %x : (tensor<10xf64>) -> tensor<10xf64>
    return %result : tensor<10xf64>
  }

  // CHECK-LABEL: @test_logistic_2d
  // CHECK:         %[[RESULT:.+]] = stablehlo.logistic %{{.+}} : tensor<3x4xf64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<3x4xf64>
  func.func @test_logistic_2d(%x: tensor<3x4xf64>) -> tensor<3x4xf64> {
    %result = enzyme.logistic %x : (tensor<3x4xf64>) -> tensor<3x4xf64>
    return %result : tensor<3x4xf64>
  }

  // CHECK-LABEL: @test_logistic_f32
  // CHECK:         %[[RESULT:.+]] = stablehlo.logistic %{{.+}} : tensor<5xf32>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<5xf32>
  func.func @test_logistic_f32(%x: tensor<5xf32>) -> tensor<5xf32> {
    %result = enzyme.logistic %x : (tensor<5xf32>) -> tensor<5xf32>
    return %result : tensor<5xf32>
  }
}
