// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-probprog-to-stablehlo{backend=cpu})" | FileCheck %s

module {
  // CHECK-LABEL: @test_dynamic_update_1d_scalar
  // CHECK:         %[[RESHAPED:.+]] = stablehlo.reshape %{{.+}} : (tensor<f64>) -> tensor<1xf64>
  // CHECK-NEXT:    %[[RESULT:.+]] = stablehlo.dynamic_update_slice %{{.+}}, %[[RESHAPED]], %{{.+}} : (tensor<10xf64>, tensor<1xf64>, tensor<i64>) -> tensor<10xf64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<10xf64>
  func.func @test_dynamic_update_1d_scalar(%input: tensor<10xf64>, %index: tensor<i64>, %value: tensor<f64>) -> tensor<10xf64> {
    %result = enzyme.dynamic_update %input, %index, %value : (tensor<10xf64>, tensor<i64>, tensor<f64>) -> tensor<10xf64>
    return %result : tensor<10xf64>
  }

  // CHECK-LABEL: @test_dynamic_update_1d_scalar_f32
  // CHECK:         %[[RESHAPED:.+]] = stablehlo.reshape %{{.+}} : (tensor<f32>) -> tensor<1xf32>
  // CHECK-NEXT:    %[[RESULT:.+]] = stablehlo.dynamic_update_slice %{{.+}}, %[[RESHAPED]], %{{.+}} : (tensor<8xf32>, tensor<1xf32>, tensor<i64>) -> tensor<8xf32>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<8xf32>
  func.func @test_dynamic_update_1d_scalar_f32(%input: tensor<8xf32>, %index: tensor<i64>, %value: tensor<f32>) -> tensor<8xf32> {
    %result = enzyme.dynamic_update %input, %index, %value : (tensor<8xf32>, tensor<i64>, tensor<f32>) -> tensor<8xf32>
    return %result : tensor<8xf32>
  }

  // CHECK-LABEL: @test_dynamic_update_2d_1d
  // CHECK:         %[[RESHAPED:.+]] = stablehlo.reshape %{{.+}} : (tensor<4xf64>) -> tensor<1x4xf64>
  // CHECK-NEXT:    %[[ZERO:.+]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK-NEXT:    %[[RESULT:.+]] = stablehlo.dynamic_update_slice %{{.+}}, %[[RESHAPED]], %{{.+}}, %[[ZERO]] : (tensor<5x4xf64>, tensor<1x4xf64>, tensor<i64>, tensor<i64>) -> tensor<5x4xf64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<5x4xf64>
  func.func @test_dynamic_update_2d_1d(%input: tensor<5x4xf64>, %index: tensor<i64>, %value: tensor<4xf64>) -> tensor<5x4xf64> {
    %result = enzyme.dynamic_update %input, %index, %value : (tensor<5x4xf64>, tensor<i64>, tensor<4xf64>) -> tensor<5x4xf64>
    return %result : tensor<5x4xf64>
  }

  // CHECK-LABEL: @test_dynamic_update_2d_1d_large
  // CHECK:         %[[RESHAPED:.+]] = stablehlo.reshape %{{.+}} : (tensor<50xf64>) -> tensor<1x50xf64>
  // CHECK-NEXT:    %[[ZERO:.+]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK-NEXT:    %[[RESULT:.+]] = stablehlo.dynamic_update_slice %{{.+}}, %[[RESHAPED]], %{{.+}}, %[[ZERO]] : (tensor<100x50xf64>, tensor<1x50xf64>, tensor<i64>, tensor<i64>) -> tensor<100x50xf64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<100x50xf64>
  func.func @test_dynamic_update_2d_1d_large(%input: tensor<100x50xf64>, %index: tensor<i64>, %value: tensor<50xf64>) -> tensor<100x50xf64> {
    %result = enzyme.dynamic_update %input, %index, %value : (tensor<100x50xf64>, tensor<i64>, tensor<50xf64>) -> tensor<100x50xf64>
    return %result : tensor<100x50xf64>
  }
}
