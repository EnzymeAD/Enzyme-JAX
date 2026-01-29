// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-probprog-to-stablehlo{backend=cpu})" | FileCheck %s

module {
  // CHECK-LABEL: @test_dynamic_extract_1d_to_scalar
  // CHECK:         %[[SLICE:.+]] = stablehlo.dynamic_slice %{{.+}}, %{{.+}}, sizes = [1] : (tensor<10xf64>, tensor<i64>) -> tensor<1xf64>
  // CHECK-NEXT:    %[[RESULT:.+]] = stablehlo.reshape %[[SLICE]] : (tensor<1xf64>) -> tensor<f64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<f64>
  func.func @test_dynamic_extract_1d_to_scalar(%input: tensor<10xf64>, %index: tensor<i64>) -> tensor<f64> {
    %result = enzyme.dynamic_extract %input, %index : (tensor<10xf64>, tensor<i64>) -> tensor<f64>
    return %result : tensor<f64>
  }

  // CHECK-LABEL: @test_dynamic_extract_1d_to_scalar_f32
  // CHECK:         %[[SLICE:.+]] = stablehlo.dynamic_slice %{{.+}}, %{{.+}}, sizes = [1] : (tensor<8xf32>, tensor<i64>) -> tensor<1xf32>
  // CHECK-NEXT:    %[[RESULT:.+]] = stablehlo.reshape %[[SLICE]] : (tensor<1xf32>) -> tensor<f32>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<f32>
  func.func @test_dynamic_extract_1d_to_scalar_f32(%input: tensor<8xf32>, %index: tensor<i64>) -> tensor<f32> {
    %result = enzyme.dynamic_extract %input, %index : (tensor<8xf32>, tensor<i64>) -> tensor<f32>
    return %result : tensor<f32>
  }

  // CHECK-LABEL: @test_dynamic_extract_2d_to_1d
  // CHECK:         %[[ZERO:.+]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK-NEXT:    %[[SLICE:.+]] = stablehlo.dynamic_slice %{{.+}}, %{{.+}}, %[[ZERO]], sizes = [1, 4] : (tensor<5x4xf64>, tensor<i64>, tensor<i64>) -> tensor<1x4xf64>
  // CHECK-NEXT:    %[[RESULT:.+]] = stablehlo.reshape %[[SLICE]] : (tensor<1x4xf64>) -> tensor<4xf64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<4xf64>
  func.func @test_dynamic_extract_2d_to_1d(%input: tensor<5x4xf64>, %index: tensor<i64>) -> tensor<4xf64> {
    %result = enzyme.dynamic_extract %input, %index : (tensor<5x4xf64>, tensor<i64>) -> tensor<4xf64>
    return %result : tensor<4xf64>
  }

  // CHECK-LABEL: @test_dynamic_extract_2d_to_1d_large
  // CHECK:         %[[ZERO:.+]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK-NEXT:    %[[SLICE:.+]] = stablehlo.dynamic_slice %{{.+}}, %{{.+}}, %[[ZERO]], sizes = [1, 50] : (tensor<100x50xf64>, tensor<i64>, tensor<i64>) -> tensor<1x50xf64>
  // CHECK-NEXT:    %[[RESULT:.+]] = stablehlo.reshape %[[SLICE]] : (tensor<1x50xf64>) -> tensor<50xf64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<50xf64>
  func.func @test_dynamic_extract_2d_to_1d_large(%input: tensor<100x50xf64>, %index: tensor<i64>) -> tensor<50xf64> {
    %result = enzyme.dynamic_extract %input, %index : (tensor<100x50xf64>, tensor<i64>) -> tensor<50xf64>
    return %result : tensor<50xf64>
  }
}
