// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-probprog-to-stablehlo{backend=cpu})" | FileCheck %s

module {
  // CHECK-LABEL: @test_dynamic_slice_2d
  // CHECK:         %[[RESULT:.+]] = stablehlo.dynamic_slice %{{.+}}, %{{.+}}, %{{.+}}, sizes = [1, 4] : (tensor<5x4xf64>, tensor<i64>, tensor<i64>) -> tensor<1x4xf64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<1x4xf64>
  func.func @test_dynamic_slice_2d(%input: tensor<5x4xf64>, %idx0: tensor<i64>, %idx1: tensor<i64>) -> tensor<1x4xf64> {
    %sliced = enzyme.dynamic_slice %input, %idx0, %idx1 {slice_sizes = array<i64: 1, 4>} : (tensor<5x4xf64>, tensor<i64>, tensor<i64>) -> tensor<1x4xf64>
    return %sliced : tensor<1x4xf64>
  }

  // CHECK-LABEL: @test_dynamic_slice_1d
  // CHECK:         %[[RESULT:.+]] = stablehlo.dynamic_slice %{{.+}}, %{{.+}}, sizes = [3] : (tensor<10xf64>, tensor<i64>) -> tensor<3xf64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<3xf64>
  func.func @test_dynamic_slice_1d(%input: tensor<10xf64>, %idx: tensor<i64>) -> tensor<3xf64> {
    %sliced = enzyme.dynamic_slice %input, %idx {slice_sizes = array<i64: 3>} : (tensor<10xf64>, tensor<i64>) -> tensor<3xf64>
    return %sliced : tensor<3xf64>
  }

  // CHECK-LABEL: @test_dynamic_slice_large
  // CHECK:         %[[RESULT:.+]] = stablehlo.dynamic_slice %{{.+}}, %{{.+}}, %{{.+}}, sizes = [1, 50] : (tensor<100x50xf64>, tensor<i64>, tensor<i64>) -> tensor<1x50xf64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<1x50xf64>
  func.func @test_dynamic_slice_large(%input: tensor<100x50xf64>, %idx0: tensor<i64>, %idx1: tensor<i64>) -> tensor<1x50xf64> {
    %sliced = enzyme.dynamic_slice %input, %idx0, %idx1 {slice_sizes = array<i64: 1, 50>} : (tensor<100x50xf64>, tensor<i64>, tensor<i64>) -> tensor<1x50xf64>
    return %sliced : tensor<1x50xf64>
  }
}
