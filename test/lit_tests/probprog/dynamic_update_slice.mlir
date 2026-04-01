// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-probprog-to-stablehlo{backend=cpu})" | FileCheck %s

module {
  // CHECK-LABEL: @test_dynamic_update_slice_2d
  // CHECK:         %[[RESULT:.+]] = stablehlo.dynamic_update_slice %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : (tensor<5x4xf64>, tensor<1x4xf64>, tensor<i64>, tensor<i64>) -> tensor<5x4xf64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<5x4xf64>
  func.func @test_dynamic_update_slice_2d(%input: tensor<5x4xf64>, %update: tensor<1x4xf64>, %idx0: tensor<i64>, %idx1: tensor<i64>) -> tensor<5x4xf64> {
    %result = enzyme.dynamic_update_slice %input, %update, %idx0, %idx1 : (tensor<5x4xf64>, tensor<1x4xf64>, tensor<i64>, tensor<i64>) -> tensor<5x4xf64>
    return %result : tensor<5x4xf64>
  }

  // CHECK-LABEL: @test_dynamic_update_slice_1d
  // CHECK:         %[[RESULT:.+]] = stablehlo.dynamic_update_slice %{{.+}}, %{{.+}}, %{{.+}} : (tensor<10xf64>, tensor<1xf64>, tensor<i64>) -> tensor<10xf64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<10xf64>
  func.func @test_dynamic_update_slice_1d(%input: tensor<10xf64>, %update: tensor<1xf64>, %idx: tensor<i64>) -> tensor<10xf64> {
    %result = enzyme.dynamic_update_slice %input, %update, %idx : (tensor<10xf64>, tensor<1xf64>, tensor<i64>) -> tensor<10xf64>
    return %result : tensor<10xf64>
  }

  // CHECK-LABEL: @test_dynamic_update_slice_large
  // CHECK:         %[[RESULT:.+]] = stablehlo.dynamic_update_slice %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : (tensor<100x50xf64>, tensor<1x50xf64>, tensor<i64>, tensor<i64>) -> tensor<100x50xf64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<100x50xf64>
  func.func @test_dynamic_update_slice_large(%input: tensor<100x50xf64>, %update: tensor<1x50xf64>, %idx0: tensor<i64>, %idx1: tensor<i64>) -> tensor<100x50xf64> {
    %result = enzyme.dynamic_update_slice %input, %update, %idx0, %idx1 : (tensor<100x50xf64>, tensor<1x50xf64>, tensor<i64>, tensor<i64>) -> tensor<100x50xf64>
    return %result : tensor<100x50xf64>
  }
}
