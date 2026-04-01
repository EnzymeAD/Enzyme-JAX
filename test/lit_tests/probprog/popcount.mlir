// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-probprog-to-stablehlo{backend=cpu})" | FileCheck %s

module {
  // CHECK-LABEL: @test_popcount_scalar_i64
  // CHECK:         %[[RESULT:.+]] = stablehlo.popcnt %{{.+}} : tensor<i64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<i64>
  func.func @test_popcount_scalar_i64(%x: tensor<i64>) -> tensor<i64> {
    %result = enzyme.popcount %x : (tensor<i64>) -> tensor<i64>
    return %result : tensor<i64>
  }

  // CHECK-LABEL: @test_popcount_scalar_i32
  // CHECK:         %[[RESULT:.+]] = stablehlo.popcnt %{{.+}} : tensor<i32>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<i32>
  func.func @test_popcount_scalar_i32(%x: tensor<i32>) -> tensor<i32> {
    %result = enzyme.popcount %x : (tensor<i32>) -> tensor<i32>
    return %result : tensor<i32>
  }

  // CHECK-LABEL: @test_popcount_1d
  // CHECK:         %[[RESULT:.+]] = stablehlo.popcnt %{{.+}} : tensor<8xi32>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<8xi32>
  func.func @test_popcount_1d(%x: tensor<8xi32>) -> tensor<8xi32> {
    %result = enzyme.popcount %x : (tensor<8xi32>) -> tensor<8xi32>
    return %result : tensor<8xi32>
  }

  // CHECK-LABEL: @test_popcount_2d
  // CHECK:         %[[RESULT:.+]] = stablehlo.popcnt %{{.+}} : tensor<4x8xi64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<4x8xi64>
  func.func @test_popcount_2d(%x: tensor<4x8xi64>) -> tensor<4x8xi64> {
    %result = enzyme.popcount %x : (tensor<4x8xi64>) -> tensor<4x8xi64>
    return %result : tensor<4x8xi64>
  }
}
