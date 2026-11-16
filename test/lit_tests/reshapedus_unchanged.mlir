// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=reshape_dus" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module {
  // Test that a failing match leaves the IR unchanged. The reshape adds a unit
  // dimension in front of a dropped one, whose start is not a constant zero.

  func.func @reshape_dus_add_unit_dim_before_dynamic_start(%arg0: tensor<2x1xf64>, %arg1: tensor<1x1xf64>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<1x2xf64> {
    %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3 : (tensor<2x1xf64>, tensor<1x1xf64>, tensor<i32>, tensor<i32>) -> tensor<2x1xf64>
    %1 = stablehlo.reshape %0 : (tensor<2x1xf64>) -> tensor<1x2xf64>
    return %1 : tensor<1x2xf64>
  }

  // CHECK: func.func @reshape_dus_add_unit_dim_before_dynamic_start(%arg0: tensor<2x1xf64>, %arg1: tensor<1x1xf64>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<1x2xf64> {
  // CHECK-NEXT:   %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3 : (tensor<2x1xf64>, tensor<1x1xf64>, tensor<i32>, tensor<i32>) -> tensor<2x1xf64>
  // CHECK-NEXT:   %1 = stablehlo.reshape %0 : (tensor<2x1xf64>) -> tensor<1x2xf64>
  // CHECK-NEXT:   return %1 : tensor<1x2xf64>
  // CHECK-NEXT: }
}
