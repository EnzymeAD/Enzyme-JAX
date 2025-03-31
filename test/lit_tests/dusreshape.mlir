// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-opt{passses=131072})" | FileCheck %s

module {
  func.func @test_reshape_of_dynamic_update_slice(%arg0: tensor<1x4x6xf32>,
                                                    %arg1: tensor<1x2x3xf32>,
                                                    %arg2: tensor<i32>,
                                                    %arg3: tensor<i32>) -> tensor<4x1x6xf32> {
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %c0, %arg2, %arg3 : (tensor<1x4x6xf32>, tensor<1x2x3xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x4x6xf32>
    %1 = stablehlo.reshape %0 : (tensor<1x4x6xf32>) -> tensor<4x1x6xf32>
    return %1 : tensor<4x1x6xf32>
  }

// CHECK:  func.func @test_reshape_of_dynamic_update_slice(%arg0: tensor<1x4x6xf32>, %arg1: tensor<1x2x3xf32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<4x1x6xf32> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<1x4x6xf32>) -> tensor<4x1x6xf32>
// CHECK-NEXT:    %1 = stablehlo.reshape %arg1 : (tensor<1x2x3xf32>) -> tensor<2x1x3xf32>
// CHECK-NEXT:    %2 = stablehlo.dynamic_update_slice %0, %1, %arg2, %c, %arg3 : (tensor<4x1x6xf32>, tensor<2x1x3xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x1x6xf32>
// CHECK-NEXT:    return %2 : tensor<4x1x6xf32>
// CHECK-NEXT:  }

}
