// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @test_transpose_of_dynamic_update_slice(%arg0: tensor<4x6xf32>, 
                                                  %arg1: tensor<2x3xf32>, 
                                                  %arg2: tensor<i32>, 
                                                  %arg3: tensor<i32>) -> tensor<6x4xf32> {
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3 : (tensor<4x6xf32>, tensor<2x3xf32>, tensor<i32>, tensor<i32>) -> tensor<4x6xf32>
  %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<4x6xf32>) -> tensor<6x4xf32>
  return %1 : tensor<6x4xf32>
}

// CHECK: func.func @test_transpose_of_dynamic_update_slice(%arg0: tensor<4x6xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<6x4xf32> {
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x6xf32>) -> tensor<6x4xf32>
// CHECK-NEXT:   %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
// CHECK-NEXT:   %2 = stablehlo.dynamic_update_slice %0, %1, %arg3, %arg2 : (tensor<6x4xf32>, tensor<3x2xf32>, tensor<i32>, tensor<i32>) -> tensor<6x4xf32>
// CHECK-NEXT:   return %2 : tensor<6x4xf32>
// CHECK-NEXT: }

func.func @test_transpose_of_dynamic_update_slice_3d(%arg0: tensor<4x5x6xf32>, 
                                                     %arg1: tensor<2x3x4xf32>, 
                                                     %arg2: tensor<i32>,
                                                     %arg3: tensor<i32>,
                                                     %arg4: tensor<i32>) -> tensor<6x4x5xf32> {
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3, %arg4 : (tensor<4x5x6xf32>, tensor<2x3x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x5x6xf32>
  %1 = stablehlo.transpose %0, dims = [2, 0, 1] : (tensor<4x5x6xf32>) -> tensor<6x4x5xf32>
  return %1 : tensor<6x4x5xf32>
}

// CHECK: func.func @test_transpose_of_dynamic_update_slice_3d(%arg0: tensor<4x5x6xf32>, %arg1: tensor<2x3x4xf32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>) -> tensor<6x4x5xf32> {
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [2, 0, 1] : (tensor<4x5x6xf32>) -> tensor<6x4x5xf32>
// CHECK-NEXT:   %1 = stablehlo.transpose %arg1, dims = [2, 0, 1] : (tensor<2x3x4xf32>) -> tensor<4x2x3xf32>
// CHECK-NEXT:   %2 = stablehlo.dynamic_update_slice %0, %1, %arg3, %arg4, %arg2 : (tensor<6x4x5xf32>, tensor<4x2x3xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x4x5xf32>
// CHECK-NEXT:   return %2 : tensor<6x4x5xf32>
// CHECK-NEXT: }
