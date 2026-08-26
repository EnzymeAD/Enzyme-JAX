// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @test_scatter_single_index(%arg0: tensor<4xf32>, %arg1: tensor<1x1xf32>, %idx : tensor<1x1x1xi32>) -> tensor<4xf32> {
  %0 = "stablehlo.scatter"(%arg0, %idx, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.subtract %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
  }) : (tensor<4xf32>, tensor<1x1x1xi32>, tensor<1x1xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK: func.func @test_scatter_single_index(%arg0: tensor<4xf32>, %arg1: tensor<1x1xf32>, %arg2: tensor<1x1x1xi32>) -> tensor<4xf32> {
// CHECK-NEXT:   %0 = stablehlo.reshape %arg2 : (tensor<1x1x1xi32>) -> tensor<i32>
// CHECK-NEXT:   %1 = stablehlo.dynamic_slice %arg0, %0, sizes = [1] : (tensor<4xf32>, tensor<i32>) -> tensor<1xf32>
// CHECK-NEXT:   %2 = stablehlo.reshape %arg1 : (tensor<1x1xf32>) -> tensor<1xf32>
// CHECK-NEXT:   %3 = stablehlo.subtract %1, %2 : tensor<1xf32>
// CHECK-NEXT:   %4 = stablehlo.dynamic_update_slice %arg0, %3, %0 : (tensor<4xf32>, tensor<1xf32>, tensor<i32>) -> tensor<4xf32>
// CHECK-NEXT:   return %4 : tensor<4xf32>
// CHECK-NEXT: }

func.func @test_scatter_single_index_outside_value(%arg0: tensor<4xf32>, %arg1: tensor<1x1xf32>, %idx : tensor<1x1x1xi32>, %out_val: tensor<f32>) -> tensor<4xf32> {
  %0 = "stablehlo.scatter"(%arg0, %idx, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.subtract %arg3, %out_val : tensor<f32>
      stablehlo.return %1 : tensor<f32>
  }) : (tensor<4xf32>, tensor<1x1x1xi32>, tensor<1x1xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK: func.func @test_scatter_single_index_outside_value(%arg0: tensor<4xf32>, %arg1: tensor<1x1xf32>, %arg2: tensor<1x1x1xi32>, %arg3: tensor<f32>) -> tensor<4xf32> {
// CHECK-NEXT:   %0 = stablehlo.reshape %arg2 : (tensor<1x1x1xi32>) -> tensor<i32>
// CHECK-NEXT:   %1 = stablehlo.dynamic_slice %arg0, %0, sizes = [1] : (tensor<4xf32>, tensor<i32>) -> tensor<1xf32>
// CHECK-NEXT:   %2 = stablehlo.reshape %arg3 : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:   %3 = stablehlo.subtract %1, %2 : tensor<1xf32>
// CHECK-NEXT:   %4 = stablehlo.dynamic_update_slice %arg0, %3, %0 : (tensor<4xf32>, tensor<1xf32>, tensor<i32>) -> tensor<4xf32>
// CHECK-NEXT:   return %4 : tensor<4xf32>
// CHECK-NEXT: }

func.func @test_scatter_single_index_outside_value2(%arg0: tensor<4xf32>, %arg1: tensor<1x1xf32>, %idx : tensor<1x1x1xi32>, %out_val: tensor<f32>) -> tensor<4xf32> {
  %cst = stablehlo.constant dense<5.0> : tensor<f32>
  %out_val2 = stablehlo.add %out_val, %cst : tensor<f32>
  %0 = "stablehlo.scatter"(%arg0, %idx, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.subtract %arg3, %out_val2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
  }) : (tensor<4xf32>, tensor<1x1x1xi32>, tensor<1x1xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK: func.func @test_scatter_single_index_outside_value2(%arg0: tensor<4xf32>, %arg1: tensor<1x1xf32>, %arg2: tensor<1x1x1xi32>, %arg3: tensor<f32>) -> tensor<4xf32> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<5.000000e+00> : tensor<f32>
// CHECK-NEXT:   %0 = stablehlo.add %arg3, %cst : tensor<f32>
// CHECK-NEXT:   %1 = stablehlo.reshape %arg2 : (tensor<1x1x1xi32>) -> tensor<i32>
// CHECK-NEXT:   %2 = stablehlo.dynamic_slice %arg0, %1, sizes = [1] : (tensor<4xf32>, tensor<i32>) -> tensor<1xf32>
// CHECK-NEXT:   %3 = stablehlo.reshape %0 : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:   %4 = stablehlo.subtract %2, %3 : tensor<1xf32>
// CHECK-NEXT:   %5 = stablehlo.dynamic_update_slice %arg0, %4, %1 : (tensor<4xf32>, tensor<1xf32>, tensor<i32>) -> tensor<4xf32>
// CHECK-NEXT:   return %5 : tensor<4xf32>
// CHECK-NEXT: }

func.func @test_scatter_single_index_const_outside_value(%arg0: tensor<4xf32>, %arg1: tensor<1x1xf32>, %idx : tensor<1x1x1xi32>) -> tensor<4xf32> {
  %cst = stablehlo.constant dense<5.0> : tensor<f32>
  %0 = "stablehlo.scatter"(%arg0, %idx, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.subtract %arg3, %cst : tensor<f32>
      stablehlo.return %1 : tensor<f32>
  }) : (tensor<4xf32>, tensor<1x1x1xi32>, tensor<1x1xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK: func.func @test_scatter_single_index_const_outside_value(%arg0: tensor<4xf32>, %arg1: tensor<1x1xf32>, %arg2: tensor<1x1x1xi32>) -> tensor<4xf32> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<5.000000e+00> : tensor<1xf32>
// CHECK-NEXT:   %0 = stablehlo.reshape %arg2 : (tensor<1x1x1xi32>) -> tensor<i32>
// CHECK-NEXT:   %1 = stablehlo.dynamic_slice %arg0, %0, sizes = [1] : (tensor<4xf32>, tensor<i32>) -> tensor<1xf32>
// CHECK-NEXT:   %2 = stablehlo.subtract %1, %cst : tensor<1xf32>
// CHECK-NEXT:   %3 = stablehlo.dynamic_update_slice %arg0, %2, %0 : (tensor<4xf32>, tensor<1xf32>, tensor<i32>) -> tensor<4xf32>
// CHECK-NEXT:   return %3 : tensor<4xf32>
// CHECK-NEXT: }

// Test scatter with iota indices starting from 0
func.func @test_scatter_iota_indices(%arg0: tensor<8xf32>, %arg1: tensor<4xf32>) -> tensor<8xf32> {
  %indices = stablehlo.constant dense<[[0], [1], [2], [3]]> : tensor<4x1xi32>
  %0 = "stablehlo.scatter"(%arg0, %indices, %arg1) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
  }) : (tensor<8xf32>, tensor<4x1xi32>, tensor<4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func.func @test_scatter_iota_indices
// CHECK:    %0 = stablehlo.slice %arg0 [0:4] : (tensor<8xf32>) -> tensor<4xf32>
// CHECK:    %1 = stablehlo.add %0, %arg1 : tensor<4xf32>
// CHECK:    %2 = stablehlo.slice %arg0 [4:8] : (tensor<8xf32>) -> tensor<4xf32>
// CHECK:    %3 = stablehlo.concatenate %1, %2, dim = 0 : (tensor<4xf32>, tensor<4xf32>) -> tensor<8xf32>
// CHECK:    return %3 : tensor<8xf32>

// Test scatter with iota indices starting from offset
func.func @test_scatter_iota_offset(%arg0: tensor<8xf32>, %arg1: tensor<3xf32>) -> tensor<8xf32> {
  %indices = stablehlo.constant dense<[[2], [3], [4]]> : tensor<3x1xi32>
  %0 = "stablehlo.scatter"(%arg0, %indices, %arg1) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.multiply %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
  }) : (tensor<8xf32>, tensor<3x1xi32>, tensor<3xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func.func @test_scatter_iota_offset
// CHECK:    %0 = stablehlo.slice %arg0 [2:5] : (tensor<8xf32>) -> tensor<3xf32>
// CHECK:    %1 = stablehlo.multiply %0, %arg1 : tensor<3xf32>
// CHECK:    %2 = stablehlo.slice %arg0 [0:2] : (tensor<8xf32>) -> tensor<2xf32>
// CHECK:    %3 = stablehlo.slice %arg0 [5:8] : (tensor<8xf32>) -> tensor<3xf32>
// CHECK:    %4 = stablehlo.concatenate %2, %1, %3, dim = 0 : (tensor<2xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<8xf32>
// CHECK:    return %4 : tensor<8xf32>

// Test scatter with negative stride (reverse order indices)
func.func @test_scatter_iota_reverse(%arg0: tensor<8xf32>, %arg1: tensor<4xf32>) -> tensor<8xf32> {
  %indices = stablehlo.constant dense<[[3], [2], [1], [0]]> : tensor<4x1xi32>
  %0 = "stablehlo.scatter"(%arg0, %indices, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.subtract %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
  }) : (tensor<8xf32>, tensor<4x1xi32>, tensor<4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func.func @test_scatter_iota_reverse
// CHECK:    %0 = stablehlo.slice %arg0 [0:4] : (tensor<8xf32>) -> tensor<4xf32>
// CHECK:    %1 = stablehlo.reverse %arg1, dims = [0] : tensor<4xf32>
// CHECK:    %2 = stablehlo.subtract %0, %1 : tensor<4xf32>
// CHECK:    %3 = stablehlo.slice %arg0 [4:8] : (tensor<8xf32>) -> tensor<4xf32>
// CHECK:    %4 = stablehlo.concatenate %2, %3, dim = 0 : (tensor<4xf32>, tensor<4xf32>) -> tensor<8xf32>
// CHECK:    return %4 : tensor<8xf32>

// Test scatter with iota indices and external value captured in region
func.func @test_scatter_iota_captured_value(%arg0: tensor<8xf32>, %arg1: tensor<4xf32>, %scale: tensor<f32>) -> tensor<8xf32> {
  %indices = stablehlo.constant dense<[[0], [1], [2], [3]]> : tensor<4x1xi32>
  %0 = "stablehlo.scatter"(%arg0, %indices, %arg1) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.multiply %arg3, %scale : tensor<f32>
      %2 = stablehlo.add %arg2, %1 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
  }) : (tensor<8xf32>, tensor<4x1xi32>, tensor<4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func.func @test_scatter_iota_captured_value
// CHECK:    %0 = stablehlo.slice %arg0 [0:4] : (tensor<8xf32>) -> tensor<4xf32>
// CHECK:    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<4xf32>
// CHECK:    %2 = stablehlo.multiply %arg1, %1 : tensor<4xf32>
// CHECK:    %3 = stablehlo.add %0, %2 : tensor<4xf32>
// CHECK:    %4 = stablehlo.slice %arg0 [4:8] : (tensor<8xf32>) -> tensor<4xf32>
// CHECK:    %5 = stablehlo.concatenate %3, %4, dim = 0 : (tensor<4xf32>, tensor<4xf32>) -> tensor<8xf32>
// CHECK:    return %5 : tensor<8xf32>
