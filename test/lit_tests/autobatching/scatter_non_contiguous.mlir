// RUN: enzymexlamlir-opt --enzyme-hlo-opt="enable_auto_batching_passes=true" %s | FileCheck %s

// CHECK-LABEL: func.func @scatter_non_contiguous
// CHECK-NOT: stablehlo.while
// CHECK: %[[INDICES:.*]] = stablehlo.constant dense<{{.*}}> : tensor<4x1xi32>
// CHECK: %[[RESULT:.*]] = "stablehlo.scatter"(%{{.*}}, %[[INDICES]], %arg0)
// CHECK-SAME: inserted_window_dims = [0]
// CHECK-SAME: scatter_dims_to_operand_dims = [0]
// CHECK-SAME: index_vector_dim = 1
// CHECK-SAME: unique_indices = true
// CHECK: stablehlo.return %{{.*}} : tensor<f64>
// CHECK: return %[[RESULT]] : tensor<4xf64>
func.func @scatter_non_contiguous(%arg0: tensor<4xf64>) -> tensor<4xf64> {
  %zero_i32 = stablehlo.constant dense<0> : tensor<i32>
  %zero_i64 = stablehlo.constant dense<0> : tensor<i64>
  %one_i64 = stablehlo.constant dense<1> : tensor<i64>
  %four_i64 = stablehlo.constant dense<4> : tensor<i64>
  %init = stablehlo.constant dense<0.0> : tensor<4xf64>
  %permutation = stablehlo.constant dense<[2, 0, 3, 1]> : tensor<4xi32>
  %result:2 = stablehlo.while(%iv = %zero_i64, %output = %init)
      : tensor<i64>, tensor<4xf64>
    cond {
      %pred = stablehlo.compare LT, %iv, %four_i64
          : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %pred : tensor<i1>
    } do {
      %next = stablehlo.add %iv, %one_i64 : tensor<i64>
      %source_index = stablehlo.convert %iv : (tensor<i64>) -> tensor<i32>
      %update = stablehlo.dynamic_slice %arg0, %source_index, sizes = [1]
          : (tensor<4xf64>, tensor<i32>) -> tensor<1xf64>
      %index_slice = stablehlo.dynamic_slice %permutation, %iv, sizes = [1]
          : (tensor<4xi32>, tensor<i64>) -> tensor<1xi32>
      %index = stablehlo.reshape %index_slice
          : (tensor<1xi32>) -> tensor<i32>
      %updated = stablehlo.dynamic_update_slice %output, %update, %index
          : (tensor<4xf64>, tensor<1xf64>, tensor<i32>) -> tensor<4xf64>
      stablehlo.return %next, %updated : tensor<i64>, tensor<4xf64>
    }
  return %result#1 : tensor<4xf64>
}

// Replacing sequential overwrites by an overwrite scatter is only valid when
// indices are unique. Repeated destinations must therefore retain the loop.
// CHECK-LABEL: func.func @scatter_repeated_indices
// CHECK: stablehlo.while
func.func @scatter_repeated_indices(%arg0: tensor<4xf64>) -> tensor<4xf64> {
  %zero_i64 = stablehlo.constant dense<0> : tensor<i64>
  %one_i64 = stablehlo.constant dense<1> : tensor<i64>
  %four_i64 = stablehlo.constant dense<4> : tensor<i64>
  %init = stablehlo.constant dense<0.0> : tensor<4xf64>
  %indices = stablehlo.constant dense<[2, 0, 2, 1]> : tensor<4xi32>
  %result:2 = stablehlo.while(%iv = %zero_i64, %output = %init)
      : tensor<i64>, tensor<4xf64>
    cond {
      %pred = stablehlo.compare LT, %iv, %four_i64
          : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %pred : tensor<i1>
    } do {
      %next = stablehlo.add %iv, %one_i64 : tensor<i64>
      %source_index = stablehlo.convert %iv : (tensor<i64>) -> tensor<i32>
      %update = stablehlo.dynamic_slice %arg0, %source_index, sizes = [1]
          : (tensor<4xf64>, tensor<i32>) -> tensor<1xf64>
      %index_slice = stablehlo.dynamic_slice %indices, %iv, sizes = [1]
          : (tensor<4xi32>, tensor<i64>) -> tensor<1xi32>
      %index = stablehlo.reshape %index_slice
          : (tensor<1xi32>) -> tensor<i32>
      %updated = stablehlo.dynamic_update_slice %output, %update, %index
          : (tensor<4xf64>, tensor<1xf64>, tensor<i32>) -> tensor<4xf64>
      stablehlo.return %next, %updated : tensor<i64>, tensor<4xf64>
    }
  return %result#1 : tensor<4xf64>
}
