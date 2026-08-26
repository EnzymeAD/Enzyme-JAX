// RUN: enzymexlamlir-opt %s --enzyme-batch | FileCheck %s

module {
  func.func private @unbatched_scatter(%arg0: tensor<8xf32>, %arg1: tensor<2x1xi32>, %arg2: tensor<2xf32>) -> tensor<8xf32> {
    %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{
      indices_are_sorted = false,
      scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [],
        inserted_window_dims = [0],
        scatter_dims_to_operand_dims = [0],
        index_vector_dim = 1
      >,
      unique_indices = false
    }> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<8xf32>, tensor<2x1xi32>, tensor<2xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }

  func.func @main(%arg0: tensor<4x8xf32>, %arg1: tensor<4x2x1xi32>, %arg2: tensor<4x2xf32>) -> tensor<4x8xf32> {
    %0 = enzyme.batch @unbatched_scatter(%arg0, %arg1, %arg2) {batch_shape = array<i64: 4>} : (tensor<4x8xf32>, tensor<4x2x1xi32>, tensor<4x2xf32>) -> tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }
}

// CHECK: func.func private @batched_unbatched_scatter(%arg0: tensor<4x8xf32>, %arg1: tensor<4x2x1xi32>, %arg2: tensor<4x2xf32>) -> tensor<4x8xf32> {
// CHECK-NEXT:   %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{
// CHECK-SAME:     indices_are_sorted = false,
// CHECK-SAME:     scatter_dimension_numbers = #stablehlo.scatter<
// CHECK-SAME:       inserted_window_dims = [1],
// CHECK-SAME:       input_batching_dims = [0],
// CHECK-SAME:       scatter_indices_batching_dims = [0],
// CHECK-SAME:       scatter_dims_to_operand_dims = [1],
// CHECK-SAME:       index_vector_dim = 2
// CHECK-SAME:     >,
// CHECK-SAME:     unique_indices = false
// CHECK-SAME:   }> ({
// CHECK-NEXT:   ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
// CHECK-NEXT:     %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
// CHECK-NEXT:     stablehlo.return %1 : tensor<f32>
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x2x1xi32>, tensor<4x2xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   return %0 : tensor<4x8xf32>
// CHECK-NEXT: }
