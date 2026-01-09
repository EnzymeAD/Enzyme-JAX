func.func @test_scatter_single_index(%arg0: tensor<4xf32>, %arg1: tensor<1x1xf32>, %idx : tensor<1x1x1xi32>) -> tensor<4xf32> {
  %0 = "stablehlo.scatter"(%arg0, %idx, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.subtract %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
  }) : (tensor<4xf32>, tensor<1x1x1xi32>, tensor<1x1xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

func.func @test_scatter_single_index_outside_value(%arg0: tensor<4xf32>, %arg1: tensor<1x1xf32>, %idx : tensor<1x1x1xi32>, %out_val: tensor<f32>) -> tensor<4xf32> {
  %0 = "stablehlo.scatter"(%arg0, %idx, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.subtract %arg3, %out_val : tensor<f32>
      stablehlo.return %1 : tensor<f32>
  }) : (tensor<4xf32>, tensor<1x1x1xi32>, tensor<1x1xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

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

func.func @test_scatter_single_index_const_outside_value(%arg0: tensor<4xf32>, %arg1: tensor<1x1xf32>, %idx : tensor<1x1x1xi32>) -> tensor<4xf32> {
  %cst = stablehlo.constant dense<5.0> : tensor<f32>
  %0 = "stablehlo.scatter"(%arg0, %idx, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.subtract %arg3, %cst : tensor<f32>
      stablehlo.return %1 : tensor<f32>
  }) : (tensor<4xf32>, tensor<1x1x1xi32>, tensor<1x1xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
