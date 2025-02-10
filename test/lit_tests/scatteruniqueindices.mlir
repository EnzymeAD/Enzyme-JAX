// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// CHECK-LABEL: func.func @test_scatter_duplicate
func.func @test_scatter_duplicate(%arg0: tensor<4x3xf32>, %arg2: tensor<3x3xf32>) -> tensor<4x3xf32> {
  %indices = stablehlo.constant dense<[[0], [2], [0]]> : tensor<3x1xi32>
  // CHECK: %{{.+}} = "stablehlo.scatter"(%{{.+}}, %{{.+}}, %{{.+}}) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  %0 = "stablehlo.scatter"(%arg0, %indices, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.multiply %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
  }) : (tensor<4x3xf32>, tensor<3x1xi32>, tensor<3x3xf32>) -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}

// CHECK-LABEL: func.func @test_scatter_unique
func.func @test_scatter_unique(%arg0: tensor<3x3xf32>, %arg2: tensor<2x3xf32>) -> tensor<3x3xf32> {
  %indices = stablehlo.constant dense<[[0], [1]]> : tensor<2x1xi32>
  // CHECK: %{{.+}} = "stablehlo.scatter"(%{{.+}}, %{{.+}}, %{{.+}}) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
  %0 = "stablehlo.scatter"(%arg0, %indices, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %1 = stablehlo.multiply %arg3, %arg4 : tensor<f32>
    stablehlo.return %1 : tensor<f32>
  }) : (tensor<3x3xf32>, tensor<2x1xi32>, tensor<2x3xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// CHECK-LABEL: func.func @test_scatter_single
func.func @test_scatter_single(%arg0: tensor<2x3xf32>, %arg2: tensor<1x3xf32>) -> tensor<2x3xf32> {
  %indices = stablehlo.constant dense<[[0]]> : tensor<1x1xi32>
  %update = stablehlo.constant dense<1.0e+00> : tensor<1x3xf32>
  // CHECK: %{{.+}} = "stablehlo.scatter"(%{{.+}}, %{{.+}}, %{{.+}}) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
  %0 = "stablehlo.scatter"(%arg0, %indices, %update) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      // CHECK-NOT: stablehlo.multiply
      %1 = stablehlo.multiply %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
  }) : (tensor<2x3xf32>, tensor<1x1xi32>, tensor<1x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
