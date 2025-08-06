// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=split_multi_result_scatter" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func private @main_2(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<3xf32>, tensor<3xf32>) {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<3xf32>
  %c = stablehlo.constant dense<[[0], [1]]> : tensor<2x1xi32>
  %0:2 = "stablehlo.scatter"(%cst, %cst, %c, %arg0, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<f32>):
    stablehlo.return %arg4, %arg5 : tensor<f32>, tensor<f32>
  }) : (tensor<3xf32>, tensor<3xf32>, tensor<2x1xi32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<3xf32>, tensor<3xf32>)
  return %0#0, %0#1 : tensor<3xf32>, tensor<3xf32>
}
