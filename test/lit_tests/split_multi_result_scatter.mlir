// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=split_multi_result_scatter" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// CHECK:      func.func private @can_rewrite(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<3xf32>, tensor<3xf32>) {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<3xf32>
// CHECK-NEXT:   %c = stablehlo.constant dense<{{\[}}{{\[}}0], [1]]> : tensor<2x1xi32>
// CHECK-NEXT:   %0 = "stablehlo.scatter"(%cst, %c, %arg0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:   ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// CHECK-NEXT:     %2 = stablehlo.abs %arg3 : tensor<f32>
// CHECK-NEXT:     stablehlo.return %2 : tensor<f32>
// CHECK-NEXT:   }) : (tensor<3xf32>, tensor<2x1xi32>, tensor<2xf32>) -> tensor<3xf32>
// CHECK-NEXT:   %1 = "stablehlo.scatter"(%cst, %c, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:   ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// CHECK-NEXT:     stablehlo.return %arg3 : tensor<f32>
// CHECK-NEXT:   }) : (tensor<3xf32>, tensor<2x1xi32>, tensor<2xf32>) -> tensor<3xf32>
// CHECK-NEXT:   return %0, %1 : tensor<3xf32>, tensor<3xf32>
// CHECK-NEXT: }
func.func private @can_rewrite(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<3xf32>, tensor<3xf32>) {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<3xf32>
  %c = stablehlo.constant dense<[[0], [1]]> : tensor<2x1xi32>
  %0:2 = "stablehlo.scatter"(%cst, %cst, %c, %arg0, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<f32>):
    %a = "stablehlo.abs"(%arg4) : (tensor<f32>) -> tensor<f32>
    stablehlo.return %a, %arg5 : tensor<f32>, tensor<f32>
  }) : (tensor<3xf32>, tensor<3xf32>, tensor<2x1xi32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<3xf32>, tensor<3xf32>)
  return %0#0, %0#1 : tensor<3xf32>, tensor<3xf32>
}

// CHECK:      func.func private @cannot_rewrite(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<3xf32>, tensor<3xf32>) {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<3xf32>
// CHECK-NEXT:   %c = stablehlo.constant dense<{{\[}}{{\[}}0], [1]]> : tensor<2x1xi32>
// CHECK-NEXT:   %0:2 = "stablehlo.scatter"(%cst, %cst, %c, %arg0, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:   ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<f32>):
// CHECK-NEXT:     %1 = stablehlo.add %arg4, %arg3 : tensor<f32>
// CHECK-NEXT:     stablehlo.return %1, %arg5 : tensor<f32>, tensor<f32>
// CHECK-NEXT:   }) : (tensor<3xf32>, tensor<3xf32>, tensor<2x1xi32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<3xf32>, tensor<3xf32>)
// CHECK-NEXT:   return %0#0, %0#1 : tensor<3xf32>, tensor<3xf32>
// CHECK-NEXT: }
func.func private @cannot_rewrite(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<3xf32>, tensor<3xf32>) {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<3xf32>
  %c = stablehlo.constant dense<[[0], [1]]> : tensor<2x1xi32>
  %0:2 = "stablehlo.scatter"(%cst, %cst, %c, %arg0, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<f32>):
    %a = "stablehlo.add"(%arg4, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32> // result depends on multiple operands: can't be transformed!
    stablehlo.return %a, %arg5 : tensor<f32>, tensor<f32>
  }) : (tensor<3xf32>, tensor<3xf32>, tensor<2x1xi32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<3xf32>, tensor<3xf32>)
  return %0#0, %0#1 : tensor<3xf32>, tensor<3xf32>
}
