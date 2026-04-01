// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt="cse=true" | FileCheck %s

func.func @should_cse(%arg0: tensor<4xi64>, %arg1: tensor<6xi64>, %arg2: tensor<1024x1024xf32>) -> (tensor<1024x1024xf32>, tensor<1024x1024xf32>) {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<24xf32>
    %c = stablehlo.constant dense<1> : tensor<24x2xi64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<1024x1024xf32>
    %0 = stablehlo.transpose %arg2, dims = [1, 0] : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %1 = stablehlo.concatenate %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, dim = 0 : (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<24xi64>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<6xi64>) -> tensor<6x4xi64>
    %3 = stablehlo.reshape %1 : (tensor<24xi64>) -> tensor<24x1xi64>
    %4 = stablehlo.reshape %2 : (tensor<6x4xi64>) -> tensor<24x1xi64>
    %5 = stablehlo.concatenate %3, %4, dim = 1 : (tensor<24x1xi64>, tensor<24x1xi64>) -> tensor<24x2xi64>
    %6 = stablehlo.subtract %5, %c : tensor<24x2xi64>
    %7 = "stablehlo.scatter"(%cst_0, %6, %cst) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      stablehlo.return %arg4 : tensor<f32>
    }) : (tensor<1024x1024xf32>, tensor<24x2xi64>, tensor<24xf32>) -> tensor<1024x1024xf32>
    %8 = "stablehlo.scatter"(%cst_0, %6, %cst) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      stablehlo.return %arg4 : tensor<f32>
    }) : (tensor<1024x1024xf32>, tensor<24x2xi64>, tensor<24xf32>) -> tensor<1024x1024xf32>
    return %7, %8 : tensor<1024x1024xf32>, tensor<1024x1024xf32>
}

// CHECK-LABEL: func.func @should_cse
// CHECK:    "stablehlo.scatter"
// CHECK-NOT:    "stablehlo.scatter"

func.func @should_not_cse(%arg0: tensor<4xi64>, %arg1: tensor<6xi64>, %arg2: tensor<1024x1024xf32>) -> (tensor<1024x1024xf32>, tensor<1024x1024xf32>) {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<24xf32>
    %c = stablehlo.constant dense<1> : tensor<24x2xi64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<1024x1024xf32>
    %0 = stablehlo.transpose %arg2, dims = [1, 0] : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %1 = stablehlo.concatenate %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, dim = 0 : (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<24xi64>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<6xi64>) -> tensor<6x4xi64>
    %3 = stablehlo.reshape %1 : (tensor<24xi64>) -> tensor<24x1xi64>
    %4 = stablehlo.reshape %2 : (tensor<6x4xi64>) -> tensor<24x1xi64>
    %5 = stablehlo.concatenate %3, %4, dim = 1 : (tensor<24x1xi64>, tensor<24x1xi64>) -> tensor<24x2xi64>
    %6 = stablehlo.subtract %5, %c : tensor<24x2xi64>
    %7 = "stablehlo.scatter"(%cst_0, %6, %cst) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      stablehlo.return %arg4 : tensor<f32>
    }) : (tensor<1024x1024xf32>, tensor<24x2xi64>, tensor<24xf32>) -> tensor<1024x1024xf32>
    %8 = "stablehlo.scatter"(%cst_0, %6, %cst) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %9 = stablehlo.multiply %arg4, %arg3 : tensor<f32>
      stablehlo.return %9 : tensor<f32>
    }) : (tensor<1024x1024xf32>, tensor<24x2xi64>, tensor<24xf32>) -> tensor<1024x1024xf32>
    return %7, %8 : tensor<1024x1024xf32>, tensor<1024x1024xf32>
}

// CHECK-LABEL: func.func @should_not_cse
// CHECK:    "stablehlo.scatter"
// CHECK:    "stablehlo.scatter"
