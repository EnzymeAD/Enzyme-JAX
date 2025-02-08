// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main1(%arg0: tensor<64x3xf32>, %arg1: tensor<45x1xi32>, %arg2: tensor<45x3xf32>) -> (tensor<64x3xf32>, tensor<45x1xi32>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<64x3xf32>
    %c = stablehlo.constant dense<0> : tensor<45x1xi32>
    %0 = "stablehlo.scatter"(%cst, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x3xf32>, tensor<45x1xi32>, tensor<45x3xf32>) -> tensor<64x3xf32>
    return %0, %c : tensor<64x3xf32>, tensor<45x1xi32>
}

// CHECK:  func.func @main1(%arg0: tensor<64x3xf32>, %arg1: tensor<45x1xi32>, %arg2: tensor<45x3xf32>) -> (tensor<64x3xf32>, tensor<45x1xi32>) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<64x3xf32>
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<45x1xi32>
// CHECK-NEXT:    %0 = "stablehlo.scatter"(%cst, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
// CHECK-NEXT:      stablehlo.return %arg4 : tensor<f32>
// CHECK-NEXT:    }) : (tensor<64x3xf32>, tensor<45x1xi32>, tensor<45x3xf32>) -> tensor<64x3xf32>
// CHECK-NEXT:    return %0, %c : tensor<64x3xf32>, tensor<45x1xi32>
// CHECK-NEXT:  }

func.func @main2(%arg0: tensor<64x3xf32>, %arg1: tensor<45x1xi32>) -> (tensor<64x3xf32>, tensor<45x1xi32>) {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<64x3xf32>
    %c = stablehlo.constant dense<0> : tensor<45x1xi32>
    %cst_2 = stablehlo.constant dense<5.000000e+00> : tensor<45x3xf32>
    %0 = "stablehlo.scatter"(%cst, %arg1, %cst_2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.multiply %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x3xf32>, tensor<45x1xi32>, tensor<45x3xf32>) -> tensor<64x3xf32>
    return %0, %c : tensor<64x3xf32>, tensor<45x1xi32>
}

// CHECK:  func.func @main2(%arg0: tensor<64x3xf32>, %arg1: tensor<45x1xi32>) -> (tensor<64x3xf32>, tensor<45x1xi32>) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<1.000000e+01> : tensor<f32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<64x3xf32>
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<45x1xi32>
// CHECK-NEXT:    %cst_1 = stablehlo.constant dense<5.000000e+00> : tensor<45x3xf32>
// CHECK-NEXT:    %0 = "stablehlo.scatter"(%cst_0, %arg1, %cst_1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// CHECK-NEXT:      stablehlo.return %cst : tensor<f32>
// CHECK-NEXT:    }) : (tensor<64x3xf32>, tensor<45x1xi32>, tensor<45x3xf32>) -> tensor<64x3xf32>
// CHECK-NEXT:    return %0, %c : tensor<64x3xf32>, tensor<45x1xi32>
// CHECK-NEXT:  }

func.func @main3(%arg0: tensor<64x3xf32>, %arg1: tensor<45x1xi32>) -> (tensor<64x3xf32>, tensor<45x1xi32>) {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<64x3xf32>
    %c = stablehlo.constant dense<0> : tensor<45x1xi32>
    %cst_2 = stablehlo.constant dense<5.000000e+00> : tensor<45x3xf32>
    %0 = "stablehlo.scatter"(%cst, %arg1, %cst_2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.multiply %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x3xf32>, tensor<45x1xi32>, tensor<45x3xf32>) -> tensor<64x3xf32>
    return %0, %c : tensor<64x3xf32>, tensor<45x1xi32>
}

// CHECK:  func.func @main3(%arg0: tensor<64x3xf32>, %arg1: tensor<45x1xi32>) -> (tensor<64x3xf32>, tensor<45x1xi32>) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<2.000000e+00> : tensor<64x3xf32>
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<45x1xi32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<5.000000e+00> : tensor<45x3xf32>
// CHECK-NEXT:    %0 = "stablehlo.scatter"(%cst, %arg1, %cst_0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// CHECK-NEXT:      %1 = stablehlo.multiply %arg2, %arg3 : tensor<f32>
// CHECK-NEXT:      stablehlo.return %1 : tensor<f32>
// CHECK-NEXT:    }) : (tensor<64x3xf32>, tensor<45x1xi32>, tensor<45x3xf32>) -> tensor<64x3xf32>
// CHECK-NEXT:    return %0, %c : tensor<64x3xf32>, tensor<45x1xi32>
// CHECK-NEXT:  }
