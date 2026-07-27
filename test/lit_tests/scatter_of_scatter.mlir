// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @fuse_inner_const_setindex_outer_add
  func.func @fuse_inner_const_setindex_outer_add(%arg0: tensor<3000x3000xf64>) -> tensor<3000x3000xf64> {
    %cst = stablehlo.constant dense<1.000000e-01> : tensor<f64>
    %cst_0 = stablehlo.constant dense<2.000000e-01> : tensor<3000xf64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<3000x3000xf64>
    %0 = stablehlo.iota dim = 0 : tensor<3000x2xi64>

    // Inner: ConstantSetindex(0.1)
    %1 = "stablehlo.scatter"(%cst_1, %0, %cst_0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
      stablehlo.return %cst : tensor<f64>
    }) : (tensor<3000x3000xf64>, tensor<3000x2xi64>, tensor<3000xf64>) -> tensor<3000x3000xf64>

    // Outer: Add(arg3, arg4) -> Add(0.1, arg4)
    // Result folds to 0.3 since update is constant 0.2
    // CHECK: %[[CST_03:.*]] = stablehlo.constant dense<0.30000000000000004> : tensor<f64>
    // CHECK: %[[RES:.*]] = "stablehlo.scatter"
    // CHECK: ^bb0(%[[ARG1:.*]]: tensor<f64>, %[[ARG2:.*]]: tensor<f64>):
    // CHECK-NEXT: stablehlo.return %[[CST_03]]
    %2 = "stablehlo.scatter"(%1, %0, %cst_0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
      %5 = stablehlo.add %arg3, %arg4 : tensor<f64>
      stablehlo.return %5 : tensor<f64>
    }) : (tensor<3000x3000xf64>, tensor<3000x2xi64>, tensor<3000xf64>) -> tensor<3000x3000xf64>

    return %2 : tensor<3000x3000xf64>
  }

  // CHECK-LABEL: func.func @outer_ignores_arg0
  func.func @outer_ignores_arg0(%arg0: tensor<3000x3000xf64>) -> tensor<3000x3000xf64> {
    %cst = stablehlo.constant dense<1.000000e-01> : tensor<f64>
    %cst_0 = stablehlo.constant dense<2.000000e-01> : tensor<3000xf64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<3000x3000xf64>
    %0 = stablehlo.iota dim = 0 : tensor<3000x2xi64>

    // Inner: Add(arg3, arg4)
    %1 = "stablehlo.scatter"(%cst_1, %0, %cst_0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
      %add = stablehlo.add %arg3, %arg4 : tensor<f64>
      stablehlo.return %add : tensor<f64>
    }) : (tensor<3000x3000xf64>, tensor<3000x2xi64>, tensor<3000xf64>) -> tensor<3000x3000xf64>

    // Outer: Setindex (returns arg4, ignores arg3)
    // CHECK: %[[CST_2_0:.*]] = stablehlo.constant dense<2.000000e-01> : tensor<f64>
    // CHECK: %[[RES:.*]] = "stablehlo.scatter"
    // CHECK: ^bb0(%[[ARG1:.*]]: tensor<f64>, %[[ARG2:.*]]: tensor<f64>):
    // CHECK-NEXT: stablehlo.return %[[CST_2_0]]
    %2 = "stablehlo.scatter"(%1, %0, %cst_0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
      stablehlo.return %arg4 : tensor<f64>
    }) : (tensor<3000x3000xf64>, tensor<3000x2xi64>, tensor<3000xf64>) -> tensor<3000x3000xf64>

    return %2 : tensor<3000x3000xf64>
  }

  // CHECK-LABEL: func.func @inner_const_setindex_outer_sub
  func.func @inner_const_setindex_outer_sub(%arg0: tensor<3000x3000xf64>) -> tensor<3000x3000xf64> {
    %cst = stablehlo.constant dense<1.000000e-01> : tensor<f64>
    %cst_0 = stablehlo.constant dense<2.000000e-01> : tensor<3000xf64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<3000x3000xf64>
    %0 = stablehlo.iota dim = 0 : tensor<3000x2xi64>

    // Inner: ConstantSetindex(0.1)
    %1 = "stablehlo.scatter"(%cst_1, %0, %cst_0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
      stablehlo.return %cst : tensor<f64>
    }) : (tensor<3000x3000xf64>, tensor<3000x2xi64>, tensor<3000xf64>) -> tensor<3000x3000xf64>

    // Outer: Subtract(arg3, arg4) -> Sub(0.1, 0.2) = -0.1
    // CHECK: %[[CST_NEG_01:.*]] = stablehlo.constant dense<-1.000000e-01> : tensor<f64>
    // CHECK: %[[RES:.*]] = "stablehlo.scatter"
    // CHECK: ^bb0(%[[ARG1:.*]]: tensor<f64>, %[[ARG2:.*]]: tensor<f64>):
    // CHECK-NEXT: stablehlo.return %[[CST_NEG_01]]
    %2 = "stablehlo.scatter"(%1, %0, %cst_0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
      %5 = stablehlo.subtract %arg3, %arg4 : tensor<f64>
      stablehlo.return %5 : tensor<f64>
    }) : (tensor<3000x3000xf64>, tensor<3000x2xi64>, tensor<3000xf64>) -> tensor<3000x3000xf64>

    return %2 : tensor<3000x3000xf64>
  }

  // CHECK-LABEL: func.func @no_fuse_non_unique
  func.func @no_fuse_non_unique(%arg0: tensor<3000x3000xf64>, %indices: tensor<3000x2xi64>) -> tensor<3000x3000xf64> {
    %cst = stablehlo.constant dense<1.000000e-01> : tensor<f64>
    %cst_0 = stablehlo.constant dense<2.000000e-01> : tensor<3000xf64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<3000x3000xf64>

    // Inner: ConstantSetindex(0.1), unique_indices = false
    // Since %indices is an argument, the compiler cannot prove it's unique.
    // CHECK: %[[INNER:.*]] = "stablehlo.scatter"{{.*}}unique_indices = false
    %1 = "stablehlo.scatter"(%cst_1, %indices, %cst_0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
      stablehlo.return %cst : tensor<f64>
    }) : (tensor<3000x3000xf64>, tensor<3000x2xi64>, tensor<3000xf64>) -> tensor<3000x3000xf64>

    // Outer: Add(arg3, arg4), unique_indices = true
    // CHECK: %[[OUTER:.*]] = "stablehlo.scatter"(%[[INNER]]
    %2 = "stablehlo.scatter"(%1, %indices, %cst_0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
      %5 = stablehlo.add %arg3, %arg4 : tensor<f64>
      stablehlo.return %5 : tensor<f64>
    }) : (tensor<3000x3000xf64>, tensor<3000x2xi64>, tensor<3000xf64>) -> tensor<3000x3000xf64>

    return %2 : tensor<3000x3000xf64>
  }

  // CHECK-LABEL: func.func @outer_const_setindex
  func.func @outer_const_setindex(%arg0: tensor<3000x3000xf64>) -> tensor<3000x3000xf64> {
    %cst = stablehlo.constant dense<1.000000e-01> : tensor<f64>
    %cst_0 = stablehlo.constant dense<2.000000e-01> : tensor<3000xf64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<3000x3000xf64>
    %cst_2 = stablehlo.constant dense<2.000000e-01> : tensor<f64>
    %0 = stablehlo.iota dim = 0 : tensor<3000x2xi64>

    // Inner: Add(arg3, arg4)
    %1 = "stablehlo.scatter"(%cst_1, %0, %cst_0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
      %add = stablehlo.add %arg3, %arg4 : tensor<f64>
      stablehlo.return %add : tensor<f64>
    }) : (tensor<3000x3000xf64>, tensor<3000x2xi64>, tensor<3000xf64>) -> tensor<3000x3000xf64>

    // Outer: ConstantSetindex(0.2)
    // CHECK: %[[CST_02:.*]] = stablehlo.constant dense<2.000000e-01> : tensor<f64>
    // CHECK: %[[RES:.*]] = "stablehlo.scatter"
    // CHECK: ^bb0(%[[ARG1:.*]]: tensor<f64>, %[[ARG2:.*]]: tensor<f64>):
    // CHECK-NEXT: stablehlo.return %[[CST_02]]
    %2 = "stablehlo.scatter"(%1, %0, %cst_0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
      stablehlo.return %cst_2 : tensor<f64>
    }) : (tensor<3000x3000xf64>, tensor<3000x2xi64>, tensor<3000xf64>) -> tensor<3000x3000xf64>

    return %2 : tensor<3000x3000xf64>
  }

  // CHECK-LABEL: func.func @inner_const_setindex_outer_mul
  func.func @inner_const_setindex_outer_mul(%arg0: tensor<3000x3000xf64>) -> tensor<3000x3000xf64> {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<3000xf64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<3000x3000xf64>
    %0 = stablehlo.iota dim = 0 : tensor<3000x2xi64>

    // Inner: ConstantSetindex(2.0)
    %1 = "stablehlo.scatter"(%cst_1, %0, %cst_0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
      stablehlo.return %cst : tensor<f64>
    }) : (tensor<3000x3000xf64>, tensor<3000x2xi64>, tensor<3000xf64>) -> tensor<3000x3000xf64>

    // Outer: Multiply(arg3, arg4) -> Mul(2.0, 3.0) = 6.0
    // CHECK: %[[CST_6:.*]] = stablehlo.constant dense<6.000000e+00> : tensor<f64>
    // CHECK: %[[RES:.*]] = "stablehlo.scatter"
    // CHECK: ^bb0(%[[ARG1:.*]]: tensor<f64>, %[[ARG2:.*]]: tensor<f64>):
    // CHECK-NEXT: stablehlo.return %[[CST_6]]
    %2 = "stablehlo.scatter"(%1, %0, %cst_0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
      %5 = stablehlo.multiply %arg3, %arg4 : tensor<f64>
      stablehlo.return %5 : tensor<f64>
    }) : (tensor<3000x3000xf64>, tensor<3000x2xi64>, tensor<3000xf64>) -> tensor<3000x3000xf64>

    return %2 : tensor<3000x3000xf64>
  }
}
