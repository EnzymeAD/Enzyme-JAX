// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --split-input-file | FileCheck %s

module {
  func.func @main(%prev : tensor<8000x3xf32>, %idxs : tensor<460000x1xi32>, %update : tensor<460000x3xf32>) -> tensor<8000x3xf32> {
    %0 = "stablehlo.scatter"(%prev, %idxs, %update) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<8000x3xf32>, tensor<460000x1xi32>, tensor<460000x3xf32>) -> tensor<8000x3xf32>
    return %0 : tensor<8000x3xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<8000x3xf32>, %arg1: tensor<460000x1xi32>, %arg2: tensor<460000x3xf32>, %arg3: tensor<8000x3xf32>) -> (tensor<8000x3xf32>, tensor<460000x1xi32>, tensor<460000x3xf32>) {
// CHECK-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<8000x3xf32>
// CHECK-NEXT:    %cst_0 = arith.constant dense<0.000000e+00> : tensor<460000x3xf32>
// CHECK-NEXT:    %cst_1 = arith.constant dense<0> : tensor<460000x1xi32>
// CHECK-NEXT:    %0 = arith.addf %arg3, %cst : tensor<8000x3xf32>
// CHECK-NEXT:    %1 = arith.addf %0, %cst : tensor<8000x3xf32>
// CHECK-NEXT:    %2 = "stablehlo.gather"(%0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 3>}> : (tensor<8000x3xf32>, tensor<460000x1xi32>) -> tensor<460000x3xf32>
// CHECK-NEXT:    %3 = arith.addf %2, %cst_0 : tensor<460000x3xf32>
// CHECK-NEXT:    return %1, %cst_1, %3 : tensor<8000x3xf32>, tensor<460000x1xi32>, tensor<460000x3xf32>
// CHECK-NEXT:  }

// -----

module {
  func.func @main(%prev : tensor<8000x3xf32>, %idxs : tensor<4600x1xi32>, %update : tensor<4600x3xf32>) -> tensor<8000x3xf32> {
    %0 = "stablehlo.scatter"(%prev, %idxs, %update) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.multiply %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<8000x3xf32>, tensor<4600x1xi32>, tensor<4600x3xf32>) -> tensor<8000x3xf32>
    return %0 : tensor<8000x3xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<8000x3xf32>, %arg1: tensor<4600x1xi32>, %arg2: tensor<4600x3xf32>, %arg3: tensor<8000x3xf32>) -> (tensor<8000x3xf32>, tensor<4600x1xi32>, tensor<4600x3xf32>) {
// CHECK-NEXT:   %cst = arith.constant dense<0.000000e+00> : tensor<8000x3xf32>
// CHECK-NEXT:   %cst_0 = arith.constant dense<0.000000e+00> : tensor<4600x3xf32>
// CHECK-NEXT:   %cst_1 = arith.constant dense<0> : tensor<4600x1xi32>
// CHECK-NEXT:   %0 = arith.addf %arg3, %cst : tensor<8000x3xf32>
// CHECK-NEXT:   %1 = "stablehlo.scatter"(%0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:   ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
// CHECK-NEXT:     %6 = stablehlo.multiply %arg4, %arg5 : tensor<f32>
// CHECK-NEXT:     stablehlo.return %6 : tensor<f32>
// CHECK-NEXT:   }) : (tensor<8000x3xf32>, tensor<4600x1xi32>, tensor<4600x3xf32>) -> tensor<8000x3xf32>
// CHECK-NEXT:   %2 = arith.addf %1, %cst : tensor<8000x3xf32>
// CHECK-NEXT:   %3 = stablehlo.multiply %0, %arg0 : tensor<8000x3xf32>
// CHECK-NEXT:   %4 = "stablehlo.gather"(%3, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 3>}> : (tensor<8000x3xf32>, tensor<4600x1xi32>) -> tensor<4600x3xf32>
// CHECK-NEXT:   %5 = arith.addf %4, %cst_0 : tensor<4600x3xf32>
// CHECK-NEXT:   return %2, %cst_1, %5 : tensor<8000x3xf32>, tensor<4600x1xi32>, tensor<4600x3xf32>
// CHECK-NEXT: }

// -----

module {
  func.func @main(%prev : tensor<8000x3xf32>, %idxs : tensor<460000x1xi32>, %update : tensor<460000x3xf32>) -> tensor<8000x3xf32> {
    %0 = "stablehlo.scatter"(%prev, %idxs, %update) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.subtract %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<8000x3xf32>, tensor<460000x1xi32>, tensor<460000x3xf32>) -> tensor<8000x3xf32>
    return %0 : tensor<8000x3xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<8000x3xf32>, %arg1: tensor<460000x1xi32>, %arg2: tensor<460000x3xf32>, %arg3: tensor<8000x3xf32>) -> (tensor<8000x3xf32>, tensor<460000x1xi32>, tensor<460000x3xf32>) {
// CHECK-NEXT:   %cst = arith.constant dense<0.000000e+00> : tensor<8000x3xf32>
// CHECK-NEXT:   %cst_0 = arith.constant dense<0.000000e+00> : tensor<460000x3xf32>
// CHECK-NEXT:   %cst_1 = arith.constant dense<0> : tensor<460000x1xi32>
// CHECK-NEXT:   %0 = arith.addf %arg3, %cst : tensor<8000x3xf32>
// CHECK-NEXT:   %1 = arith.addf %0, %cst : tensor<8000x3xf32>
// CHECK-NEXT:   %2 = "stablehlo.gather"(%0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 3>}> : (tensor<8000x3xf32>, tensor<460000x1xi32>) -> tensor<460000x3xf32>
// CHECK-NEXT:   %3 = stablehlo.negate %2 : tensor<460000x3xf32>
// CHECK-NEXT:   %4 = arith.addf %3, %cst_0 : tensor<460000x3xf32>
// CHECK-NEXT:   return %1, %cst_1, %4 : tensor<8000x3xf32>, tensor<460000x1xi32>, tensor<460000x3xf32>
// CHECK-NEXT: }

// -----

module {
  func.func @main(%prev : tensor<8000x3xf32>, %idxs : tensor<460000x1xi32>, %update : tensor<460000x3xf32>) -> tensor<8000x3xf32> {
    %0 = "stablehlo.scatter"(%prev, %idxs, %update) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      stablehlo.return %arg4 : tensor<f32>
    }) : (tensor<8000x3xf32>, tensor<460000x1xi32>, tensor<460000x3xf32>) -> tensor<8000x3xf32>
    return %0 : tensor<8000x3xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<8000x3xf32>, %arg1: tensor<460000x1xi32>, %arg2: tensor<460000x3xf32>, %arg3: tensor<8000x3xf32>) -> (tensor<8000x3xf32>, tensor<460000x1xi32>, tensor<460000x3xf32>) {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:   %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<460000x3xf32>
// CHECK-NEXT:   %cst_1 = arith.constant dense<0.000000e+00> : tensor<8000x3xf32>
// CHECK-NEXT:   %cst_2 = arith.constant dense<0.000000e+00> : tensor<460000x3xf32>
// CHECK-NEXT:   %cst_3 = arith.constant dense<0> : tensor<460000x1xi32>
// CHECK-NEXT:   %0 = arith.addf %arg3, %cst_1 : tensor<8000x3xf32>
// CHECK-NEXT:   %1 = "stablehlo.scatter"(%0, %arg1, %cst_0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
// CHECK-NEXT:     stablehlo.return %cst : tensor<f32>
// CHECK-NEXT:   }) : (tensor<8000x3xf32>, tensor<460000x1xi32>, tensor<460000x3xf32>) -> tensor<8000x3xf32>
// CHECK-NEXT:   %2 = arith.addf %1, %cst_1 : tensor<8000x3xf32>
// CHECK-NEXT:   %3 = "stablehlo.gather"(%0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 3>}> : (tensor<8000x3xf32>, tensor<460000x1xi32>) -> tensor<460000x3xf32>
// CHECK-NEXT:   %4 = arith.addf %3, %cst_2 : tensor<460000x3xf32>
// CHECK-NEXT:   return %2, %cst_3, %4 : tensor<8000x3xf32>, tensor<460000x1xi32>, tensor<460000x3xf32>
// CHECK-NEXT: }

// -----

module {
  func.func @main(%prev : tensor<8000x3xf32>, %idxs : tensor<4600x1xi32>, %update : tensor<4600x3xf32>) -> tensor<8000x3xf32> {
    %cst = stablehlo.constant dense<5.0> : tensor<f32>
    %0 = "stablehlo.scatter"(%prev, %idxs, %update) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.multiply %cst, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<8000x3xf32>, tensor<4600x1xi32>, tensor<4600x3xf32>) -> tensor<8000x3xf32>
    return %0 : tensor<8000x3xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<8000x3xf32>, %arg1: tensor<4600x1xi32>, %arg2: tensor<4600x3xf32>, %arg3: tensor<8000x3xf32>) -> (tensor<8000x3xf32>, tensor<4600x1xi32>, tensor<4600x3xf32>) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<5.000000e+00> : tensor<8000x3xf32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<4600x3xf32>
// CHECK-NEXT:    %cst_2 = arith.constant dense<0.000000e+00> : tensor<8000x3xf32>
// CHECK-NEXT:    %cst_3 = arith.constant dense<0.000000e+00> : tensor<4600x3xf32>
// CHECK-NEXT:    %cst_4 = arith.constant dense<0> : tensor<4600x1xi32>
// CHECK-NEXT:    %0 = arith.addf %arg3, %cst_2 : tensor<8000x3xf32>
// CHECK-NEXT:    %1 = "stablehlo.scatter"(%0, %arg1, %cst_1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
// CHECK-NEXT:      stablehlo.return %cst_0 : tensor<f32>
// CHECK-NEXT:    }) : (tensor<8000x3xf32>, tensor<4600x1xi32>, tensor<4600x3xf32>) -> tensor<8000x3xf32>
// CHECK-NEXT:    %2 = arith.addf %1, %cst_2 : tensor<8000x3xf32>
// CHECK-NEXT:    %3 = stablehlo.multiply %0, %cst : tensor<8000x3xf32>
// CHECK-NEXT:    %4 = "stablehlo.gather"(%3, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 3>}> : (tensor<8000x3xf32>, tensor<4600x1xi32>) -> tensor<4600x3xf32>
// CHECK-NEXT:    %5 = arith.addf %4, %cst_3 : tensor<4600x3xf32>
// CHECK-NEXT:    return %2, %cst_4, %5 : tensor<8000x3xf32>, tensor<4600x1xi32>, tensor<4600x3xf32>
// CHECK-NEXT:  }

// -----

module {
  func.func @main(%prev : tensor<8000x3xf32>, %idxs : tensor<4600x1xi32>, %update : tensor<4600x3xf32>) -> tensor<8000x3xf32> {
    %cst = stablehlo.constant dense<5.0> : tensor<f32>
    %0 = "stablehlo.scatter"(%prev, %idxs, %update) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.add %cst, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<8000x3xf32>, tensor<4600x1xi32>, tensor<4600x3xf32>) -> tensor<8000x3xf32>
    return %0 : tensor<8000x3xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<8000x3xf32>, %arg1: tensor<4600x1xi32>, %arg2: tensor<4600x3xf32>, %arg3: tensor<8000x3xf32>) -> (tensor<8000x3xf32>, tensor<4600x1xi32>, tensor<4600x3xf32>) {
// CHECK-NEXT:   %cst = arith.constant dense<0.000000e+00> : tensor<8000x3xf32>
// CHECK-NEXT:   %cst_0 = arith.constant dense<0> : tensor<4600x1xi32>
// CHECK-NEXT:   %cst_1 = arith.constant dense<0.000000e+00> : tensor<4600x3xf32>
// CHECK-NEXT:   %0 = arith.addf %arg3, %cst : tensor<8000x3xf32>
// CHECK-NEXT:   %1 = arith.addf %0, %cst : tensor<8000x3xf32>
// CHECK-NEXT:   return %1, %cst_0, %cst_1 : tensor<8000x3xf32>, tensor<4600x1xi32>, tensor<4600x3xf32>
// CHECK-NEXT: }

// -----

module {
  func.func @main(%prev : tensor<8000x3xf32>, %idxs : tensor<4600x1xi32>, %update : tensor<4600x3xf32>) -> tensor<8000x3xf32> {
    %cst = stablehlo.constant dense<5.0> : tensor<f32>
    %0 = "stablehlo.scatter"(%prev, %idxs, %update) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.add %arg3, %cst : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<8000x3xf32>, tensor<4600x1xi32>, tensor<4600x3xf32>) -> tensor<8000x3xf32>
    return %0 : tensor<8000x3xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<8000x3xf32>, %arg1: tensor<4600x1xi32>, %arg2: tensor<4600x3xf32>, %arg3: tensor<8000x3xf32>) -> (tensor<8000x3xf32>, tensor<4600x1xi32>, tensor<4600x3xf32>) {
// CHECK-NEXT:   %cst = arith.constant dense<0.000000e+00> : tensor<8000x3xf32>
// CHECK-NEXT:   %cst_0 = arith.constant dense<0> : tensor<4600x1xi32>
// CHECK-NEXT:   %cst_1 = arith.constant dense<0.000000e+00> : tensor<4600x3xf32>
// CHECK-NEXT:   %0 = arith.addf %arg3, %cst : tensor<8000x3xf32>
// CHECK-NEXT:   %1 = arith.addf %0, %cst : tensor<8000x3xf32>
// CHECK-NEXT:   return %1, %cst_0, %cst_1 : tensor<8000x3xf32>, tensor<4600x1xi32>, tensor<4600x3xf32>
// CHECK-NEXT: }

// -----

module {
  func.func @main(%prev : tensor<8000x3xf32>, %idxs : tensor<4600x1xi32>, %update : tensor<4600x3xf32>) -> tensor<8000x3xf32> {
    %cst = stablehlo.constant dense<5.0> : tensor<f32>
    %0 = "stablehlo.scatter"(%prev, %idxs, %update) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.add %arg4, %cst : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<8000x3xf32>, tensor<4600x1xi32>, tensor<4600x3xf32>) -> tensor<8000x3xf32>
    return %0 : tensor<8000x3xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<8000x3xf32>, %arg1: tensor<4600x1xi32>, %arg2: tensor<4600x3xf32>, %arg3: tensor<8000x3xf32>) -> (tensor<8000x3xf32>, tensor<4600x1xi32>, tensor<4600x3xf32>) {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:   %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<4600x3xf32>
// CHECK-NEXT:   %cst_1 = arith.constant dense<0.000000e+00> : tensor<8000x3xf32>
// CHECK-NEXT:   %cst_2 = arith.constant dense<0.000000e+00> : tensor<4600x3xf32>
// CHECK-NEXT:   %cst_3 = arith.constant dense<0> : tensor<4600x1xi32>
// CHECK-NEXT:   %0 = arith.addf %arg3, %cst_1 : tensor<8000x3xf32>
// CHECK-NEXT:   %1 = "stablehlo.scatter"(%0, %arg1, %cst_0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:     ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
// CHECK-NEXT:       stablehlo.return %cst : tensor<f32>
// CHECK-NEXT:     }) : (tensor<8000x3xf32>, tensor<4600x1xi32>, tensor<4600x3xf32>) -> tensor<8000x3xf32>
// CHECK-NEXT:   %2 = arith.addf %1, %cst_1 : tensor<8000x3xf32>
// CHECK-NEXT:   %3 = "stablehlo.gather"(%0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 3>}> : (tensor<8000x3xf32>, tensor<4600x1xi32>) -> tensor<4600x3xf32>
// CHECK-NEXT:   %4 = arith.addf %3, %cst_2 : tensor<4600x3xf32>
// CHECK-NEXT:   return %2, %cst_3, %4 : tensor<8000x3xf32>, tensor<4600x1xi32>, tensor<4600x3xf32>
// CHECK-NEXT: }

// -----

module {
  func.func @main(%prev : tensor<8000x3xf32>, %idxs : tensor<4600x1xi32>, %update : tensor<4600x3xf32>) -> tensor<8000x3xf32> {
    %cst = stablehlo.constant dense<5.0> : tensor<f32>
    %0 = "stablehlo.scatter"(%prev, %idxs, %update) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.multiply %arg3, %cst : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<8000x3xf32>, tensor<4600x1xi32>, tensor<4600x3xf32>) -> tensor<8000x3xf32>
    return %0 : tensor<8000x3xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<8000x3xf32>, %arg1: tensor<4600x1xi32>, %arg2: tensor<4600x3xf32>, %arg3: tensor<8000x3xf32>) -> (tensor<8000x3xf32>, tensor<4600x1xi32>, tensor<4600x3xf32>) {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<4600x3xf32>
// CHECK-NEXT:   %cst_0 = stablehlo.constant dense<5.000000e+00> : tensor<f32>
// CHECK-NEXT:   %cst_1 = arith.constant dense<0.000000e+00> : tensor<8000x3xf32>
// CHECK-NEXT:   %cst_2 = arith.constant dense<0> : tensor<4600x1xi32>
// CHECK-NEXT:   %cst_3 = arith.constant dense<0.000000e+00> : tensor<4600x3xf32>
// CHECK-NEXT:   %0 = arith.addf %arg3, %cst_1 : tensor<8000x3xf32>
// CHECK-NEXT:   %1 = "stablehlo.scatter"(%0, %arg1, %cst) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:   ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
// CHECK-NEXT:     %3 = stablehlo.multiply %arg4, %cst_0 : tensor<f32>
// CHECK-NEXT:     stablehlo.return %3 : tensor<f32>
// CHECK-NEXT:   }) : (tensor<8000x3xf32>, tensor<4600x1xi32>, tensor<4600x3xf32>) -> tensor<8000x3xf32>
// CHECK-NEXT:   %2 = arith.addf %1, %cst_1 : tensor<8000x3xf32>
// CHECK-NEXT:   return %2, %cst_2, %cst_3 : tensor<8000x3xf32>, tensor<4600x1xi32>, tensor<4600x3xf32>
// CHECK-NEXT: }

// -----

module {
  func.func @main(%prev : tensor<8000x3xf32>, %idxs : tensor<4600x1xi32>, %update : tensor<4600x3xf32>) -> tensor<8000x3xf32> {
    %cst = stablehlo.constant dense<5.0> : tensor<f32>
    %0 = "stablehlo.scatter"(%prev, %idxs, %update) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.multiply %cst, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<8000x3xf32>, tensor<4600x1xi32>, tensor<4600x3xf32>) -> tensor<8000x3xf32>
    return %0 : tensor<8000x3xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<8000x3xf32>, %arg1: tensor<4600x1xi32>, %arg2: tensor<4600x3xf32>, %arg3: tensor<8000x3xf32>) -> (tensor<8000x3xf32>, tensor<4600x1xi32>, tensor<4600x3xf32>) {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<4600x3xf32>
// CHECK-NEXT:   %cst_0 = stablehlo.constant dense<5.000000e+00> : tensor<f32>
// CHECK-NEXT:   %cst_1 = arith.constant dense<0.000000e+00> : tensor<8000x3xf32>
// CHECK-NEXT:   %cst_2 = arith.constant dense<0> : tensor<4600x1xi32>
// CHECK-NEXT:   %cst_3 = arith.constant dense<0.000000e+00> : tensor<4600x3xf32>
// CHECK-NEXT:   %0 = arith.addf %arg3, %cst_1 : tensor<8000x3xf32>
// CHECK-NEXT:   %1 = "stablehlo.scatter"(%0, %arg1, %cst) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:   ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
// CHECK-NEXT:     %3 = stablehlo.multiply %arg4, %cst_0 : tensor<f32>
// CHECK-NEXT:     stablehlo.return %3 : tensor<f32>
// CHECK-NEXT:   }) : (tensor<8000x3xf32>, tensor<4600x1xi32>, tensor<4600x3xf32>) -> tensor<8000x3xf32>
// CHECK-NEXT:   %2 = arith.addf %1, %cst_1 : tensor<8000x3xf32>
// CHECK-NEXT:   return %2, %cst_2, %cst_3 : tensor<8000x3xf32>, tensor<4600x1xi32>, tensor<4600x3xf32>
// CHECK-NEXT: }
