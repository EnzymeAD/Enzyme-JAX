// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=gather_of_masked_scatter;scatter_of_scatter_simplify" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// The scatter writes the slots the mask holds (the others go to -1, out of
// range); the gather of the same slots reads what it wrote where the mask
// holds, what was there otherwise.
func.func @masked(%y: tensor<100xf64>, %idx: tensor<8x1xi64>, %m: tensor<8x1xi1>, %v: tensor<8xf64>) -> tensor<8xf64> {
  %neg = stablehlo.constant dense<-1> : tensor<8x1xi64>
  %j = stablehlo.select %m, %idx, %neg : tensor<8x1xi1>, tensor<8x1xi64>
  %s = "stablehlo.scatter"(%y, %j, %v) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<100xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<100xf64>
  %r = "stablehlo.gather"(%s, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<100xf64>, tensor<8x1xi64>) -> tensor<8xf64>
  return %r : tensor<8xf64>
}

// CHECK:  func.func @masked(%arg0: tensor<100xf64>, %arg1: tensor<8x1xi64>, %arg2: tensor<8x1xi1>, %arg3: tensor<8xf64>) -> tensor<8xf64> {
// CHECK-NEXT:    %0 = stablehlo.reshape %arg2 : (tensor<8x1xi1>) -> tensor<8xi1>
// CHECK-NEXT:    %1 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<100xf64>, tensor<8x1xi64>) -> tensor<8xf64>
// CHECK-NEXT:    %2 = stablehlo.select %0, %arg3, %1 : tensor<8xi1>, tensor<8xf64>
// CHECK-NEXT:    return %2 : tensor<8xf64>
// CHECK-NEXT:  }

// -----


// A kernel's `y[i] += v` twice over, raised with the running sum forwarded:
// the second store overwrites every slot of the first (the same indices,
// the same slots masked off), unique or not, so only it remains.
func.func @overwrite_chain(%y: tensor<100xf64>, %j: tensor<8x1xi64>, %v1: tensor<8xf64>, %v2: tensor<8xf64>) -> tensor<100xf64> {
  %s1 = "stablehlo.scatter"(%y, %j, %v1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<100xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<100xf64>
  %sum = stablehlo.add %v1, %v2 : tensor<8xf64>
  %s2 = "stablehlo.scatter"(%s1, %j, %sum) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<100xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<100xf64>
  return %s2 : tensor<100xf64>
}

// CHECK:  func.func @overwrite_chain(%arg0: tensor<100xf64>, %arg1: tensor<8x1xi64>, %arg2: tensor<8xf64>, %arg3: tensor<8xf64>) -> tensor<100xf64> {
// CHECK-NEXT:    %0 = stablehlo.add %arg2, %arg3 : tensor<8xf64>
// CHECK-NEXT:    %1 = "stablehlo.scatter"(%arg0, %arg1, %0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:    ^bb0(%arg4: tensor<f64>, %arg5: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg5 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<100xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<100xf64>
// CHECK-NEXT:    return %1 : tensor<100xf64>
// CHECK-NEXT:  }

// -----


// The second scatter adds: the first still counts, and without unique
// indices the two are not merged.
func.func @add_chain(%y: tensor<100xf64>, %j: tensor<8x1xi64>, %v1: tensor<8xf64>, %v2: tensor<8xf64>) -> tensor<100xf64> {
  %s1 = "stablehlo.scatter"(%y, %j, %v1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<100xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<100xf64>
  %s2 = "stablehlo.scatter"(%s1, %j, %v2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    %c = stablehlo.add %a, %b : tensor<f64>
    stablehlo.return %c : tensor<f64>
  }) : (tensor<100xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<100xf64>
  return %s2 : tensor<100xf64>
}

// CHECK:  func.func @add_chain(%arg0: tensor<100xf64>, %arg1: tensor<8x1xi64>, %arg2: tensor<8xf64>, %arg3: tensor<8xf64>) -> tensor<100xf64> {
// CHECK-NEXT:    %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:    ^bb0(%arg4: tensor<f64>, %arg5: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg5 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<100xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<100xf64>
// CHECK-NEXT:    %1 = "stablehlo.scatter"(%0, %arg1, %arg3) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:    ^bb0(%arg4: tensor<f64>, %arg5: tensor<f64>):
// CHECK-NEXT:      %2 = stablehlo.add %arg4, %arg5 : tensor<f64>
// CHECK-NEXT:      stablehlo.return %2 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<100xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<100xf64>
// CHECK-NEXT:    return %1 : tensor<100xf64>
// CHECK-NEXT:  }
