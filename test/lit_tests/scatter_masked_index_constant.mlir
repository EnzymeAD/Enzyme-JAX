// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=scatter_masked_index_simplify" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// A masked store of a vector at constant, distinct places (a per-thread
// scratch written lane by lane): the masked-off lanes write their original
// values back, and the scatter folds to a select.
func.func @constant_masked(%input: tensor<4xf64>, %update: tensor<4xf64>, %pred: tensor<4x1xi1>) -> tensor<4xf64> {
  %idx = stablehlo.constant dense<[[2], [0], [3], [1]]> : tensor<4x1xi64>
  %off = stablehlo.constant dense<-1> : tensor<4x1xi64>
  %sel = stablehlo.select %pred, %idx, %off : tensor<4x1xi1>, tensor<4x1xi64>
  %0 = "stablehlo.scatter"(%input, %sel, %update) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<4xf64>, tensor<4x1xi64>, tensor<4xf64>) -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

// An iota plus an offset that differs by lane brings lanes 0 and 1 to the
// same place: the live indices are not unique, and the scatter stays.
func.func @offset_collides(%input: tensor<4xf64>, %update: tensor<4xf64>, %pred: tensor<4x1xi1>) -> tensor<4xf64> {
  %i = stablehlo.iota dim = 0 : tensor<4x1xi64>
  %c = stablehlo.constant dense<[[1], [0], [0], [0]]> : tensor<4x1xi64>
  %idx = stablehlo.add %i, %c : tensor<4x1xi64>
  %off = stablehlo.constant dense<-1> : tensor<4x1xi64>
  %sel = stablehlo.select %pred, %idx, %off : tensor<4x1xi1>, tensor<4x1xi64>
  %0 = "stablehlo.scatter"(%input, %sel, %update) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<4xf64>, tensor<4x1xi64>, tensor<4xf64>) -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

// CHECK-LABEL: func.func @constant_masked(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4xf64>, %[[V1:[^ ,:)]+]]: tensor<4xf64>, %[[V2:[^ ,:)]+]]: tensor<4x1xi1>) -> tensor<4xf64> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<{{\[\[}}2], [0], [3], [1]]> : tensor<4x1xi64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = "stablehlo.gather"(%[[V0]], %[[V3]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<4xf64>, tensor<4x1xi64>) -> tensor<4xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.reshape %[[V2]] : (tensor<4x1xi1>) -> tensor<4xi1>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.select %[[V5]], %[[V1]], %[[V4]] : tensor<4xi1>, tensor<4xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = "stablehlo.scatter"(%[[V0]], %[[V3]], %[[V6]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT: ^bb0(%[[V8:[^ ,:)]+]]: tensor<f64>, %[[V9:[^ ,:)]+]]: tensor<f64>):
// CHECK-NEXT: stablehlo.return %[[V9]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<4xf64>, tensor<4x1xi64>, tensor<4xf64>) -> tensor<4xf64>
// CHECK-NEXT: return %[[V7]] : tensor<4xf64>
// CHECK-NEXT: }
// CHECK-LABEL: func.func @offset_collides(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4xf64>, %[[V1:[^ ,:)]+]]: tensor<4xf64>, %[[V2:[^ ,:)]+]]: tensor<4x1xi1>) -> tensor<4xf64> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<-1> : tensor<4x1xi64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.constant dense<{{\[\[}}1], [0], [0], [0]]> : tensor<4x1xi64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.iota dim = 0 : tensor<4x1xi64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.add %[[V5]], %[[V4]] : tensor<4x1xi64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.select %[[V2]], %[[V6]], %[[V3]] : tensor<4x1xi1>, tensor<4x1xi64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = "stablehlo.scatter"(%[[V0]], %[[V7]], %[[V1]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT: ^bb0(%[[V9:[^ ,:)]+]]: tensor<f64>, %[[V10:[^ ,:)]+]]: tensor<f64>):
// CHECK-NEXT: stablehlo.return %[[V10]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<4xf64>, tensor<4x1xi64>, tensor<4xf64>) -> tensor<4xf64>
// CHECK-NEXT: return %[[V8]] : tensor<4xf64>
// CHECK-NEXT: }
