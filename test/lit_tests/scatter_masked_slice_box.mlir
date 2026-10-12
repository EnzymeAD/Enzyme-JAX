// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=scatter_masked_index_slice" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// A lane store into a two-dimensional scratch, under the mask a padded 8x8
// lane grid has once its extent (3x3) is a constant: the mask is the folded
// constant, broadcast over the index vector.
func.func @scratch2d(%smem: tensor<32x32xf64>, %update: tensor<8x8xf64>, %idx: tensor<8x8x2xi64>) -> tensor<32x32xf64> {
  %mask = stablehlo.constant dense<[[true, true, true, false, false, false, false, false], [true, true, true, false, false, false, false, false], [true, true, true, false, false, false, false, false], [false, false, false, false, false, false, false, false], [false, false, false, false, false, false, false, false], [false, false, false, false, false, false, false, false], [false, false, false, false, false, false, false, false], [false, false, false, false, false, false, false, false]]> : tensor<8x8xi1>
  %off = stablehlo.constant dense<-1> : tensor<8x8x2xi64>
  %pred = stablehlo.broadcast_in_dim %mask, dims = [0, 1] : (tensor<8x8xi1>) -> tensor<8x8x2xi1>
  %sel = stablehlo.select %pred, %idx, %off : tensor<8x8x2xi1>, tensor<8x8x2xi64>
  %0 = "stablehlo.scatter"(%smem, %sel, %update) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<32x32xf64>, tensor<8x8x2xi64>, tensor<8x8xf64>) -> tensor<32x32xf64>
  return %0 : tensor<32x32xf64>
}

// The constant lane mask anded with a term known only at run time: the
// constant still bounds the live lanes, and the mask stays on them.
func.func @and_runtime(%out: tensor<64xf64>, %update: tensor<8xf64>, %idx: tensor<8x1xi64>, %k: tensor<8xi1>) -> tensor<64xf64> {
  %mask = stablehlo.constant dense<[true, true, true, false, false, false, false, false]> : tensor<8xi1>
  %off = stablehlo.constant dense<-1> : tensor<8x1xi64>
  %live = stablehlo.and %mask, %k : tensor<8xi1>
  %pred = stablehlo.reshape %live : (tensor<8xi1>) -> tensor<8x1xi1>
  %sel = stablehlo.select %pred, %idx, %off : tensor<8x1xi1>, tensor<8x1xi64>
  %0 = "stablehlo.scatter"(%out, %sel, %update) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<64xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<64xf64>
  return %0 : tensor<64xf64>
}

// Two live runs: their bounding box is cut to, and the mask stays.
func.func @two_runs(%out: tensor<64xf64>, %update: tensor<8xf64>, %idx: tensor<8x1xi64>) -> tensor<64xf64> {
  %mask = stablehlo.constant dense<[[true], [true], [false], [true], [false], [false], [false], [false]]> : tensor<8x1xi1>
  %off = stablehlo.constant dense<-1> : tensor<8x1xi64>
  %sel = stablehlo.select %mask, %idx, %off : tensor<8x1xi1>, tensor<8x1xi64>
  %0 = "stablehlo.scatter"(%out, %sel, %update) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<64xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<64xf64>
  return %0 : tensor<64xf64>
}

// A window scatter writes more than one element per position: unchanged.
func.func @window(%out: tensor<64x4xf64>, %update: tensor<8x4xf64>, %idx: tensor<8x1xi64>) -> tensor<64x4xf64> {
  %mask = stablehlo.constant dense<[[true], [true], [true], [false], [false], [false], [false], [false]]> : tensor<8x1xi1>
  %off = stablehlo.constant dense<-1> : tensor<8x1xi64>
  %sel = stablehlo.select %mask, %idx, %off : tensor<8x1xi1>, tensor<8x1xi64>
  %0 = "stablehlo.scatter"(%out, %sel, %update) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<64x4xf64>, tensor<8x1xi64>, tensor<8x4xf64>) -> tensor<64x4xf64>
  return %0 : tensor<64x4xf64>
}

// CHECK-LABEL: func.func @scratch2d(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<32x32xf64>, %[[V1:[^ ,:)]+]]: tensor<8x8xf64>, %[[V2:[^ ,:)]+]]: tensor<8x8x2xi64>) -> tensor<32x32xf64> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.slice %[[V2]] [0:3, 0:3, 0:2] : (tensor<8x8x2xi64>) -> tensor<3x3x2xi64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.slice %[[V1]] [0:3, 0:3] : (tensor<8x8xf64>) -> tensor<3x3xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = "stablehlo.scatter"(%[[V0]], %[[V3]], %[[V4]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>, unique_indices = false}> ({
// CHECK-NEXT: ^bb0(%[[V6:[^ ,:)]+]]: tensor<f64>, %[[V7:[^ ,:)]+]]: tensor<f64>):
// CHECK-NEXT: stablehlo.return %[[V7]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<32x32xf64>, tensor<3x3x2xi64>, tensor<3x3xf64>) -> tensor<32x32xf64>
// CHECK-NEXT: return %[[V5]] : tensor<32x32xf64>
// CHECK-NEXT: }

// CHECK-LABEL: func.func @and_runtime(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<64xf64>, %[[V1:[^ ,:)]+]]: tensor<8xf64>, %[[V2:[^ ,:)]+]]: tensor<8x1xi64>, %[[V3:[^ ,:)]+]]: tensor<8xi1>) -> tensor<64xf64> {
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.constant dense<[true, true, true, false, false, false, false, false]> : tensor<8xi1>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.constant dense<-1> : tensor<8x1xi64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.and %[[V4]], %[[V3]] : tensor<8xi1>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.reshape %[[V6]] : (tensor<8xi1>) -> tensor<8x1xi1>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.select %[[V7]], %[[V2]], %[[V5]] : tensor<8x1xi1>, tensor<8x1xi64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.slice %[[V8]] [0:3, 0:1] : (tensor<8x1xi64>) -> tensor<3x1xi64>
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = stablehlo.slice %[[V1]] [0:3] : (tensor<8xf64>) -> tensor<3xf64>
// CHECK-NEXT: %[[V11:[^ ,:)]+]] = "stablehlo.scatter"(%[[V0]], %[[V9]], %[[V10]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT: ^bb0(%[[V12:[^ ,:)]+]]: tensor<f64>, %[[V13:[^ ,:)]+]]: tensor<f64>):
// CHECK-NEXT: stablehlo.return %[[V13]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<64xf64>, tensor<3x1xi64>, tensor<3xf64>) -> tensor<64xf64>
// CHECK-NEXT: return %[[V11]] : tensor<64xf64>
// CHECK-NEXT: }

// CHECK-LABEL: func.func @two_runs(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<64xf64>, %[[V1:[^ ,:)]+]]: tensor<8xf64>, %[[V2:[^ ,:)]+]]: tensor<8x1xi64>) -> tensor<64xf64> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<{{\[\[}}true], [true], [false], [true], [false], [false], [false], [false]]> : tensor<8x1xi1>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.constant dense<-1> : tensor<8x1xi64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.select %[[V3]], %[[V2]], %[[V4]] : tensor<8x1xi1>, tensor<8x1xi64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.slice %[[V5]] [0:4, 0:1] : (tensor<8x1xi64>) -> tensor<4x1xi64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.slice %[[V1]] [0:4] : (tensor<8xf64>) -> tensor<4xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = "stablehlo.scatter"(%[[V0]], %[[V6]], %[[V7]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT: ^bb0(%[[V9:[^ ,:)]+]]: tensor<f64>, %[[V10:[^ ,:)]+]]: tensor<f64>):
// CHECK-NEXT: stablehlo.return %[[V10]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<64xf64>, tensor<4x1xi64>, tensor<4xf64>) -> tensor<64xf64>
// CHECK-NEXT: return %[[V8]] : tensor<64xf64>
// CHECK-NEXT: }

// CHECK-LABEL: func.func @window(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<64x4xf64>, %[[V1:[^ ,:)]+]]: tensor<8x4xf64>, %[[V2:[^ ,:)]+]]: tensor<8x1xi64>) -> tensor<64x4xf64> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<{{\[\[}}true], [true], [true], [false], [false], [false], [false], [false]]> : tensor<8x1xi1>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.constant dense<-1> : tensor<8x1xi64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.select %[[V3]], %[[V2]], %[[V4]] : tensor<8x1xi1>, tensor<8x1xi64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = "stablehlo.scatter"(%[[V0]], %[[V5]], %[[V1]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT: ^bb0(%[[V7:[^ ,:)]+]]: tensor<f64>, %[[V8:[^ ,:)]+]]: tensor<f64>):
// CHECK-NEXT: stablehlo.return %[[V8]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<64x4xf64>, tensor<8x1xi64>, tensor<8x4xf64>) -> tensor<64x4xf64>
// CHECK-NEXT: return %[[V6]] : tensor<64x4xf64>
// CHECK-NEXT: }
