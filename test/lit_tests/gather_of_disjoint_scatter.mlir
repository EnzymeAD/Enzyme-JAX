// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=gather_of_disjoint_scatter" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// The scatter writes positions 0 and 1; the gather reads 2 and 3.
func.func @disjoint(%x: tensor<8xf64>, %u: tensor<2xf64>) -> tensor<2xf64> {
  %si = stablehlo.constant dense<[[0], [1]]> : tensor<2x1xi64>
  %gi = stablehlo.constant dense<[[2], [3]]> : tensor<2x1xi64>
  %s = "stablehlo.scatter"(%x, %si, %u) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<8xf64>, tensor<2x1xi64>, tensor<2xf64>) -> tensor<8xf64>
  %g = "stablehlo.gather"(%s, %gi) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<8xf64>, tensor<2x1xi64>) -> tensor<2xf64>
  return %g : tensor<2xf64>
}

// The gather reads position 1, which the scatter writes: unchanged.
func.func @overlap(%x: tensor<8xf64>, %u: tensor<2xf64>) -> tensor<2xf64> {
  %si = stablehlo.constant dense<[[0], [1]]> : tensor<2x1xi64>
  %gi = stablehlo.constant dense<[[1], [3]]> : tensor<2x1xi64>
  %s = "stablehlo.scatter"(%x, %si, %u) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<8xf64>, tensor<2x1xi64>, tensor<2xf64>) -> tensor<8xf64>
  %g = "stablehlo.gather"(%s, %gi) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<8xf64>, tensor<2x1xi64>) -> tensor<2xf64>
  return %g : tensor<2xf64>
}

// A masked-off store (index -1) writes nothing: the gather of position 5
// reads the input.
func.func @masked(%x: tensor<8xf64>, %u: tensor<2xf64>) -> tensor<2xf64> {
  %si = stablehlo.constant dense<[[4], [-1]]> : tensor<2x1xi64>
  %gi = stablehlo.constant dense<[[5], [0]]> : tensor<2x1xi64>
  %s = "stablehlo.scatter"(%x, %si, %u) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<8xf64>, tensor<2x1xi64>, tensor<2xf64>) -> tensor<8xf64>
  %g = "stablehlo.gather"(%s, %gi) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<8xf64>, tensor<2x1xi64>) -> tensor<2xf64>
  return %g : tensor<2xf64>
}

// The scatter writes whole rows 0 and 2 of a 4x3 tensor; the gather reads
// windows of row 1 and row 3.
func.func @rows(%x: tensor<4x3xf64>, %u: tensor<2x3xf64>) -> tensor<2x2xf64> {
  %si = stablehlo.constant dense<[[0], [2]]> : tensor<2x1xi64>
  %gi = stablehlo.constant dense<[[1, 0], [3, 1]]> : tensor<2x2xi64>
  %s = "stablehlo.scatter"(%x, %si, %u) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<4x3xf64>, tensor<2x1xi64>, tensor<2x3xf64>) -> tensor<4x3xf64>
  %g = "stablehlo.gather"(%s, %gi) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 2>}> : (tensor<4x3xf64>, tensor<2x2xi64>) -> tensor<2x2xf64>
  return %g : tensor<2x2xf64>
}

// A chain of two scatters: the gather reads what the first wrote, past the
// second, which writes elsewhere.
func.func @chain(%x: tensor<8xf64>, %u: tensor<1xf64>, %v: tensor<1xf64>) -> tensor<1xf64> {
  %i0 = stablehlo.constant dense<[[0]]> : tensor<1x1xi64>
  %i1 = stablehlo.constant dense<[[1]]> : tensor<1x1xi64>
  %s0 = "stablehlo.scatter"(%x, %i0, %u) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<8xf64>, tensor<1x1xi64>, tensor<1xf64>) -> tensor<8xf64>
  %s1 = "stablehlo.scatter"(%s0, %i1, %v) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<8xf64>, tensor<1x1xi64>, tensor<1xf64>) -> tensor<8xf64>
  %g = "stablehlo.gather"(%s1, %i0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<8xf64>, tensor<1x1xi64>) -> tensor<1xf64>
  return %g : tensor<1xf64>
}

// CHECK-LABEL: func.func @disjoint(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8xf64>, %[[V1:[^ ,:)]+]]: tensor<2xf64>) -> tensor<2xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<{{\[\[}}2], [3]]> : tensor<2x1xi64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = "stablehlo.gather"(%[[V0]], %[[V2]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<8xf64>, tensor<2x1xi64>) -> tensor<2xf64>
// CHECK-NEXT: return %[[V3]] : tensor<2xf64>
// CHECK-NEXT: }

// CHECK-LABEL: func.func @overlap(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8xf64>, %[[V1:[^ ,:)]+]]: tensor<2xf64>) -> tensor<2xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<{{\[\[}}0], [1]]> : tensor<2x1xi64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<{{\[\[}}1], [3]]> : tensor<2x1xi64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = "stablehlo.scatter"(%[[V0]], %[[V2]], %[[V1]]) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
// CHECK-NEXT: ^bb0(%[[V5:[^ ,:)]+]]: tensor<f64>, %[[V6:[^ ,:)]+]]: tensor<f64>):
// CHECK-NEXT: stablehlo.return %[[V6]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<8xf64>, tensor<2x1xi64>, tensor<2xf64>) -> tensor<8xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = "stablehlo.gather"(%[[V4]], %[[V3]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<8xf64>, tensor<2x1xi64>) -> tensor<2xf64>
// CHECK-NEXT: return %[[V7]] : tensor<2xf64>
// CHECK-NEXT: }

// CHECK-LABEL: func.func @masked(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8xf64>, %[[V1:[^ ,:)]+]]: tensor<2xf64>) -> tensor<2xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<{{\[\[}}5], [0]]> : tensor<2x1xi64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = "stablehlo.gather"(%[[V0]], %[[V2]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<8xf64>, tensor<2x1xi64>) -> tensor<2xf64>
// CHECK-NEXT: return %[[V3]] : tensor<2xf64>
// CHECK-NEXT: }

// CHECK-LABEL: func.func @rows(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4x3xf64>, %[[V1:[^ ,:)]+]]: tensor<2x3xf64>) -> tensor<2x2xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<{{\[\[}}1, 0], [3, 1]]> : tensor<2x2xi64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = "stablehlo.gather"(%[[V0]], %[[V2]]) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 2>}> : (tensor<4x3xf64>, tensor<2x2xi64>) -> tensor<2x2xf64>
// CHECK-NEXT: return %[[V3]] : tensor<2x2xf64>
// CHECK-NEXT: }

// CHECK-LABEL: func.func @chain(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8xf64>, %[[V1:[^ ,:)]+]]: tensor<1xf64>, %[[V2:[^ ,:)]+]]: tensor<1xf64>) -> tensor<1xf64> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1x1xi64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = "stablehlo.scatter"(%[[V0]], %[[V3]], %[[V1]]) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
// CHECK-NEXT: ^bb0(%[[V5:[^ ,:)]+]]: tensor<f64>, %[[V6:[^ ,:)]+]]: tensor<f64>):
// CHECK-NEXT: stablehlo.return %[[V6]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<8xf64>, tensor<1x1xi64>, tensor<1xf64>) -> tensor<8xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = "stablehlo.gather"(%[[V4]], %[[V3]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<8xf64>, tensor<1x1xi64>) -> tensor<1xf64>
// CHECK-NEXT: return %[[V7]] : tensor<1xf64>
// CHECK-NEXT: }
