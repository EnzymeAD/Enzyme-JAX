// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=gather_operand_box" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// The gathers of a value read only a box of it (start indices built of
// constants and iotas, each window under its bound): they gather from the
// slice of that box, the union of what they read.
func.func @box(%x: tensor<64x8xf64>) -> (tensor<3x8xf64>, tensor<2x4xf64>) {
  %i = stablehlo.constant dense<[[2], [5], [9]]> : tensor<3x1xi64>
  %0 = "stablehlo.gather"(%x, %i) {dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 8>} : (tensor<64x8xf64>, tensor<3x1xi64>) -> tensor<3x8xf64>
  %j = stablehlo.iota dim = 0 : tensor<2x1xi64>
  %1 = "stablehlo.gather"(%x, %j) {dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 4>} : (tensor<64x8xf64>, tensor<2x1xi64>) -> tensor<2x4xf64>
  return %0, %1 : tensor<3x8xf64>, tensor<2x4xf64>
}

// CHECK-LABEL: func.func @box(
// CHECK-SAME: %[[X:[^ ,:)]+]]: tensor<64x8xf64>) -> (tensor<3x8xf64>, tensor<2x4xf64>) {
// CHECK-NEXT: %[[I:[^ ,:)]+]] = stablehlo.constant dense<{{\[\[}}2], [5], [9]]> : tensor<3x1xi64>
// CHECK-NEXT: %[[S:[^ ,:)]+]] = stablehlo.slice %[[X]] [0:10, 0:8] : (tensor<64x8xf64>) -> tensor<10x8xf64>
// CHECK-NEXT: %[[G0:[^ ,:)]+]] = "stablehlo.gather"(%[[S]], %[[I]]) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 8>}> : (tensor<10x8xf64>, tensor<3x1xi64>) -> tensor<3x8xf64>
// CHECK-NEXT: %[[J:[^ ,:)]+]] = stablehlo.iota dim = 0 : tensor<2x1xi64>
// CHECK-NEXT: %[[G1:[^ ,:)]+]] = "stablehlo.gather"(%[[S]], %[[J]]) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 4>}> : (tensor<10x8xf64>, tensor<2x1xi64>) -> tensor<2x4xf64>
// CHECK-NEXT: return %[[G0]], %[[G1]] : tensor<3x8xf64>, tensor<2x4xf64>
// CHECK-NEXT: }

// A start index past the last the window fits at is clamped to it: the
// whole dimension is read, nothing to cut.
func.func @clamped(%x: tensor<64x8xf64>) -> tensor<2x8xf64> {
  %i = stablehlo.constant dense<[[2], [70]]> : tensor<2x1xi64>
  %0 = "stablehlo.gather"(%x, %i) {dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 8>} : (tensor<64x8xf64>, tensor<2x1xi64>) -> tensor<2x8xf64>
  return %0 : tensor<2x8xf64>
}

// CHECK-LABEL: func.func @clamped(
// CHECK-SAME: %[[X:[^ ,:)]+]]: tensor<64x8xf64>) -> tensor<2x8xf64> {
// CHECK-NEXT: %[[I:[^ ,:)]+]] = stablehlo.constant dense<{{\[\[}}2], [70]]> : tensor<2x1xi64>
// CHECK-NEXT: %[[G:[^ ,:)]+]] = "stablehlo.gather"(%[[X]], %[[I]]) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 8>}> : (tensor<64x8xf64>, tensor<2x1xi64>) -> tensor<2x8xf64>
// CHECK-NEXT: return %[[G]] : tensor<2x8xf64>
// CHECK-NEXT: }

// A value read by anything but gathers is left whole.
func.func @other_user(%x: tensor<64x8xf64>) -> (tensor<3x8xf64>, tensor<64x8xf64>) {
  %i = stablehlo.constant dense<[[2], [5], [9]]> : tensor<3x1xi64>
  %0 = "stablehlo.gather"(%x, %i) {dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 8>} : (tensor<64x8xf64>, tensor<3x1xi64>) -> tensor<3x8xf64>
  %1 = stablehlo.negate %x : tensor<64x8xf64>
  return %0, %1 : tensor<3x8xf64>, tensor<64x8xf64>
}

// CHECK-LABEL: func.func @other_user(
// CHECK-SAME: %[[X:[^ ,:)]+]]: tensor<64x8xf64>) -> (tensor<3x8xf64>, tensor<64x8xf64>) {
// CHECK-NEXT: %[[I:[^ ,:)]+]] = stablehlo.constant dense<{{\[\[}}2], [5], [9]]> : tensor<3x1xi64>
// CHECK-NEXT: %[[G:[^ ,:)]+]] = "stablehlo.gather"(%[[X]], %[[I]]) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 8>}> : (tensor<64x8xf64>, tensor<3x1xi64>) -> tensor<3x8xf64>
// CHECK-NEXT: %[[N:[^ ,:)]+]] = stablehlo.negate %[[X]] : tensor<64x8xf64>
// CHECK-NEXT: return %[[G]], %[[N]] : tensor<3x8xf64>, tensor<64x8xf64>
// CHECK-NEXT: }
