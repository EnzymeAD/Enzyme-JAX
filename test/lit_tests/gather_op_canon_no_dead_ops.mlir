// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=gather_op_canon" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// The index 6 * (2 * j + i) + k reads each component in bounds, but k runs
// past the factor 6 it would have in the flattened operand: no slice. Every
// check is made before anything is built, so the pattern fails cleanly (it
// used to build the flattening reshape first and leave it dead, which the
// greedy driver erased and retried forever).
func.func @overlapping_terms(%x: tensor<3x48xf64>) -> tensor<3x2x3x10xf64> {
  %i = stablehlo.iota dim = 1 : tensor<3x2x3x10x1xi64>
  %j = stablehlo.iota dim = 2 : tensor<3x2x3x10x1xi64>
  %k = stablehlo.iota dim = 3 : tensor<3x2x3x10x1xi64>
  %c2 = stablehlo.constant dense<2> : tensor<3x2x3x10x1xi64>
  %c6 = stablehlo.constant dense<6> : tensor<3x2x3x10x1xi64>
  %j2 = stablehlo.multiply %j, %c2 : tensor<3x2x3x10x1xi64>
  %s = stablehlo.add %j2, %i : tensor<3x2x3x10x1xi64>
  %s6 = stablehlo.multiply %s, %c6 : tensor<3x2x3x10x1xi64>
  %idx = stablehlo.add %s6, %k : tensor<3x2x3x10x1xi64>
  %g = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 4>, slice_sizes = array<i64: 1, 1>}> : (tensor<3x48xf64>, tensor<3x2x3x10x1xi64>) -> tensor<3x2x3x10xf64>
  return %g : tensor<3x2x3x10xf64>
}

// CHECK-LABEL: func.func @overlapping_terms(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<3x48xf64>) -> tensor<3x2x3x10xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.constant dense<6> : tensor<3x2x3x10x1xi64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<2> : tensor<3x2x3x10x1xi64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.iota dim = 1 : tensor<3x2x3x10x1xi64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.iota dim = 2 : tensor<3x2x3x10x1xi64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.iota dim = 3 : tensor<3x2x3x10x1xi64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.multiply %[[V4]], %[[V2]] : tensor<3x2x3x10x1xi64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.add %[[V6]], %[[V3]] : tensor<3x2x3x10x1xi64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.multiply %[[V7]], %[[V1]] : tensor<3x2x3x10x1xi64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.add %[[V8]], %[[V5]] : tensor<3x2x3x10x1xi64>
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = "stablehlo.gather"(%[[V0]], %[[V9]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 4>, slice_sizes = array<i64: 1, 1>}> : (tensor<3x48xf64>, tensor<3x2x3x10x1xi64>) -> tensor<3x2x3x10xf64>
// CHECK-NEXT: return %[[V10]] : tensor<3x2x3x10xf64>
// CHECK-NEXT: }
