// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=reshape_gather" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// A unit dimension added to an element-wise gather: added to its index grid.
func.func @unit_dim(%x: tensor<810xf64>, %i: tensor<10x4x4x1xi64>) -> tensor<10x4x4x1xf64> {
  %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<810xf64>, tensor<10x4x4x1xi64>) -> tensor<10x4x4xf64>
  %r = stablehlo.reshape %g : (tensor<10x4x4xf64>) -> tensor<10x4x4x1xf64>
  return %r : tensor<10x4x4x1xf64>
}

// CHECK-LABEL: func.func @unit_dim(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<810xf64>, %[[V1:[^ ,:)]+]]: tensor<10x4x4x1xi64>) -> tensor<10x4x4x1xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<10x4x4x1xi64>) -> tensor<10x4x4x1x1xi64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = "stablehlo.gather"(%[[V0]], %[[V2]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 4>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<810xf64>, tensor<10x4x4x1x1xi64>) -> tensor<10x4x4x1xf64>
// CHECK-NEXT: return %[[V3]] : tensor<10x4x4x1xf64>
// CHECK-NEXT: }

// Grid dimensions merged.
func.func @merge(%x: tensor<810xf64>, %i: tensor<10x4x1xi64>) -> tensor<40xf64> {
  %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<810xf64>, tensor<10x4x1xi64>) -> tensor<10x4xf64>
  %r = stablehlo.reshape %g : (tensor<10x4xf64>) -> tensor<40xf64>
  return %r : tensor<40xf64>
}

// CHECK-LABEL: func.func @merge(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<810xf64>, %[[V1:[^ ,:)]+]]: tensor<10x4x1xi64>) -> tensor<40xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<10x4x1xi64>) -> tensor<40x1xi64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = "stablehlo.gather"(%[[V0]], %[[V2]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<810xf64>, tensor<40x1xi64>) -> tensor<40xf64>
// CHECK-NEXT: return %[[V3]] : tensor<40xf64>
// CHECK-NEXT: }

// The index vector is implicit (index_vector_dim is the rank): made explicit.
func.func @implicit_vector(%x: tensor<810xf64>, %i: tensor<10x4xi64>) -> tensor<40xf64> {
  %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<810xf64>, tensor<10x4xi64>) -> tensor<10x4xf64>
  %r = stablehlo.reshape %g : (tensor<10x4xf64>) -> tensor<40xf64>
  return %r : tensor<40xf64>
}

// CHECK-LABEL: func.func @implicit_vector(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<810xf64>, %[[V1:[^ ,:)]+]]: tensor<10x4xi64>) -> tensor<40xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<10x4xi64>) -> tensor<40x1xi64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = "stablehlo.gather"(%[[V0]], %[[V2]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<810xf64>, tensor<40x1xi64>) -> tensor<40xf64>
// CHECK-NEXT: return %[[V3]] : tensor<40xf64>
// CHECK-NEXT: }

// Each index reads a window of 3 (an offset dimension): left alone.
func.func @window(%x: tensor<810xf64>, %i: tensor<10x1xi64>) -> tensor<30xf64> {
  %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 3>}> : (tensor<810xf64>, tensor<10x1xi64>) -> tensor<10x3xf64>
  %r = stablehlo.reshape %g : (tensor<10x3xf64>) -> tensor<30xf64>
  return %r : tensor<30xf64>
}

// CHECK-LABEL: func.func @window(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<810xf64>, %[[V1:[^ ,:)]+]]: tensor<10x1xi64>) -> tensor<30xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = "stablehlo.gather"(%[[V0]], %[[V1]]) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 3>}> : (tensor<810xf64>, tensor<10x1xi64>) -> tensor<10x3xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reshape %[[V2]] : (tensor<10x3xf64>) -> tensor<30xf64>
// CHECK-NEXT: return %[[V3]] : tensor<30xf64>
// CHECK-NEXT: }

// The gather is also used as it is: left alone.
func.func @other_user(%x: tensor<810xf64>, %i: tensor<10x4x1xi64>) -> (tensor<40xf64>, tensor<10x4xf64>) {
  %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<810xf64>, tensor<10x4x1xi64>) -> tensor<10x4xf64>
  %r = stablehlo.reshape %g : (tensor<10x4xf64>) -> tensor<40xf64>
  return %r, %g : tensor<40xf64>, tensor<10x4xf64>
}

// CHECK-LABEL: func.func @other_user(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<810xf64>, %[[V1:[^ ,:)]+]]: tensor<10x4x1xi64>) -> (tensor<40xf64>, tensor<10x4xf64>) {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = "stablehlo.gather"(%[[V0]], %[[V1]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<810xf64>, tensor<10x4x1xi64>) -> tensor<10x4xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reshape %[[V2]] : (tensor<10x4xf64>) -> tensor<40xf64>
// CHECK-NEXT: return %[[V3]], %[[V2]] : tensor<40xf64>, tensor<10x4xf64>
// CHECK-NEXT: }
