// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=scatter_op_canon;gather_op_canon" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// A per-element scratch row of 216 holds a 5x5x5 block at its start: its
// strides 25 and 5 do not divide the row, so the row is cut to the 125 the
// indices reach before it is split.
func.func @gather_cut(%x: tensor<16x3x216xf64>) -> tensor<16x5x5xf64> {
  %c = stablehlo.constant dense<[[[2, 0], [2, 25], [2, 50], [2, 75], [2, 100]], [[2, 5], [2, 30], [2, 55], [2, 80], [2, 105]], [[2, 10], [2, 35], [2, 60], [2, 85], [2, 110]], [[2, 15], [2, 40], [2, 65], [2, 90], [2, 115]], [[2, 20], [2, 45], [2, 70], [2, 95], [2, 120]]]> : tensor<5x5x2xi64>
  %i = stablehlo.broadcast_in_dim %c, dims = [1, 2, 3] : (tensor<5x5x2xi64>) -> tensor<16x5x5x2xi64>
  %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1, 2], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1, 2], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<16x3x216xf64>, tensor<16x5x5x2xi64>) -> tensor<16x5x5xf64>
  return %g : tensor<16x5x5xf64>
}

// CHECK-LABEL: func.func @gather_cut(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<16x3x216xf64>) -> tensor<16x5x5xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<16x3x216xf64>) -> tensor<16x648xf64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.slice %[[V1]] [0:16, 432:557] : (tensor<16x648xf64>) -> tensor<16x125xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reshape %[[V2]] : (tensor<16x125xf64>) -> tensor<16x5x5x5xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.slice %[[V3]] [0:16, 0:5, 0:5, 0:1] : (tensor<16x5x5x5xf64>) -> tensor<16x5x5x1xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.reshape %[[V4]] : (tensor<16x5x5x1xf64>) -> tensor<16x5x5xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V5]], dims = [0, 2, 1] : (tensor<16x5x5xf64>) -> tensor<16x5x5xf64>
// CHECK-NEXT: return %[[V6]] : tensor<16x5x5xf64>
// CHECK-NEXT: }

// The same block written: the cut is written back into the row.
func.func @scatter_cut(%x: tensor<16x3x216xf64>, %u: tensor<16x5x5x5xf64>) -> tensor<16x3x216xf64> {
  %c = stablehlo.constant dense<[[[[0], [5], [10], [15], [20]], [[25], [30], [35], [40], [45]], [[50], [55], [60], [65], [70]], [[75], [80], [85], [90], [95]], [[100], [105], [110], [115], [120]]], [[[1], [6], [11], [16], [21]], [[26], [31], [36], [41], [46]], [[51], [56], [61], [66], [71]], [[76], [81], [86], [91], [96]], [[101], [106], [111], [116], [121]]], [[[2], [7], [12], [17], [22]], [[27], [32], [37], [42], [47]], [[52], [57], [62], [67], [72]], [[77], [82], [87], [92], [97]], [[102], [107], [112], [117], [122]]], [[[3], [8], [13], [18], [23]], [[28], [33], [38], [43], [48]], [[53], [58], [63], [68], [73]], [[78], [83], [88], [93], [98]], [[103], [108], [113], [118], [123]]], [[[4], [9], [14], [19], [24]], [[29], [34], [39], [44], [49]], [[54], [59], [64], [69], [74]], [[79], [84], [89], [94], [99]], [[104], [109], [114], [119], [124]]]]> : tensor<5x5x5x1xi64>
  %b = stablehlo.broadcast_in_dim %c, dims = [1, 2, 3, 4] : (tensor<5x5x5x1xi64>) -> tensor<16x5x5x5x1xi64>
  %two = stablehlo.constant dense<2> : tensor<16x5x5x5x1xi64>
  %i = stablehlo.concatenate %two, %b, dim = 4 : (tensor<16x5x5x5x1xi64>, tensor<16x5x5x5x1xi64>) -> tensor<16x5x5x5x2xi64>
  %s = "stablehlo.scatter"(%x, %i, %u) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1, 2], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 4>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %v: tensor<f64>):
    stablehlo.return %v : tensor<f64>
  }) : (tensor<16x3x216xf64>, tensor<16x5x5x5x2xi64>, tensor<16x5x5x5xf64>) -> tensor<16x3x216xf64>
  return %s : tensor<16x3x216xf64>
}

// CHECK-LABEL: func.func @scatter_cut(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<16x3x216xf64>, %[[V1:[^ ,:)]+]]: tensor<16x5x5x5xf64>) -> tensor<16x3x216xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<432> : tensor<i32>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<16x3x216xf64>) -> tensor<16x648xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.slice %[[V4]] [0:16, 432:557] : (tensor<16x648xf64>) -> tensor<16x125xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.reshape %[[V5]] : (tensor<16x125xf64>) -> tensor<16x5x5x5xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.transpose %[[V1]], dims = [0, 2, 3, 1] : (tensor<16x5x5x5xf64>) -> tensor<16x5x5x5xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.dynamic_update_slice %[[V6]], %[[V7]], %[[V3]], %[[V3]], %[[V3]], %[[V3]] : (tensor<16x5x5x5xf64>, tensor<16x5x5x5xf64>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<16x5x5x5xf64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.reshape %[[V8]] : (tensor<16x5x5x5xf64>) -> tensor<16x125xf64>
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = stablehlo.dynamic_update_slice %[[V4]], %[[V9]], %[[V3]], %[[V2]] : (tensor<16x648xf64>, tensor<16x125xf64>, tensor<i32>, tensor<i32>) -> tensor<16x648xf64>
// CHECK-NEXT: %[[V11:[^ ,:)]+]] = stablehlo.reshape %[[V10]] : (tensor<16x648xf64>) -> tensor<16x3x216xf64>
// CHECK-NEXT: return %[[V11]] : tensor<16x3x216xf64>
// CHECK-NEXT: }

// The run fits in a row of 123, but not whole blocks of 25 from where it
// starts: left alone.
func.func @gather_cut_past_end(%x: tensor<16x3x123xf64>) -> tensor<16x5x5xf64> {
  %c = stablehlo.constant dense<[[[2, 0], [2, 25], [2, 50], [2, 75], [2, 100]], [[2, 5], [2, 30], [2, 55], [2, 80], [2, 105]], [[2, 10], [2, 35], [2, 60], [2, 85], [2, 110]], [[2, 15], [2, 40], [2, 65], [2, 90], [2, 115]], [[2, 20], [2, 45], [2, 70], [2, 95], [2, 120]]]> : tensor<5x5x2xi64>
  %i = stablehlo.broadcast_in_dim %c, dims = [1, 2, 3] : (tensor<5x5x2xi64>) -> tensor<16x5x5x2xi64>
  %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1, 2], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1, 2], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<16x3x123xf64>, tensor<16x5x5x2xi64>) -> tensor<16x5x5xf64>
  return %g : tensor<16x5x5xf64>
}

// CHECK-LABEL: func.func @gather_cut_past_end(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<16x3x123xf64>) -> tensor<16x5x5xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.constant dense<{{\[\[}}[2, 0], [2, 25], [2, 50], [2, 75], [2, 100]], {{\[\[}}2, 5], [2, 30], [2, 55], [2, 80], [2, 105]], {{\[\[}}2, 10], [2, 35], [2, 60], [2, 85], [2, 110]], {{\[\[}}2, 15], [2, 40], [2, 65], [2, 90], [2, 115]], {{\[\[}}2, 20], [2, 45], [2, 70], [2, 95], [2, 120]]]> : tensor<5x5x2xi64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V1]], dims = [1, 2, 3] : (tensor<5x5x2xi64>) -> tensor<16x5x5x2xi64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = "stablehlo.gather"(%[[V0]], %[[V2]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1, 2], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1, 2], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<16x3x123xf64>, tensor<16x5x5x2xi64>) -> tensor<16x5x5xf64>
// CHECK-NEXT: return %[[V3]] : tensor<16x5x5xf64>
// CHECK-NEXT: }


// A 3x4x4 block at i + 12 j + 3 k in row 2 of a scratch of 3x64: the row
// (at 128) does not start on a multiple of 12, so the 4x12 the block spans
// is cut out of it, from the row's start, before it is split.
func.func @scatter_cut_unaligned(%x: tensor<16x3x64xf64>, %u: tensor<16x3x4x4xf64>) -> tensor<16x3x64xf64> {
  %c = stablehlo.constant dense<[[[[2, 0], [2, 3], [2, 6], [2, 9]], [[2, 12], [2, 15], [2, 18], [2, 21]], [[2, 24], [2, 27], [2, 30], [2, 33]], [[2, 36], [2, 39], [2, 42], [2, 45]]], [[[2, 1], [2, 4], [2, 7], [2, 10]], [[2, 13], [2, 16], [2, 19], [2, 22]], [[2, 25], [2, 28], [2, 31], [2, 34]], [[2, 37], [2, 40], [2, 43], [2, 46]]], [[[2, 2], [2, 5], [2, 8], [2, 11]], [[2, 14], [2, 17], [2, 20], [2, 23]], [[2, 26], [2, 29], [2, 32], [2, 35]], [[2, 38], [2, 41], [2, 44], [2, 47]]]]> : tensor<3x4x4x2xi64>
  %i = stablehlo.broadcast_in_dim %c, dims = [1, 2, 3, 4] : (tensor<3x4x4x2xi64>) -> tensor<16x3x4x4x2xi64>
  %s = "stablehlo.scatter"(%x, %i, %u) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1, 2], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 4>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %v: tensor<f64>):
    stablehlo.return %v : tensor<f64>
  }) : (tensor<16x3x64xf64>, tensor<16x3x4x4x2xi64>, tensor<16x3x4x4xf64>) -> tensor<16x3x64xf64>
  return %s : tensor<16x3x64xf64>
}

// CHECK-LABEL: func.func @scatter_cut_unaligned(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<16x3x64xf64>, %[[V1:[^ ,:)]+]]: tensor<16x3x4x4xf64>) -> tensor<16x3x64xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<128> : tensor<i32>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<16x3x64xf64>) -> tensor<16x192xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.slice %[[V4]] [0:16, 128:176] : (tensor<16x192xf64>) -> tensor<16x48xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.reshape %[[V5]] : (tensor<16x48xf64>) -> tensor<16x4x4x3xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.transpose %[[V1]], dims = [0, 2, 3, 1] : (tensor<16x3x4x4xf64>) -> tensor<16x4x4x3xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.dynamic_update_slice %[[V6]], %[[V7]], %[[V3]], %[[V3]], %[[V3]], %[[V3]] : (tensor<16x4x4x3xf64>, tensor<16x4x4x3xf64>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<16x4x4x3xf64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.reshape %[[V8]] : (tensor<16x4x4x3xf64>) -> tensor<16x48xf64>
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = stablehlo.dynamic_update_slice %[[V4]], %[[V9]], %[[V3]], %[[V2]] : (tensor<16x192xf64>, tensor<16x48xf64>, tensor<i32>, tensor<i32>) -> tensor<16x192xf64>
// CHECK-NEXT: %[[V11:[^ ,:)]+]] = stablehlo.reshape %[[V10]] : (tensor<16x192xf64>) -> tensor<16x3x64xf64>
// CHECK-NEXT: return %[[V11]] : tensor<16x3x64xf64>
// CHECK-NEXT: }
