// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=reduce_broadcast_source;select_broadcast_constants;pad_broadcast_constant(4096)" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=reduce_broadcast_source;select_broadcast_constants;pad_broadcast_constant(4096);reduce_const_prop<16>;select_op_canon<16>(1024);broadcast_in_dim_op_canon<16>;broadcast_in_dim_simplify<16>(1024);slice_broadcast;scatter_masked_index_slice;scatter_op_canon<16>" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s --check-prefix=FOLD

// An or over a dimension the broadcast's operand has reduces the operand;
// the batch dimension only repeats its values.
func.func @reduce_source(%m: tensor<2x4xi1>) -> tensor<19652x4xi1> {
  %f = stablehlo.constant dense<false> : tensor<i1>
  %b = stablehlo.broadcast_in_dim %m, dims = [2, 1] : (tensor<2x4xi1>) -> tensor<19652x4x2xi1>
  %r = stablehlo.reduce(%b init: %f) applies stablehlo.or across dimensions = [2] : (tensor<19652x4x2xi1>, tensor<i1>) -> tensor<19652x4xi1>
  return %r : tensor<19652x4xi1>
}

// CHECK:    func.func @reduce_source(%arg0: tensor<2x4xi1>) -> tensor<19652x4xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:    %0 = stablehlo.reduce(%arg0 init: %c) applies stablehlo.or across dimensions = [0] : (tensor<2x4xi1>, tensor<i1>) -> tensor<4xi1>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<4xi1>) -> tensor<19652x4xi1>
// CHECK-NEXT:    return %1 : tensor<19652x4xi1>
// CHECK-NEXT:  }

// FOLD:    func.func @reduce_source(%arg0: tensor<2x4xi1>) -> tensor<19652x4xi1> {
// FOLD-NEXT:    %c = stablehlo.constant dense<false> : tensor<i1>
// FOLD-NEXT:    %0 = stablehlo.reduce(%arg0 init: %c) applies stablehlo.or across dimensions = [0] : (tensor<2x4xi1>, tensor<i1>) -> tensor<4xi1>
// FOLD-NEXT:    %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<4xi1>) -> tensor<19652x4xi1>
// FOLD-NEXT:    return %1 : tensor<19652x4xi1>
// FOLD-NEXT:  }

// -----

// A reduce over a dimension the broadcast made is left to broadcast_reduce.
func.func @reduce_new_dim(%m: tensor<2x4xi1>) -> tensor<2x4xi1> {
  %f = stablehlo.constant dense<false> : tensor<i1>
  %b = stablehlo.broadcast_in_dim %m, dims = [1, 2] : (tensor<2x4xi1>) -> tensor<5x2x4xi1>
  %r = stablehlo.reduce(%b init: %f) applies stablehlo.or across dimensions = [0] : (tensor<5x2x4xi1>, tensor<i1>) -> tensor<2x4xi1>
  return %r : tensor<2x4xi1>
}

// CHECK:    func.func @reduce_new_dim(%arg0: tensor<2x4xi1>) -> tensor<2x4xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] : (tensor<2x4xi1>) -> tensor<5x2x4xi1>
// CHECK-NEXT:    %1 = stablehlo.reduce(%0 init: %c) applies stablehlo.or across dimensions = [0] : (tensor<5x2x4xi1>, tensor<i1>) -> tensor<2x4xi1>
// CHECK-NEXT:    return %1 : tensor<2x4xi1>
// CHECK-NEXT:  }

// FOLD:    func.func @reduce_new_dim(%arg0: tensor<2x4xi1>) -> tensor<2x4xi1> {
// FOLD-NEXT:    %c = stablehlo.constant dense<false> : tensor<i1>
// FOLD-NEXT:    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] : (tensor<2x4xi1>) -> tensor<5x2x4xi1>
// FOLD-NEXT:    %1 = stablehlo.reduce(%0 init: %c) applies stablehlo.or across dimensions = [0] : (tensor<5x2x4xi1>, tensor<i1>) -> tensor<2x4xi1>
// FOLD-NEXT:    return %1 : tensor<2x4xi1>
// FOLD-NEXT:  }

// -----

// The mask, the table and the splat vary along the table's dimensions only:
// the select is made there, and folds.
func.func @select_constants() -> tensor<19652x4x4x1xi64> {
  %c = stablehlo.constant dense<-1> : tensor<19652x4x4x1xi64>
  %t = stablehlo.constant dense<[[[0], [3], [6], [9]], [[1], [4], [7], [10]], [[2], [5], [8], [11]], [[3], [6], [9], [12]]]> : tensor<4x4x1xi64>
  %m = stablehlo.constant dense<[true, true, true, false]> : tensor<4xi1>
  %bm = stablehlo.broadcast_in_dim %m, dims = [1] : (tensor<4xi1>) -> tensor<19652x4x4x1xi1>
  %bt = stablehlo.broadcast_in_dim %t, dims = [1, 2, 3] : (tensor<4x4x1xi64>) -> tensor<19652x4x4x1xi64>
  %s = stablehlo.select %bm, %bt, %c : tensor<19652x4x4x1xi1>, tensor<19652x4x4x1xi64>
  return %s : tensor<19652x4x4x1xi64>
}

// CHECK:    func.func @select_constants() -> tensor<19652x4x4x1xi64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<{{\[\[}}[0], [3], [6], [9]], {{\[\[}}1], [4], [7], [10]], {{\[\[}}2], [5], [8], [11]], {{\[\[}}3], [6], [9], [12]]]> : tensor<4x4x1xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<[true, true, true, false]> : tensor<4xi1>
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %c_1, dims = [0] : (tensor<4xi1>) -> tensor<4x4x1xi1>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %c_0, dims = [0, 1, 2] : (tensor<4x4x1xi64>) -> tensor<4x4x1xi64>
// CHECK-NEXT:    %2 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<4x4x1xi64>
// CHECK-NEXT:    %3 = stablehlo.select %0, %1, %2 : tensor<4x4x1xi1>, tensor<4x4x1xi64>
// CHECK-NEXT:    %4 = stablehlo.broadcast_in_dim %3, dims = [1, 2, 3] : (tensor<4x4x1xi64>) -> tensor<19652x4x4x1xi64>
// CHECK-NEXT:    return %4 : tensor<19652x4x4x1xi64>
// CHECK-NEXT:  }

// FOLD:    func.func @select_constants() -> tensor<19652x4x4x1xi64> {
// FOLD-NEXT:    %c = stablehlo.constant dense<{{\[\[}}[0], [3], [6], [9]], {{\[\[}}1], [4], [7], [10]], {{\[\[}}2], [5], [8], [11]], {{\[\[}}-1], [-1], [-1], [-1]]]> : tensor<4x4x1xi64>
// FOLD-NEXT:    %0 = stablehlo.broadcast_in_dim %c, dims = [1, 2, 3] : (tensor<4x4x1xi64>) -> tensor<19652x4x4x1xi64>
// FOLD-NEXT:    return %0 : tensor<19652x4x4x1xi64>
// FOLD-NEXT:  }

// -----

// The masked store of a raised kernel: the rows of the table the mask (an
// or over a constant) drops are out of bounds, the rest write a + 3 b for
// a < 3, b < 4 into each row of 12: a dynamic update slice of the rows
// viewed as [4, 3], over the batch.
func.func @masked_scatter(%buf: tensor<19652x12xf64>, %upd: tensor<19652x4x4xf64>) -> tensor<19652x12xf64> {
  %c = stablehlo.constant dense<-1> : tensor<19652x4x4x1xi64>
  %t = stablehlo.constant dense<[[[0], [3], [6], [9]], [[1], [4], [7], [10]], [[2], [5], [8], [11]], [[3], [6], [9], [12]]]> : tensor<4x4x1xi64>
  %f = stablehlo.constant dense<false> : tensor<i1>
  %m = stablehlo.constant dense<[[true, true, true, false], [true, true, true, false]]> : tensor<2x4xi1>
  %0 = stablehlo.broadcast_in_dim %m, dims = [2, 1] : (tensor<2x4xi1>) -> tensor<19652x4x2xi1>
  %1 = stablehlo.reduce(%0 init: %f) applies stablehlo.or across dimensions = [2] : (tensor<19652x4x2xi1>, tensor<i1>) -> tensor<19652x4xi1>
  %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<19652x4xi1>) -> tensor<19652x4x4x1xi1>
  %3 = stablehlo.broadcast_in_dim %t, dims = [1, 2, 3] : (tensor<4x4x1xi64>) -> tensor<19652x4x4x1xi64>
  %4 = stablehlo.select %2, %3, %c : tensor<19652x4x4x1xi1>, tensor<19652x4x4x1xi64>
  %5 = "stablehlo.scatter"(%buf, %4, %upd) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 3>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<19652x12xf64>, tensor<19652x4x4x1xi64>, tensor<19652x4x4xf64>) -> tensor<19652x12xf64>
  return %5 : tensor<19652x12xf64>
}

// CHECK:    func.func @masked_scatter(%arg0: tensor<19652x12xf64>, %arg1: tensor<19652x4x4xf64>) -> tensor<19652x12xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<-1> : tensor<19652x4x4x1xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<{{\[\[}}[0], [3], [6], [9]], {{\[\[}}1], [4], [7], [10]], {{\[\[}}2], [5], [8], [11]], {{\[\[}}3], [6], [9], [12]]]> : tensor<4x4x1xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<{{\[\[}}true, true, true, false], [true, true, true, false]]> : tensor<2x4xi1>
// CHECK-NEXT:    %0 = stablehlo.reduce(%c_2 init: %c_1) applies stablehlo.or across dimensions = [0] : (tensor<2x4xi1>, tensor<i1>) -> tensor<4xi1>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<4xi1>) -> tensor<19652x4xi1>
// CHECK-NEXT:    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<19652x4xi1>) -> tensor<19652x4x4x1xi1>
// CHECK-NEXT:    %3 = stablehlo.broadcast_in_dim %c_0, dims = [1, 2, 3] : (tensor<4x4x1xi64>) -> tensor<19652x4x4x1xi64>
// CHECK-NEXT:    %4 = stablehlo.select %2, %3, %c : tensor<19652x4x4x1xi1>, tensor<19652x4x4x1xi64>
// CHECK-NEXT:    %5 = "stablehlo.scatter"(%arg0, %4, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 3>, unique_indices = false}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<19652x12xf64>, tensor<19652x4x4x1xi64>, tensor<19652x4x4xf64>) -> tensor<19652x12xf64>
// CHECK-NEXT:    return %5 : tensor<19652x12xf64>
// CHECK-NEXT:  }

// FOLD:    func.func @masked_scatter(%arg0: tensor<19652x12xf64>, %arg1: tensor<19652x4x4xf64>) -> tensor<19652x12xf64> {
// FOLD-NEXT:    %c = stablehlo.constant dense<0> : tensor<i32>
// FOLD-NEXT:    %0 = stablehlo.slice %arg1 [0:19652, 0:3, 0:4] : (tensor<19652x4x4xf64>) -> tensor<19652x3x4xf64>
// FOLD-NEXT:    %1 = stablehlo.reshape %arg0 : (tensor<19652x12xf64>) -> tensor<19652x4x3xf64>
// FOLD-NEXT:    %2 = stablehlo.transpose %0, dims = [0, 2, 1] : (tensor<19652x3x4xf64>) -> tensor<19652x4x3xf64>
// FOLD-NEXT:    %3 = stablehlo.dynamic_update_slice %1, %2, %c, %c, %c : (tensor<19652x4x3xf64>, tensor<19652x4x3xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<19652x4x3xf64>
// FOLD-NEXT:    %4 = stablehlo.reshape %3 : (tensor<19652x4x3xf64>) -> tensor<19652x12xf64>
// FOLD-NEXT:    return %4 : tensor<19652x12xf64>
// FOLD-NEXT:  }

// -----

// The row mask of a guarded store, a pad of a splat over the batch, is the
// canonical form of a mask (compare_iota_const_simplify's): left alone.
func.func @pad_splat_mask() -> tensor<19652x4x4xi1> {
  %f = stablehlo.constant dense<false> : tensor<19652x1x4xi1>
  %t = stablehlo.constant dense<true> : tensor<i1>
  %m = stablehlo.pad %f, %t, low = [0, 3, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<19652x1x4xi1>, tensor<i1>) -> tensor<19652x4x4xi1>
  return %m : tensor<19652x4x4xi1>
}

// CHECK:    func.func @pad_splat_mask() -> tensor<19652x4x4xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<false> : tensor<19652x1x4xi1>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<true> : tensor<i1>
// CHECK-NEXT:    %0 = stablehlo.pad %c, %c_0, low = [0, 3, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<19652x1x4xi1>, tensor<i1>) -> tensor<19652x4x4xi1>
// CHECK-NEXT:    return %0 : tensor<19652x4x4xi1>
// CHECK-NEXT:  }

// FOLD:    func.func @pad_splat_mask() -> tensor<19652x4x4xi1> {
// FOLD-NEXT:    %c = stablehlo.constant dense<false> : tensor<19652x1x4xi1>
// FOLD-NEXT:    %c_0 = stablehlo.constant dense<true> : tensor<i1>
// FOLD-NEXT:    %0 = stablehlo.pad %c, %c_0, low = [0, 3, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<19652x1x4xi1>, tensor<i1>) -> tensor<19652x4x4xi1>
// FOLD-NEXT:    return %0 : tensor<19652x4x4xi1>
// FOLD-NEXT:  }

// -----

// A table's rows cut to the live ones and padded back with -1, over the
// batch: the broadcast of the padded table.
func.func @pad_broadcast_table() -> tensor<19652x4x4x1xi64> {
  %t = stablehlo.constant dense<[[[0], [3], [6], [9]], [[1], [4], [7], [10]], [[2], [5], [8], [11]]]> : tensor<3x4x1xi64>
  %m1 = stablehlo.constant dense<-1> : tensor<i64>
  %b = stablehlo.broadcast_in_dim %t, dims = [1, 2, 3] : (tensor<3x4x1xi64>) -> tensor<19652x3x4x1xi64>
  %p = stablehlo.pad %b, %m1, low = [0, 0, 0, 0], high = [0, 1, 0, 0], interior = [0, 0, 0, 0] : (tensor<19652x3x4x1xi64>, tensor<i64>) -> tensor<19652x4x4x1xi64>
  return %p : tensor<19652x4x4x1xi64>
}

// CHECK:    func.func @pad_broadcast_table() -> tensor<19652x4x4x1xi64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<{{\[\[}}[0], [3], [6], [9]], {{\[\[}}1], [4], [7], [10]], {{\[\[}}2], [5], [8], [11]], {{\[\[}}-1], [-1], [-1], [-1]]]> : tensor<4x4x1xi64>
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %c, dims = [1, 2, 3] : (tensor<4x4x1xi64>) -> tensor<19652x4x4x1xi64>
// CHECK-NEXT:    return %0 : tensor<19652x4x4x1xi64>
// CHECK-NEXT:  }

// FOLD:    func.func @pad_broadcast_table() -> tensor<19652x4x4x1xi64> {
// FOLD-NEXT:    %c = stablehlo.constant dense<{{\[\[}}[0], [3], [6], [9]], {{\[\[}}1], [4], [7], [10]], {{\[\[}}2], [5], [8], [11]], {{\[\[}}-1], [-1], [-1], [-1]]]> : tensor<4x4x1xi64>
// FOLD-NEXT:    %0 = stablehlo.broadcast_in_dim %c, dims = [1, 2, 3] : (tensor<4x4x1xi64>) -> tensor<19652x4x4x1xi64>
// FOLD-NEXT:    return %0 : tensor<19652x4x4x1xi64>
// FOLD-NEXT:  }

// -----

// A pad along a dimension the broadcast made is left alone.
func.func @pad_new_dim(%x: tensor<3xi64>) -> tensor<5x3xi64> {
  %z = stablehlo.constant dense<0> : tensor<i64>
  %b = stablehlo.broadcast_in_dim %x, dims = [1] : (tensor<3xi64>) -> tensor<4x3xi64>
  %p = stablehlo.pad %b, %z, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<4x3xi64>, tensor<i64>) -> tensor<5x3xi64>
  return %p : tensor<5x3xi64>
}

// CHECK:    func.func @pad_new_dim(%arg0: tensor<3xi64>) -> tensor<5x3xi64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<3xi64>) -> tensor<4x3xi64>
// CHECK-NEXT:    %1 = stablehlo.pad %0, %c, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<4x3xi64>, tensor<i64>) -> tensor<5x3xi64>
// CHECK-NEXT:    return %1 : tensor<5x3xi64>
// CHECK-NEXT:  }

// FOLD:    func.func @pad_new_dim(%arg0: tensor<3xi64>) -> tensor<5x3xi64> {
// FOLD-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// FOLD-NEXT:    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<3xi64>) -> tensor<4x3xi64>
// FOLD-NEXT:    %1 = stablehlo.pad %0, %c, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<4x3xi64>, tensor<i64>) -> tensor<5x3xi64>
// FOLD-NEXT:    return %1 : tensor<5x3xi64>
// FOLD-NEXT:  }

// -----

// The second store of the same kernel: the mask a pad of a splat over the
// batch, read by the select fold as the constant over its padded rows, and
// the table broadcast; the select folds to one constant index, whose dead
// row the masked-index cut drops.
func.func @masked_scatter_padded(%buf: tensor<19652x12xf64>, %upd: tensor<19652x4x4xf64>) -> tensor<19652x12xf64> {
  %f = stablehlo.constant dense<false> : tensor<19652x1x4xi1>
  %t = stablehlo.constant dense<true> : tensor<i1>
  %m1 = stablehlo.constant dense<-1> : tensor<19652x4x4x1xi64>
  %tab = stablehlo.constant dense<[[[0], [3], [6], [9]], [[1], [4], [7], [10]], [[2], [5], [8], [11]], [[3], [6], [9], [12]]]> : tensor<4x4x1xi64>
  %0 = stablehlo.pad %f, %t, low = [0, 3, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<19652x1x4xi1>, tensor<i1>) -> tensor<19652x4x4xi1>
  %1 = stablehlo.reshape %0 : (tensor<19652x4x4xi1>) -> tensor<19652x4x4x1xi1>
  %2 = stablehlo.broadcast_in_dim %tab, dims = [1, 2, 3] : (tensor<4x4x1xi64>) -> tensor<19652x4x4x1xi64>
  %3 = stablehlo.select %1, %2, %m1 : tensor<19652x4x4x1xi1>, tensor<19652x4x4x1xi64>
  %4 = "stablehlo.scatter"(%buf, %3, %upd) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 3>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<19652x12xf64>, tensor<19652x4x4x1xi64>, tensor<19652x4x4xf64>) -> tensor<19652x12xf64>
  return %4 : tensor<19652x12xf64>
}

// CHECK:    func.func @masked_scatter_padded(%arg0: tensor<19652x12xf64>, %arg1: tensor<19652x4x4xf64>) -> tensor<19652x12xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<{{\[\[}}[0], [3], [6], [9]], {{\[\[}}1], [4], [7], [10]], {{\[\[}}2], [5], [8], [11]], {{\[\[}}3], [6], [9], [12]]]> : tensor<4x4x1xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<[true, true, true, false]> : tensor<4xi1>
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %c_1, dims = [0] : (tensor<4xi1>) -> tensor<4x4x1xi1>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %c_0, dims = [0, 1, 2] : (tensor<4x4x1xi64>) -> tensor<4x4x1xi64>
// CHECK-NEXT:    %2 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<4x4x1xi64>
// CHECK-NEXT:    %3 = stablehlo.select %0, %1, %2 : tensor<4x4x1xi1>, tensor<4x4x1xi64>
// CHECK-NEXT:    %4 = stablehlo.broadcast_in_dim %3, dims = [1, 2, 3] : (tensor<4x4x1xi64>) -> tensor<19652x4x4x1xi64>
// CHECK-NEXT:    %5 = "stablehlo.scatter"(%arg0, %4, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 3>, unique_indices = false}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<19652x12xf64>, tensor<19652x4x4x1xi64>, tensor<19652x4x4xf64>) -> tensor<19652x12xf64>
// CHECK-NEXT:    return %5 : tensor<19652x12xf64>
// CHECK-NEXT:  }

// FOLD:    func.func @masked_scatter_padded(%arg0: tensor<19652x12xf64>, %arg1: tensor<19652x4x4xf64>) -> tensor<19652x12xf64> {
// FOLD-NEXT:    %c = stablehlo.constant dense<{{\[\[}}[0], [3], [6], [9]], {{\[\[}}1], [4], [7], [10]], {{\[\[}}2], [5], [8], [11]], {{\[\[}}-1], [-1], [-1], [-1]]]> : tensor<4x4x1xi64>
// FOLD-NEXT:    %0 = stablehlo.slice %c [0:3, 0:4, 0:1] : (tensor<4x4x1xi64>) -> tensor<3x4x1xi64>
// FOLD-NEXT:    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2, 3] : (tensor<3x4x1xi64>) -> tensor<19652x3x4x1xi64>
// FOLD-NEXT:    %2 = stablehlo.slice %arg1 [0:19652, 0:3, 0:4] : (tensor<19652x4x4xf64>) -> tensor<19652x3x4xf64>
// FOLD-NEXT:    %3 = "stablehlo.scatter"(%arg0, %1, %2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 3>, unique_indices = false}> ({
// FOLD-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// FOLD-NEXT:      stablehlo.return %arg3 : tensor<f64>
// FOLD-NEXT:    }) : (tensor<19652x12xf64>, tensor<19652x3x4x1xi64>, tensor<19652x3x4xf64>) -> tensor<19652x12xf64>
// FOLD-NEXT:    return %3 : tensor<19652x12xf64>
// FOLD-NEXT:  }
