// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt --split-input-file | FileCheck %s

// A scatter DROPS an out-of-bounds update while dynamic-update-slice CLAMPS
// the start, so the single-index scatter-to-DUS conversion fires only when
// the index provably lands in bounds.

// A dead-lane scatter sends its index out of bounds through a negative
// select arm: stays a scatter.

module {
  func.func @masked_single_scatter(%buf: tensor<12xf64>, %idx: tensor<i64>, %pred: tensor<i1>, %val: tensor<f64>) -> tensor<12xf64> {
    %cm1 = stablehlo.constant dense<-1> : tensor<i64>
    %sel = stablehlo.select %pred, %idx, %cm1 : tensor<i1>, tensor<i64>
    %indices = stablehlo.reshape %sel : (tensor<i64>) -> tensor<1xi64>
    %r = "stablehlo.scatter"(%buf, %indices, %val) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = false}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      stablehlo.return %arg1 : tensor<f64>
    }) : (tensor<12xf64>, tensor<1xi64>, tensor<f64>) -> tensor<12xf64>
    return %r : tensor<12xf64>
  }
}

// CHECK:  func.func @masked_single_scatter(%arg0: tensor<12xf64>, %arg1: tensor<i64>, %arg2: tensor<i1>, %arg3: tensor<f64>) -> tensor<12xf64> {
// CHECK-NEXT:    %0 = stablehlo.reshape %arg1 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:    %1 = stablehlo.dynamic_slice %arg0, %arg1, sizes = [1] : (tensor<12xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:    %2 = stablehlo.reshape %1 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:    %3 = stablehlo.select %arg2, %arg3, %2 : tensor<i1>, tensor<f64>
// CHECK-NEXT:    %4 = "stablehlo.scatter"(%arg0, %0, %3) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg4: tensor<f64>, %arg5: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg5 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<12xf64>, tensor<1xi64>, tensor<f64>) -> tensor<12xf64>
// CHECK-NEXT:    return %4 : tensor<12xf64>
// CHECK-NEXT:  }

// -----


// An arbitrary dynamic index proves nothing: stays a scatter.

module {
  func.func @unknown_single_scatter(%buf: tensor<12xf64>, %idx: tensor<1xi64>, %val: tensor<f64>) -> tensor<12xf64> {
    %r = "stablehlo.scatter"(%buf, %idx, %val) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = false}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      stablehlo.return %arg1 : tensor<f64>
    }) : (tensor<12xf64>, tensor<1xi64>, tensor<f64>) -> tensor<12xf64>
    return %r : tensor<12xf64>
  }
}

// CHECK:  func.func @unknown_single_scatter(%arg0: tensor<12xf64>, %arg1: tensor<1xi64>, %arg2: tensor<f64>) -> tensor<12xf64> {
// CHECK-NEXT:    %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg4 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<12xf64>, tensor<1xi64>, tensor<f64>) -> tensor<12xf64>
// CHECK-NEXT:    return %0 : tensor<12xf64>
// CHECK-NEXT:  }

// -----


// A clamp proves the bounds: converts to a dynamic-update-slice.

module {
  func.func @clamped_single_scatter(%buf: tensor<12xf64>, %idx: tensor<i64>, %val: tensor<f64>) -> tensor<12xf64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c11 = stablehlo.constant dense<11> : tensor<i64>
    %cl = stablehlo.clamp %c0, %idx, %c11 : tensor<i64>
    %indices = stablehlo.reshape %cl : (tensor<i64>) -> tensor<1xi64>
    %r = "stablehlo.scatter"(%buf, %indices, %val) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = false}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      stablehlo.return %arg1 : tensor<f64>
    }) : (tensor<12xf64>, tensor<1xi64>, tensor<f64>) -> tensor<12xf64>
    return %r : tensor<12xf64>
  }
}

// CHECK:  func.func @clamped_single_scatter(%arg0: tensor<12xf64>, %arg1: tensor<i64>, %arg2: tensor<f64>) -> tensor<12xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<11> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.clamp %c, %arg1, %c_0 : tensor<i64>
// CHECK-NEXT:    %1 = stablehlo.reshape %arg2 : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:    %2 = stablehlo.dynamic_update_slice %arg0, %1, %0 : (tensor<12xf64>, tensor<1xf64>, tensor<i64>) -> tensor<12xf64>
// CHECK-NEXT:    return %2 : tensor<12xf64>
// CHECK-NEXT:  }

// -----


// A constant index inside the buffer converts; one past the end stays.

module {
  func.func @const_single_scatter(%buf: tensor<12xf64>, %val: tensor<f64>) -> tensor<12xf64> {
    %indices = stablehlo.constant dense<7> : tensor<1xi64>
    %r = "stablehlo.scatter"(%buf, %indices, %val) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = false}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      stablehlo.return %arg1 : tensor<f64>
    }) : (tensor<12xf64>, tensor<1xi64>, tensor<f64>) -> tensor<12xf64>
    return %r : tensor<12xf64>
  }
}

// CHECK:  func.func @const_single_scatter(%arg0: tensor<12xf64>, %arg1: tensor<f64>) -> tensor<12xf64> {
// CHECK-NEXT:    %0 = stablehlo.reshape %arg1 : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [0:7] : (tensor<12xf64>) -> tensor<7xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %arg0 [8:12] : (tensor<12xf64>) -> tensor<4xf64>
// CHECK-NEXT:    %3 = stablehlo.concatenate %1, %0, %2, dim = 0 : (tensor<7xf64>, tensor<1xf64>, tensor<4xf64>) -> tensor<12xf64>
// CHECK-NEXT:    return %3 : tensor<12xf64>
// CHECK-NEXT:  }

// -----


module {
  func.func @const_oob_single_scatter(%buf: tensor<12xf64>, %val: tensor<f64>) -> tensor<12xf64> {
    %indices = stablehlo.constant dense<12> : tensor<1xi64>
    %r = "stablehlo.scatter"(%buf, %indices, %val) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = false}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      stablehlo.return %arg1 : tensor<f64>
    }) : (tensor<12xf64>, tensor<1xi64>, tensor<f64>) -> tensor<12xf64>
    return %r : tensor<12xf64>
  }
}

// CHECK:  func.func @const_oob_single_scatter(%arg0: tensor<12xf64>, %arg1: tensor<f64>) -> tensor<12xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<12> : tensor<1xi64>
// CHECK-NEXT:    %0 = "stablehlo.scatter"(%arg0, %c, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<12xf64>, tensor<1xi64>, tensor<f64>) -> tensor<12xf64>
// CHECK-NEXT:    return %0 : tensor<12xf64>
// CHECK-NEXT:  }

// -----


// A widening convert of a small unsigned type is provably in bounds.

module {
  func.func @narrow_unsigned_single_scatter(%buf: tensor<300xf64>, %idx: tensor<ui8>, %val: tensor<f64>) -> tensor<300xf64> {
    %cvt = stablehlo.convert %idx : (tensor<ui8>) -> tensor<i64>
    %indices = stablehlo.reshape %cvt : (tensor<i64>) -> tensor<1xi64>
    %r = "stablehlo.scatter"(%buf, %indices, %val) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = false}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      stablehlo.return %arg1 : tensor<f64>
    }) : (tensor<300xf64>, tensor<1xi64>, tensor<f64>) -> tensor<300xf64>
    return %r : tensor<300xf64>
  }
}

// CHECK:  func.func @narrow_unsigned_single_scatter(%arg0: tensor<300xf64>, %arg1: tensor<ui8>, %arg2: tensor<f64>) -> tensor<300xf64> {
// CHECK-NEXT:    %0 = stablehlo.convert %arg1 : (tensor<ui8>) -> tensor<i64>
// CHECK-NEXT:    %1 = stablehlo.reshape %arg2 : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:    %2 = stablehlo.dynamic_update_slice %arg0, %1, %0 : (tensor<300xf64>, tensor<1xf64>, tensor<i64>) -> tensor<300xf64>
// CHECK-NEXT:    return %2 : tensor<300xf64>
// CHECK-NEXT:  }

// -----


// The iota path bound-checks its constant start the same way: a negative
// start would clamp where the scatter drops.

module {
  func.func @iota_negative_start(%buf: tensor<12xf64>, %val: tensor<4x1xf64>) -> tensor<12xf64> {
    %iota = stablehlo.iota dim = 0 : tensor<4xi64>
    %cm2 = stablehlo.constant dense<-2> : tensor<4xi64>
    %shift = stablehlo.add %iota, %cm2 : tensor<4xi64>
    %indices = stablehlo.reshape %shift : (tensor<4xi64>) -> tensor<4x1xi64>
    %r = "stablehlo.scatter"(%buf, %indices, %val) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      stablehlo.return %arg1 : tensor<f64>
    }) : (tensor<12xf64>, tensor<4x1xi64>, tensor<4x1xf64>) -> tensor<12xf64>
    return %r : tensor<12xf64>
  }
}

// CHECK:  func.func @iota_negative_start(%arg0: tensor<12xf64>, %arg1: tensor<4x1xf64>) -> tensor<12xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<{{\[\[}}-2], [-1], [0], [1]]> : tensor<4x1xi64>
// CHECK-NEXT:    %0 = "stablehlo.scatter"(%arg0, %c, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<12xf64>, tensor<4x1xi64>, tensor<4x1xf64>) -> tensor<12xf64>
// CHECK-NEXT:    return %0 : tensor<12xf64>
// CHECK-NEXT:  }
