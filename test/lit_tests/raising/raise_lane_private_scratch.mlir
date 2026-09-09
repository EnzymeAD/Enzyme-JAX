// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo="prefer_while_raising=false err_if_not_fully_raised=true" --split-input-file | FileCheck %s
// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo="prefer_while_raising=false err_if_not_fully_raised=true" --enzyme-hlo-opt --split-input-file | FileCheck %s --check-prefix=OPT

// Scratch allocated inside batched axes is private to each lane: every lane
// writes its own values at the same indices, so the buffer gains one leading
// dimension per axis in scope and every access indexes its lane's copy first.
// Modeled as one shared buffer all lanes would read lane 0's values.

module {
  func.func private @lane(%out: memref<64xf64, 1>, %in: memref<64xf64, 1>) {
    %two = arith.constant 2.0 : f64
    affine.parallel (%e) = (0) to (4) {
      affine.parallel (%t) = (0) to (16) {
        %loc = memref.alloca() : memref<2xf64>
        %v = affine.load %in[%t] : memref<64xf64, 1>
        affine.store %v, %loc[0] : memref<2xf64>
        %v2 = arith.mulf %v, %two : f64
        affine.store %v2, %loc[1] : memref<2xf64>
        %a = affine.load %loc[0] : memref<2xf64>
        %b = affine.load %loc[1] : memref<2xf64>
        %s = arith.addf %a, %b : f64
        affine.store %s, %out[%e * 16 + %t] : memref<64xf64, 1>
      }
    }
    return
  }
}

// CHECK:    func.func private @lane_raised(%arg0: tensor<64xf64>, %arg1: tensor<64xf64>) -> (tensor<64xf64>, tensor<64xf64>) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<4xi64>
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<4xi64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %c : tensor<4xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<4xi64>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %c_0 : tensor<4xi64>
// CHECK-NEXT:    %3 = stablehlo.iota dim = 0 : tensor<16xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<16xi64>
// CHECK-NEXT:    %4 = stablehlo.add %3, %c_1 : tensor<16xi64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<1> : tensor<16xi64>
// CHECK-NEXT:    %5 = stablehlo.multiply %4, %c_2 : tensor<16xi64>
// CHECK-NEXT:    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<4x16x2xf64>
// CHECK-NEXT:    %6 = stablehlo.slice %arg1 [0:16] : (tensor<64xf64>) -> tensor<16xf64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_8 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_9 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_10 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_11 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_12 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_13 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_14 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_15 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_16 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_17 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_18 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_19 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_20 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_21 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %7 = stablehlo.broadcast_in_dim %6, dims = [1] : (tensor<16xf64>) -> tensor<4x16x1xf64>
// CHECK-NEXT:    %8 = stablehlo.dynamic_update_slice %cst_3, %7, %c_9, %c_15, %c_21 : (tensor<4x16x2xf64>, tensor<4x16x1xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<4x16x2xf64>
// CHECK-NEXT:    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<16xf64>
// CHECK-NEXT:    %10 = arith.mulf %6, %9 : tensor<16xf64>
// CHECK-NEXT:    %c_22 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_23 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_24 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_25 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_26 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_27 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_28 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_29 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_30 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_31 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_32 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_33 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_34 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_35 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_36 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_37 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_38 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_39 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %11 = stablehlo.broadcast_in_dim %10, dims = [1] : (tensor<16xf64>) -> tensor<4x16x1xf64>
// CHECK-NEXT:    %12 = stablehlo.dynamic_update_slice %8, %11, %c_27, %c_33, %c_39 : (tensor<4x16x2xf64>, tensor<4x16x1xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<4x16x2xf64>
// CHECK-NEXT:    %13 = stablehlo.slice %12 [0:4, 0:16, 0:1] : (tensor<4x16x2xf64>) -> tensor<4x16x1xf64>
// CHECK-NEXT:    %14 = stablehlo.reshape %13 : (tensor<4x16x1xf64>) -> tensor<4x16xf64>
// CHECK-NEXT:    %15 = stablehlo.slice %12 [0:4, 0:16, 1:2] : (tensor<4x16x2xf64>) -> tensor<4x16x1xf64>
// CHECK-NEXT:    %16 = stablehlo.reshape %15 : (tensor<4x16x1xf64>) -> tensor<4x16xf64>
// CHECK-NEXT:    %17 = arith.addf %14, %16 : tensor<4x16xf64>
// CHECK-NEXT:    %c_40 = stablehlo.constant dense<16> : tensor<i64>
// CHECK-NEXT:    %18 = stablehlo.broadcast_in_dim %c_40, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:    %19 = stablehlo.multiply %2, %18 : tensor<4xi64>
// CHECK-NEXT:    %20 = stablehlo.broadcast_in_dim %19, dims = [0] : (tensor<4xi64>) -> tensor<4x16xi64>
// CHECK-NEXT:    %21 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<16xi64>) -> tensor<4x16xi64>
// CHECK-NEXT:    %22 = stablehlo.add %20, %21 : tensor<4x16xi64>
// CHECK-NEXT:    %23 = stablehlo.reshape %22 : (tensor<4x16xi64>) -> tensor<4x16x1xi64>
// CHECK-NEXT:    %24 = "stablehlo.scatter"(%arg0, %23, %17) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<64xf64>, tensor<4x16x1xi64>, tensor<4x16xf64>) -> tensor<64xf64>
// CHECK-NEXT:    return %24, %arg1 : tensor<64xf64>, tensor<64xf64>
// CHECK-NEXT:  }

// OPT:  module {
// OPT-NEXT:  func.func private @lane_raised(%arg0: tensor<64xf64>, %arg1: tensor<64xf64>) -> (tensor<64xf64>, tensor<64xf64>) {
// OPT-NEXT:    %cst = stablehlo.constant dense<2.000000e+00> : tensor<16xf64>
// OPT-NEXT:    %0 = stablehlo.slice %arg1 [0:16] : (tensor<64xf64>) -> tensor<16xf64>
// OPT-NEXT:    %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<16xf64>) -> tensor<4x16x1xf64>
// OPT-NEXT:    %2 = arith.mulf %0, %cst : tensor<16xf64>
// OPT-NEXT:    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<16xf64>) -> tensor<4x16x1xf64>
// OPT-NEXT:    %4 = stablehlo.reshape %1 : (tensor<4x16x1xf64>) -> tensor<64xf64>
// OPT-NEXT:    %5 = stablehlo.reshape %3 : (tensor<4x16x1xf64>) -> tensor<64xf64>
// OPT-NEXT:    %6 = arith.addf %4, %5 : tensor<64xf64>
// OPT-NEXT:    return %6, %arg1 : tensor<64xf64>, tensor<64xf64>
// OPT-NEXT:  }

// -----
// The same through memref.load/store: the lane induction variables are
// prepended to the indices.

module {
  func.func private @lane_memref(%out: memref<16xf64, 1>, %in: memref<16xf64, 1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    affine.parallel (%t) = (0) to (16) {
      %loc = memref.alloca() : memref<2xf64>
      %v = affine.load %in[%t] : memref<16xf64, 1>
      memref.store %v, %loc[%c0] : memref<2xf64>
      memref.store %v, %loc[%c1] : memref<2xf64>
      %a = memref.load %loc[%c0] : memref<2xf64>
      %b = memref.load %loc[%c1] : memref<2xf64>
      %s = arith.addf %a, %b : f64
      affine.store %s, %out[%t] : memref<16xf64, 1>
    }
    return
  }
}

// CHECK:    func.func private @lane_memref_raised(%arg0: tensor<16xf64>, %arg1: tensor<16xf64>) -> (tensor<16xf64>, tensor<16xf64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<16xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<16xi64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %c_1 : tensor<16xi64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<1> : tensor<16xi64>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %c_2 : tensor<16xi64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<16x2xf64>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<16xi64>) -> tensor<16x1xi64>
// CHECK-NEXT:    %4 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<16x1xi64>
// CHECK-NEXT:    %5 = stablehlo.concatenate %3, %4, dim = 1 : (tensor<16x1xi64>, tensor<16x1xi64>) -> tensor<16x2xi64>
// CHECK-NEXT:    %6 = "stablehlo.scatter"(%cst, %5, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<16x2xf64>, tensor<16x2xi64>, tensor<16xf64>) -> tensor<16x2xf64>
// CHECK-NEXT:    %7 = stablehlo.reshape %2 : (tensor<16xi64>) -> tensor<16x1xi64>
// CHECK-NEXT:    %8 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i64>) -> tensor<16x1xi64>
// CHECK-NEXT:    %9 = stablehlo.concatenate %7, %8, dim = 1 : (tensor<16x1xi64>, tensor<16x1xi64>) -> tensor<16x2xi64>
// CHECK-NEXT:    %10 = "stablehlo.scatter"(%6, %9, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<16x2xf64>, tensor<16x2xi64>, tensor<16xf64>) -> tensor<16x2xf64>
// CHECK-NEXT:    %11 = stablehlo.reshape %2 : (tensor<16xi64>) -> tensor<16x1xi64>
// CHECK-NEXT:    %12 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<16x1xi64>
// CHECK-NEXT:    %13 = stablehlo.concatenate %11, %12, dim = 1 : (tensor<16x1xi64>, tensor<16x1xi64>) -> tensor<16x2xi64>
// CHECK-NEXT:    %14 = "stablehlo.gather"(%10, %13) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<16x2xf64>, tensor<16x2xi64>) -> tensor<16xf64>
// CHECK-NEXT:    %15 = stablehlo.reshape %2 : (tensor<16xi64>) -> tensor<16x1xi64>
// CHECK-NEXT:    %16 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i64>) -> tensor<16x1xi64>
// CHECK-NEXT:    %17 = stablehlo.concatenate %15, %16, dim = 1 : (tensor<16x1xi64>, tensor<16x1xi64>) -> tensor<16x2xi64>
// CHECK-NEXT:    %18 = "stablehlo.gather"(%10, %17) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<16x2xf64>, tensor<16x2xi64>) -> tensor<16xf64>
// CHECK-NEXT:    %19 = arith.addf %14, %18 : tensor<16xf64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_8 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %20 = stablehlo.dynamic_update_slice %arg0, %19, %c_8 : (tensor<16xf64>, tensor<16xf64>, tensor<i64>) -> tensor<16xf64>
// CHECK-NEXT:    return %20, %arg1 : tensor<16xf64>, tensor<16xf64>
// CHECK-NEXT:  }

// OPT:  module {
// OPT-NEXT:  func.func private @lane_memref_raised(%arg0: tensor<16xf64>, %arg1: tensor<16xf64>) -> (tensor<16xf64>, tensor<16xf64>) {
// OPT-NEXT:    %c = stablehlo.constant dense<{{\[\[}}0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [7, 1], [8, 1], [9, 1], [10, 1], [11, 1], [12, 1], [13, 1], [14, 1], [15, 1]]> : tensor<16x2xi64>
// OPT-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<16x2xf64>
// OPT-NEXT:    %c_0 = stablehlo.constant dense<{{\[\[}}0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0], [12, 0], [13, 0], [14, 0], [15, 0]]> : tensor<16x2xi64>
// OPT-NEXT:    %0 = "stablehlo.scatter"(%cst, %c_0, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
// OPT-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// OPT-NEXT:      stablehlo.return %arg3 : tensor<f64>
// OPT-NEXT:    }) : (tensor<16x2xf64>, tensor<16x2xi64>, tensor<16xf64>) -> tensor<16x2xf64>
// OPT-NEXT:    %1 = "stablehlo.scatter"(%0, %c, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
// OPT-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// OPT-NEXT:      stablehlo.return %arg3 : tensor<f64>
// OPT-NEXT:    }) : (tensor<16x2xf64>, tensor<16x2xi64>, tensor<16xf64>) -> tensor<16x2xf64>
// OPT-NEXT:    %2 = "stablehlo.gather"(%1, %c_0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<16x2xf64>, tensor<16x2xi64>) -> tensor<16xf64>
// OPT-NEXT:    %3 = arith.addf %2, %arg1 : tensor<16xf64>
// OPT-NEXT:    return %3, %arg1 : tensor<16xf64>, tensor<16xf64>
// OPT-NEXT:  }

// -----
// Scratch between the two parallels is one copy per outer iteration, shared
// by the inner lanes: it gains only the outer axis, and an inner-lane
// indexed access into it stays a plain index.

module {
  func.func private @block(%out: memref<64xf64, 1>, %in: memref<64xf64, 1>) {
    affine.parallel (%e) = (0) to (4) {
      %sh = memref.alloca() : memref<16xf64>
      affine.parallel (%t) = (0) to (16) {
        %v = affine.load %in[%e * 16 + %t] : memref<64xf64, 1>
        affine.store %v, %sh[%t] : memref<16xf64>
      }
      affine.parallel (%t) = (0) to (16) {
        %a = affine.load %sh[15 - %t] : memref<16xf64>
        affine.store %a, %out[%e * 16 + %t] : memref<64xf64, 1>
      }
    }
    return
  }
}

// CHECK:    func.func private @block_raised(%arg0: tensor<64xf64>, %arg1: tensor<64xf64>) -> (tensor<64xf64>, tensor<64xf64>) {
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<4xi64>
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<4xi64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %c : tensor<4xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<4xi64>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %c_0 : tensor<4xi64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x16xf64>
// CHECK-NEXT:    %3 = stablehlo.iota dim = 0 : tensor<16xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<16xi64>
// CHECK-NEXT:    %4 = stablehlo.add %3, %c_1 : tensor<16xi64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<1> : tensor<16xi64>
// CHECK-NEXT:    %5 = stablehlo.multiply %4, %c_2 : tensor<16xi64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<16> : tensor<i64>
// CHECK-NEXT:    %6 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:    %7 = stablehlo.multiply %2, %6 : tensor<4xi64>
// CHECK-NEXT:    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<4xi64>) -> tensor<4x16xi64>
// CHECK-NEXT:    %9 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<16xi64>) -> tensor<4x16xi64>
// CHECK-NEXT:    %10 = stablehlo.add %8, %9 : tensor<4x16xi64>
// CHECK-NEXT:    %11 = stablehlo.reshape %10 : (tensor<4x16xi64>) -> tensor<4x16x1xi64>
// CHECK-NEXT:    %12 = "stablehlo.gather"(%arg1, %11) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<64xf64>, tensor<4x16x1xi64>) -> tensor<4x16xf64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_8 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_9 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_10 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_11 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_12 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_13 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_14 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_15 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %13 = stablehlo.dynamic_update_slice %cst, %12, %c_9, %c_15 : (tensor<4x16xf64>, tensor<4x16xf64>, tensor<i64>, tensor<i64>) -> tensor<4x16xf64>
// CHECK-NEXT:    %14 = stablehlo.iota dim = 0 : tensor<16xi64>
// CHECK-NEXT:    %c_16 = stablehlo.constant dense<0> : tensor<16xi64>
// CHECK-NEXT:    %15 = stablehlo.add %14, %c_16 : tensor<16xi64>
// CHECK-NEXT:    %c_17 = stablehlo.constant dense<1> : tensor<16xi64>
// CHECK-NEXT:    %16 = stablehlo.multiply %15, %c_17 : tensor<16xi64>
// CHECK-NEXT:    %17 = stablehlo.reverse %13, dims = [1] : tensor<4x16xf64>
// CHECK-NEXT:    %c_18 = stablehlo.constant dense<16> : tensor<i64>
// CHECK-NEXT:    %18 = stablehlo.broadcast_in_dim %c_18, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:    %19 = stablehlo.multiply %2, %18 : tensor<4xi64>
// CHECK-NEXT:    %20 = stablehlo.broadcast_in_dim %19, dims = [0] : (tensor<4xi64>) -> tensor<4x16xi64>
// CHECK-NEXT:    %21 = stablehlo.broadcast_in_dim %16, dims = [1] : (tensor<16xi64>) -> tensor<4x16xi64>
// CHECK-NEXT:    %22 = stablehlo.add %20, %21 : tensor<4x16xi64>
// CHECK-NEXT:    %23 = stablehlo.reshape %22 : (tensor<4x16xi64>) -> tensor<4x16x1xi64>
// CHECK-NEXT:    %24 = "stablehlo.scatter"(%arg0, %23, %17) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<64xf64>, tensor<4x16x1xi64>, tensor<4x16xf64>) -> tensor<64xf64>
// CHECK-NEXT:    return %24, %arg1 : tensor<64xf64>, tensor<64xf64>
// CHECK-NEXT:  }

// OPT:  module {
// OPT-NEXT:  func.func private @block_raised(%arg0: tensor<64xf64>, %arg1: tensor<64xf64>) -> (tensor<64xf64>, tensor<64xf64>) {
// OPT-NEXT:    %0 = stablehlo.reshape %arg1 : (tensor<64xf64>) -> tensor<4x16xf64>
// OPT-NEXT:    %1 = stablehlo.reverse %0, dims = [1] : tensor<4x16xf64>
// OPT-NEXT:    %2 = stablehlo.reshape %1 : (tensor<4x16xf64>) -> tensor<64xf64>
// OPT-NEXT:    return %2, %arg1 : tensor<64xf64>, tensor<64xf64>
// OPT-NEXT:  }

// -----
// An atomic add into lane-private scratch accumulates per lane: the lane
// induction variable joins the scatter indices.

module {
  func.func private @lane_atomic(%out: memref<16xf64, 1>, %in: memref<16xf64, 1>) {
    %c0 = arith.constant 0 : index
    affine.parallel (%t) = (0) to (16) {
      %loc = memref.alloca() : memref<1xf64>
      %v = affine.load %in[%t] : memref<16xf64, 1>
      %r0 = memref.atomic_rmw addf %v, %loc[%c0] : (f64, memref<1xf64>) -> f64
      %r1 = memref.atomic_rmw addf %v, %loc[%c0] : (f64, memref<1xf64>) -> f64
      %a = affine.load %loc[0] : memref<1xf64>
      affine.store %a, %out[%t] : memref<16xf64, 1>
    }
    return
  }
}

// CHECK:    func.func private @lane_atomic_raised(%arg0: tensor<16xf64>, %arg1: tensor<16xf64>) -> (tensor<16xf64>, tensor<16xf64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<16xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<16xi64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %c_0 : tensor<16xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<1> : tensor<16xi64>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %c_1 : tensor<16xi64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<16x1xf64>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<16xi64>) -> tensor<16x1xi64>
// CHECK-NEXT:    %4 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<16x1xi64>
// CHECK-NEXT:    %5 = stablehlo.concatenate %3, %4, dim = 1 : (tensor<16x1xi64>, tensor<16x1xi64>) -> tensor<16x2xi64>
// CHECK-NEXT:    %6 = "stablehlo.scatter"(%cst, %5, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      %13 = stablehlo.add %arg2, %arg3 : tensor<f64>
// CHECK-NEXT:      stablehlo.return %13 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<16x1xf64>, tensor<16x2xi64>, tensor<16xf64>) -> tensor<16x1xf64>
// CHECK-NEXT:    %7 = stablehlo.reshape %2 : (tensor<16xi64>) -> tensor<16x1xi64>
// CHECK-NEXT:    %8 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<16x1xi64>
// CHECK-NEXT:    %9 = stablehlo.concatenate %7, %8, dim = 1 : (tensor<16x1xi64>, tensor<16x1xi64>) -> tensor<16x2xi64>
// CHECK-NEXT:    %10 = "stablehlo.scatter"(%6, %9, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      %13 = stablehlo.add %arg2, %arg3 : tensor<f64>
// CHECK-NEXT:      stablehlo.return %13 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<16x1xf64>, tensor<16x2xi64>, tensor<16xf64>) -> tensor<16x1xf64>
// CHECK-NEXT:    %11 = stablehlo.reshape %10 : (tensor<16x1xf64>) -> tensor<16xf64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %12 = stablehlo.dynamic_update_slice %arg0, %11, %c_7 : (tensor<16xf64>, tensor<16xf64>, tensor<i64>) -> tensor<16xf64>
// CHECK-NEXT:    return %12, %arg1 : tensor<16xf64>, tensor<16xf64>
// CHECK-NEXT:  }

// OPT:  module {
// OPT-NEXT:  func.func private @lane_atomic_raised(%arg0: tensor<16xf64>, %arg1: tensor<16xf64>) -> (tensor<16xf64>, tensor<16xf64>) {
// OPT-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<16x1xf64>
// OPT-NEXT:    %c = stablehlo.constant dense<{{\[\[}}0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0], [12, 0], [13, 0], [14, 0], [15, 0]]> : tensor<16x2xi64>
// OPT-NEXT:    %0 = "stablehlo.scatter"(%cst, %c, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
// OPT-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// OPT-NEXT:      stablehlo.return %arg3 : tensor<f64>
// OPT-NEXT:    }) : (tensor<16x1xf64>, tensor<16x2xi64>, tensor<16xf64>) -> tensor<16x1xf64>
// OPT-NEXT:    %1 = "stablehlo.scatter"(%0, %c, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
// OPT-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// OPT-NEXT:      %3 = stablehlo.add %arg2, %arg3 : tensor<f64>
// OPT-NEXT:      stablehlo.return %3 : tensor<f64>
// OPT-NEXT:    }) : (tensor<16x1xf64>, tensor<16x2xi64>, tensor<16xf64>) -> tensor<16x1xf64>
// OPT-NEXT:    %2 = stablehlo.reshape %1 : (tensor<16x1xf64>) -> tensor<16xf64>
// OPT-NEXT:    return %2, %arg1 : tensor<16xf64>, tensor<16xf64>
// OPT-NEXT:  }
