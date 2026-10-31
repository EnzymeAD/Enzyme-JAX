// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=gather_grid_unroll;gather_op_canon<16>" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// Each element e of a vector of 8 elements of 4 rows of 16 reads rows
// 4 e + 3 r + c (r < 2, c < 3: six rows, two of them the next element's) at
// columns 4 k + m: the grid 64 e + 48 r + 16 c + 4 k + m does not factor,
// the run of r and c crossing the blocks of 64. Unrolled over r, the piece
// r = 0 is a slice of the vector as [8, 16, 4]; the piece r = 1 crosses too
// and is unrolled over c, where each piece starts in the element's block or
// the next one's.
func.func @overlapping_rows(%x: tensor<512xf64>) -> tensor<6x2x3x3x3xf64> {
  %c64 = stablehlo.constant dense<64> : tensor<6xi32>
  %c4 = stablehlo.constant dense<4> : tensor<6x2x3xi32>
  %rc = stablehlo.constant dense<[[0, 1, 2], [3, 4, 5]]> : tensor<2x3xi32>
  %c4b = stablehlo.constant dense<4> : tensor<6x2x3x3xi32>
  %e = stablehlo.iota dim = 0 : tensor<6xi32>
  %0 = stablehlo.multiply %e, %c64 : tensor<6xi32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<6xi32>) -> tensor<6x2x3xi32>
  %2 = stablehlo.broadcast_in_dim %rc, dims = [1, 2] : (tensor<2x3xi32>) -> tensor<6x2x3xi32>
  %3 = stablehlo.multiply %2, %c4 : tensor<6x2x3xi32>
  %4 = stablehlo.multiply %3, %c4 : tensor<6x2x3xi32>
  %5 = stablehlo.add %1, %4 : tensor<6x2x3xi32>
  %k = stablehlo.iota dim = 3 : tensor<6x2x3x3xi32>
  %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<6x2x3xi32>) -> tensor<6x2x3x3xi32>
  %7 = stablehlo.multiply %k, %c4b : tensor<6x2x3x3xi32>
  %8 = stablehlo.add %6, %7 : tensor<6x2x3x3xi32>
  %m = stablehlo.iota dim = 4 : tensor<6x2x3x3x3xi32>
  %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2, 3] : (tensor<6x2x3x3xi32>) -> tensor<6x2x3x3x3xi32>
  %10 = stablehlo.add %9, %m : tensor<6x2x3x3x3xi32>
  %idx = stablehlo.reshape %10 : (tensor<6x2x3x3x3xi32>) -> tensor<6x2x3x3x3x1xi32>
  %g = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 5>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<512xf64>, tensor<6x2x3x3x3x1xi32>) -> tensor<6x2x3x3x3xf64>
  return %g : tensor<6x2x3x3x3xf64>
}

// CHECK:    func.func @overlapping_rows(%arg0: tensor<512xf64>) -> tensor<6x2x3x3x3xf64> {
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<512xf64>) -> tensor<8x64xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %0 [0:8, 0:48] : (tensor<8x64xf64>) -> tensor<8x48xf64>
// CHECK-NEXT:    %2 = stablehlo.reshape %1 : (tensor<8x48xf64>) -> tensor<8x1x3x4x4xf64>
// CHECK-NEXT:    %3 = stablehlo.slice %2 [0:6, 0:1, 0:3, 0:3, 0:3] : (tensor<8x1x3x4x4xf64>) -> tensor<6x1x3x3x3xf64>
// CHECK-NEXT:    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2, 3, 4] : (tensor<6x1x3x3x3xf64>) -> tensor<6x1x3x3x3x1xf64>
// CHECK-NEXT:    %5 = stablehlo.reshape %4 : (tensor<6x1x3x3x3x1xf64>) -> tensor<6x1x3x3x3xf64>
// CHECK-NEXT:    %6 = stablehlo.slice %arg0 [48:432] : (tensor<512xf64>) -> tensor<384xf64>
// CHECK-NEXT:    %7 = stablehlo.reshape %6 : (tensor<384xf64>) -> tensor<6x64xf64>
// CHECK-NEXT:    %8 = stablehlo.slice %7 [0:6, 0:48] : (tensor<6x64xf64>) -> tensor<6x48xf64>
// CHECK-NEXT:    %9 = stablehlo.reshape %8 : (tensor<6x48xf64>) -> tensor<6x1x3x4x4xf64>
// CHECK-NEXT:    %10 = stablehlo.slice %9 [0:6, 0:1, 0:3, 0:3, 0:3] : (tensor<6x1x3x4x4xf64>) -> tensor<6x1x3x3x3xf64>
// CHECK-NEXT:    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1, 2, 3, 4] : (tensor<6x1x3x3x3xf64>) -> tensor<6x1x3x3x3x1xf64>
// CHECK-NEXT:    %12 = stablehlo.reshape %11 : (tensor<6x1x3x3x3x1xf64>) -> tensor<6x1x3x3x3xf64>
// CHECK-NEXT:    %13 = stablehlo.concatenate %5, %12, dim = 1 : (tensor<6x1x3x3x3xf64>, tensor<6x1x3x3x3xf64>) -> tensor<6x2x3x3x3xf64>
// CHECK-NEXT:    return %13 : tensor<6x2x3x3x3xf64>
// CHECK-NEXT:  }

// -----

// A grid that factors as it is is left to the grid rewrite alone.
func.func @factors(%x: tensor<512xf64>) -> tensor<8x2x3xf64> {
  %c64 = stablehlo.constant dense<64> : tensor<8xi32>
  %c16 = stablehlo.constant dense<16> : tensor<8x2xi32>
  %e = stablehlo.iota dim = 0 : tensor<8xi32>
  %r = stablehlo.iota dim = 1 : tensor<8x2xi32>
  %c = stablehlo.iota dim = 2 : tensor<8x2x3xi32>
  %0 = stablehlo.multiply %e, %c64 : tensor<8xi32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<8xi32>) -> tensor<8x2xi32>
  %2 = stablehlo.multiply %r, %c16 : tensor<8x2xi32>
  %3 = stablehlo.add %1, %2 : tensor<8x2xi32>
  %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<8x2xi32>) -> tensor<8x2x3xi32>
  %5 = stablehlo.add %4, %c : tensor<8x2x3xi32>
  %idx = stablehlo.reshape %5 : (tensor<8x2x3xi32>) -> tensor<8x2x3x1xi32>
  %g = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<512xf64>, tensor<8x2x3x1xi32>) -> tensor<8x2x3xf64>
  return %g : tensor<8x2x3xf64>
}

// CHECK:    func.func @factors(%arg0: tensor<512xf64>) -> tensor<8x2x3xf64> {
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<512xf64>) -> tensor<8x4x16xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %0 [0:8, 0:2, 0:3] : (tensor<8x4x16xf64>) -> tensor<8x2x3xf64>
// CHECK-NEXT:    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2] : (tensor<8x2x3xf64>) -> tensor<8x2x3xf64>
// CHECK-NEXT:    return %2 : tensor<8x2x3xf64>
// CHECK-NEXT:  }
