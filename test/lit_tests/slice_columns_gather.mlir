// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=slice_columns_gather" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// A per-point field of 9 components read one component at a time: each
// column slice is an uncoalesced copy for XLA. One gather of the three
// columns reads every point's row once; the columns are slices of it.
func.func @columns(%x: tensor<4096x6x6x6x9xf64>) -> (tensor<4096x6x6x6x1xf64>, tensor<4096x6x6x6x1xf64>, tensor<4096x6x6x6x1xf64>) {
  %a = stablehlo.slice %x [0:4096, 0:6, 0:6, 0:6, 0:1] : (tensor<4096x6x6x6x9xf64>) -> tensor<4096x6x6x6x1xf64>
  %b = stablehlo.slice %x [0:4096, 0:6, 0:6, 0:6, 4:5] : (tensor<4096x6x6x6x9xf64>) -> tensor<4096x6x6x6x1xf64>
  %c = stablehlo.slice %x [0:4096, 0:6, 0:6, 0:6, 8:9] : (tensor<4096x6x6x6x9xf64>) -> tensor<4096x6x6x6x1xf64>
  return %a, %b, %c : tensor<4096x6x6x6x1xf64>, tensor<4096x6x6x6x1xf64>, tensor<4096x6x6x6x1xf64>
}

// CHECK:  func.func @columns(%arg0: tensor<4096x6x6x6x9xf64>) -> (tensor<4096x6x6x6x1xf64>, tensor<4096x6x6x6x1xf64>, tensor<4096x6x6x6x1xf64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<{{\[\[0\], \[4\], \[8\]\]}}> : tensor<3x1xi64>
// CHECK-NEXT:    %0 = "stablehlo.gather"(%arg0, %c) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], collapsed_slice_dims = [4], start_index_map = [4], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4096, 6, 6, 6, 1>}> : (tensor<4096x6x6x6x9xf64>, tensor<3x1xi64>) -> tensor<4096x6x6x6x3xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %0 [0:4096, 0:6, 0:6, 0:6, 0:1] : (tensor<4096x6x6x6x3xf64>) -> tensor<4096x6x6x6x1xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %0 [0:4096, 0:6, 0:6, 0:6, 1:2] : (tensor<4096x6x6x6x3xf64>) -> tensor<4096x6x6x6x1xf64>
// CHECK-NEXT:    %3 = stablehlo.slice %0 [0:4096, 0:6, 0:6, 0:6, 2:3] : (tensor<4096x6x6x6x3xf64>) -> tensor<4096x6x6x6x1xf64>
// CHECK-NEXT:    return %1, %2, %3 : tensor<4096x6x6x6x1xf64>, tensor<4096x6x6x6x1xf64>, tensor<4096x6x6x6x1xf64>
// CHECK-NEXT:  }

// -----

// Two columns do not pay for a gather; a slice of a row (not the last
// dimension) is not a column.
func.func @few(%x: tensor<4096x6x6x6x9xf64>) -> (tensor<4096x6x6x6x1xf64>, tensor<4096x6x6x6x1xf64>, tensor<4096x6x6x1x9xf64>) {
  %a = stablehlo.slice %x [0:4096, 0:6, 0:6, 0:6, 0:1] : (tensor<4096x6x6x6x9xf64>) -> tensor<4096x6x6x6x1xf64>
  %b = stablehlo.slice %x [0:4096, 0:6, 0:6, 0:6, 4:5] : (tensor<4096x6x6x6x9xf64>) -> tensor<4096x6x6x6x1xf64>
  %r = stablehlo.slice %x [0:4096, 0:6, 0:6, 2:3, 0:9] : (tensor<4096x6x6x6x9xf64>) -> tensor<4096x6x6x1x9xf64>
  return %a, %b, %r : tensor<4096x6x6x6x1xf64>, tensor<4096x6x6x6x1xf64>, tensor<4096x6x6x1x9xf64>
}

// CHECK:  func.func @few(%arg0: tensor<4096x6x6x6x9xf64>) -> (tensor<4096x6x6x6x1xf64>, tensor<4096x6x6x6x1xf64>, tensor<4096x6x6x1x9xf64>) {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:4096, 0:6, 0:6, 0:6, 0:1] : (tensor<4096x6x6x6x9xf64>) -> tensor<4096x6x6x6x1xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [0:4096, 0:6, 0:6, 0:6, 4:5] : (tensor<4096x6x6x6x9xf64>) -> tensor<4096x6x6x6x1xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %arg0 [0:4096, 0:6, 0:6, 2:3, 0:9] : (tensor<4096x6x6x6x9xf64>) -> tensor<4096x6x6x1x9xf64>
// CHECK-NEXT:    return %0, %1, %2 : tensor<4096x6x6x6x1xf64>, tensor<4096x6x6x6x1xf64>, tensor<4096x6x6x1x9xf64>
// CHECK-NEXT:  }

// -----

// Every column of a value read in order is the value itself.
func.func @all(%x: tensor<4096x6x6x6x3xf64>) -> (tensor<4096x6x6x6x1xf64>, tensor<4096x6x6x6x1xf64>, tensor<4096x6x6x6x1xf64>) {
  %a = stablehlo.slice %x [0:4096, 0:6, 0:6, 0:6, 0:1] : (tensor<4096x6x6x6x3xf64>) -> tensor<4096x6x6x6x1xf64>
  %b = stablehlo.slice %x [0:4096, 0:6, 0:6, 0:6, 1:2] : (tensor<4096x6x6x6x3xf64>) -> tensor<4096x6x6x6x1xf64>
  %c = stablehlo.slice %x [0:4096, 0:6, 0:6, 0:6, 2:3] : (tensor<4096x6x6x6x3xf64>) -> tensor<4096x6x6x6x1xf64>
  return %a, %b, %c : tensor<4096x6x6x6x1xf64>, tensor<4096x6x6x6x1xf64>, tensor<4096x6x6x6x1xf64>
}

// CHECK:  func.func @all(%arg0: tensor<4096x6x6x6x3xf64>) -> (tensor<4096x6x6x6x1xf64>, tensor<4096x6x6x6x1xf64>, tensor<4096x6x6x6x1xf64>) {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:4096, 0:6, 0:6, 0:6, 0:1] : (tensor<4096x6x6x6x3xf64>) -> tensor<4096x6x6x6x1xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [0:4096, 0:6, 0:6, 0:6, 1:2] : (tensor<4096x6x6x6x3xf64>) -> tensor<4096x6x6x6x1xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %arg0 [0:4096, 0:6, 0:6, 0:6, 2:3] : (tensor<4096x6x6x6x3xf64>) -> tensor<4096x6x6x6x1xf64>
// CHECK-NEXT:    return %0, %1, %2 : tensor<4096x6x6x6x1xf64>, tensor<4096x6x6x6x1xf64>, tensor<4096x6x6x6x1xf64>
// CHECK-NEXT:  }

// -----

// A small value is left alone: its columns are cheap.
func.func @small(%x: tensor<16x9xf64>) -> (tensor<16x1xf64>, tensor<16x1xf64>, tensor<16x1xf64>) {
  %a = stablehlo.slice %x [0:16, 0:1] : (tensor<16x9xf64>) -> tensor<16x1xf64>
  %b = stablehlo.slice %x [0:16, 3:4] : (tensor<16x9xf64>) -> tensor<16x1xf64>
  %c = stablehlo.slice %x [0:16, 8:9] : (tensor<16x9xf64>) -> tensor<16x1xf64>
  return %a, %b, %c : tensor<16x1xf64>, tensor<16x1xf64>, tensor<16x1xf64>
}

// CHECK:  func.func @small(%arg0: tensor<16x9xf64>) -> (tensor<16x1xf64>, tensor<16x1xf64>, tensor<16x1xf64>) {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:16, 0:1] : (tensor<16x9xf64>) -> tensor<16x1xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [0:16, 3:4] : (tensor<16x9xf64>) -> tensor<16x1xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %arg0 [0:16, 8:9] : (tensor<16x9xf64>) -> tensor<16x1xf64>
// CHECK-NEXT:    return %0, %1, %2 : tensor<16x1xf64>, tensor<16x1xf64>, tensor<16x1xf64>
// CHECK-NEXT:  }

// -----

// Columns of one value read through several views of it: two through a
// reshape, two through that value reshaped and transposed (one reshaped
// again). One gather of the value as [rows, columns]; each column is the
// view of its slice of the gather.
func.func @views(%x: tensor<294912xf64>) -> (tensor<4096x8x1xf64>, tensor<4096x8x1xf64>, tensor<1x4096x4x2xf64>, tensor<1x4096x8xf64>) {
  %v = stablehlo.reshape %x : (tensor<294912xf64>) -> tensor<4096x8x9xf64>
  %a = stablehlo.slice %v [0:4096, 0:8, 0:1] : (tensor<4096x8x9xf64>) -> tensor<4096x8x1xf64>
  %b = stablehlo.slice %v [0:4096, 0:8, 4:5] : (tensor<4096x8x9xf64>) -> tensor<4096x8x1xf64>
  %w = stablehlo.reshape %x : (tensor<294912xf64>) -> tensor<4096x2x4x9xf64>
  %t = stablehlo.transpose %w, dims = [3, 0, 2, 1] : (tensor<4096x2x4x9xf64>) -> tensor<9x4096x4x2xf64>
  %c = stablehlo.slice %t [6:7, 0:4096, 0:4, 0:2] : (tensor<9x4096x4x2xf64>) -> tensor<1x4096x4x2xf64>
  %r = stablehlo.reshape %t : (tensor<9x4096x4x2xf64>) -> tensor<9x4096x8xf64>
  %d = stablehlo.slice %r [8:9, 0:4096, 0:8] : (tensor<9x4096x8xf64>) -> tensor<1x4096x8xf64>
  return %a, %b, %c, %d : tensor<4096x8x1xf64>, tensor<4096x8x1xf64>, tensor<1x4096x4x2xf64>, tensor<1x4096x8xf64>
}

// CHECK:  func.func @views(%arg0: tensor<294912xf64>) -> (tensor<4096x8x1xf64>, tensor<4096x8x1xf64>, tensor<1x4096x4x2xf64>, tensor<1x4096x8xf64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<{{\[\[0\], \[4\], \[6\], \[8\]\]}}> : tensor<4x1xi64>
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<294912xf64>) -> tensor<32768x9xf64>
// CHECK-NEXT:    %1 = "stablehlo.gather"(%0, %c) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 32768, 1>}> : (tensor<32768x9xf64>, tensor<4x1xi64>) -> tensor<32768x4xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %1 [0:32768, 0:1] : (tensor<32768x4xf64>) -> tensor<32768x1xf64>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<32768x1xf64>) -> tensor<4096x8x1xf64>
// CHECK-NEXT:    %4 = stablehlo.slice %1 [0:32768, 1:2] : (tensor<32768x4xf64>) -> tensor<32768x1xf64>
// CHECK-NEXT:    %5 = stablehlo.reshape %4 : (tensor<32768x1xf64>) -> tensor<4096x8x1xf64>
// CHECK-NEXT:    %6 = stablehlo.slice %1 [0:32768, 2:3] : (tensor<32768x4xf64>) -> tensor<32768x1xf64>
// CHECK-NEXT:    %7 = stablehlo.reshape %6 : (tensor<32768x1xf64>) -> tensor<4096x2x4x1xf64>
// CHECK-NEXT:    %8 = stablehlo.transpose %7, dims = [3, 0, 2, 1] : (tensor<4096x2x4x1xf64>) -> tensor<1x4096x4x2xf64>
// CHECK-NEXT:    %9 = stablehlo.slice %1 [0:32768, 3:4] : (tensor<32768x4xf64>) -> tensor<32768x1xf64>
// CHECK-NEXT:    %10 = stablehlo.reshape %9 : (tensor<32768x1xf64>) -> tensor<4096x2x4x1xf64>
// CHECK-NEXT:    %11 = stablehlo.transpose %10, dims = [3, 0, 2, 1] : (tensor<4096x2x4x1xf64>) -> tensor<1x4096x4x2xf64>
// CHECK-NEXT:    %12 = stablehlo.reshape %11 : (tensor<1x4096x4x2xf64>) -> tensor<1x4096x8xf64>
// CHECK-NEXT:    return %3, %5, %8, %12 : tensor<4096x8x1xf64>, tensor<4096x8x1xf64>, tensor<1x4096x4x2xf64>, tensor<1x4096x8xf64>
// CHECK-NEXT:  }
