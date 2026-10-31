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
