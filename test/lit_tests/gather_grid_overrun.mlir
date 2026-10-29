// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=gather_op_canon" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// A launch of 1360 blocks of 256 lanes reading chunk 1 of 8 of each block's
// 2048 elements: the last block reaches past the 2738019 elements, where a
// gather reads the last one. The operand padded with its last element to
// the grid's reach is read as a slice.
func.func @lanes(%x: tensor<2738019xf64>) -> tensor<1360x256xf64> {
  %c_4 = stablehlo.constant dense<256> : tensor<1360xi32>
  %c_5 = stablehlo.constant dense<2048> : tensor<1360xi32>
  %0 = stablehlo.iota dim = 0 : tensor<1360xi32>
  %1 = stablehlo.multiply %c_5, %0 : tensor<1360xi32>
  %2 = stablehlo.add %1, %c_4 : tensor<1360xi32>
  %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<1360xi32>) -> tensor<1360x256x1xi32>
  %4 = stablehlo.convert %3 : (tensor<1360x256x1xi32>) -> tensor<1360x256x1xi64>
  %5 = stablehlo.iota dim = 0 : tensor<256x1xi64>
  %6 = stablehlo.broadcast_in_dim %5, dims = [1, 2] : (tensor<256x1xi64>) -> tensor<1360x256x1xi64>
  %7 = stablehlo.add %6, %4 : tensor<1360x256x1xi64>
  %8 = "stablehlo.gather"(%x, %7) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<2738019xf64>, tensor<1360x256x1xi64>) -> tensor<1360x256xf64>
  return %8 : tensor<1360x256xf64>
}

// CHECK:      func.func @lanes(%arg0: tensor<2738019xf64>) -> tensor<1360x256xf64> {
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [2738018:2738019] : (tensor<2738019xf64>) -> tensor<1xf64>
// CHECK-NEXT:   %1 = stablehlo.reshape %0 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:   %2 = stablehlo.pad %arg0, %1, low = [0], high = [47261], interior = [0] : (tensor<2738019xf64>, tensor<f64>) -> tensor<2785280xf64>
// CHECK-NEXT:   %3 = stablehlo.reshape %2 : (tensor<2785280xf64>) -> tensor<1360x2048xf64>
// CHECK-NEXT:   %4 = stablehlo.slice %3 [0:1360, 256:512] : (tensor<1360x2048xf64>) -> tensor<1360x256xf64>
// CHECK-NEXT:   %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<1360x256xf64>) -> tensor<1360x256x1xf64>
// CHECK-NEXT:   %6 = stablehlo.reshape %5 : (tensor<1360x256x1xf64>) -> tensor<1360x256xf64>
// CHECK-NEXT:   return %6 : tensor<1360x256xf64>

// -----






// A grid that starts before the operand is left alone.
func.func @before(%x: tensor<100xf64>) -> tensor<10xf64> {
  %c = stablehlo.constant dense<-5> : tensor<10x1xi64>
  %0 = stablehlo.iota dim = 0 : tensor<10x1xi64>
  %1 = stablehlo.add %0, %c : tensor<10x1xi64>
  %2 = "stablehlo.gather"(%x, %1) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<100xf64>, tensor<10x1xi64>) -> tensor<10xf64>
  return %2 : tensor<10xf64>
}

// CHECK:      func.func @before(%arg0: tensor<100xf64>) -> tensor<10xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<-5> : tensor<10x1xi64>
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<10x1xi64>
// CHECK-NEXT:   %1 = stablehlo.add %0, %c : tensor<10x1xi64>
// CHECK-NEXT:   %2 = "stablehlo.gather"(%arg0, %1) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<100xf64>, tensor<10x1xi64>) -> tensor<10xf64>
// CHECK-NEXT:   return %2 : tensor<10xf64>
