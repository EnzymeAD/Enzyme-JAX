// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=scatter_op_canon;gather_op_canon" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// The values of static indices, read without the reference interpreter: each
// op as it computes it, and none where it would divide by zero.

// Index 2 * i + 1, through a select on a compare that always holds and a
// remainder that never wraps.
func.func @select_compare(%x: tensor<9xf64>) -> tensor<4xf64> {
  %i = stablehlo.iota dim = 0 : tensor<4x1xi64>
  %two = stablehlo.constant dense<2> : tensor<4x1xi64>
  %one = stablehlo.constant dense<1> : tensor<4x1xi64>
  %d = stablehlo.multiply %i, %two : tensor<4x1xi64>
  %o = stablehlo.add %d, %one : tensor<4x1xi64>
  %big = stablehlo.constant dense<100> : tensor<4x1xi64>
  %lt = stablehlo.compare LT, %i, %big : (tensor<4x1xi64>, tensor<4x1xi64>) -> tensor<4x1xi1>
  %s = stablehlo.select %lt, %o, %big : tensor<4x1xi1>, tensor<4x1xi64>
  %r = stablehlo.remainder %s, %big : tensor<4x1xi64>
  %g = "stablehlo.gather"(%x, %r) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<9xf64>, tensor<4x1xi64>) -> tensor<4xf64>
  return %g : tensor<4xf64>
}

// CHECK-LABEL: func.func @select_compare(%arg0: tensor<9xf64>) -> tensor<4xf64> {
// CHECK-NEXT: %0 = stablehlo.slice %arg0 [1:9:2] : (tensor<9xf64>) -> tensor<4xf64>
// CHECK-NEXT: return %0 : tensor<4xf64>
// CHECK-NEXT: }

// A transpose and a strided slice of an iota, through a convert from i32 and
// a max and a min that do not bind: rows 0 and 2 of a 4x3 operand, transposed.
func.func @transpose_slice_convert(%x: tensor<4x3xf64>) -> tensor<3x2xf64> {
  %i = stablehlo.iota dim = 0 : tensor<4x3xi32>
  %t = stablehlo.transpose %i, dims = [1, 0] : (tensor<4x3xi32>) -> tensor<3x4xi32>
  %s = stablehlo.slice %t [0:3, 0:4:2] : (tensor<3x4xi32>) -> tensor<3x2xi32>
  %c = stablehlo.convert %s : (tensor<3x2xi32>) -> tensor<3x2xi64>
  %lo = stablehlo.constant dense<-5> : tensor<3x2xi64>
  %hi = stablehlo.constant dense<50> : tensor<3x2xi64>
  %m = stablehlo.maximum %c, %lo : tensor<3x2xi64>
  %n = stablehlo.minimum %m, %hi : tensor<3x2xi64>
  %j = stablehlo.iota dim = 0 : tensor<3x2xi64>
  %row = stablehlo.reshape %n : (tensor<3x2xi64>) -> tensor<3x2x1xi64>
  %col = stablehlo.reshape %j : (tensor<3x2xi64>) -> tensor<3x2x1xi64>
  %idx = stablehlo.concatenate %row, %col, dim = 2 : (tensor<3x2x1xi64>, tensor<3x2x1xi64>) -> tensor<3x2x2xi64>
  %g = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<4x3xf64>, tensor<3x2x2xi64>) -> tensor<3x2xf64>
  return %g : tensor<3x2xf64>
}

// CHECK-LABEL: func.func @transpose_slice_convert(%arg0: tensor<4x3xf64>) -> tensor<3x2xf64> {
// CHECK-NEXT: %0 = stablehlo.reshape %arg0 : (tensor<4x3xf64>) -> tensor<2x6xf64>
// CHECK-NEXT: %1 = stablehlo.slice %0 [0:2, 0:3] : (tensor<2x6xf64>) -> tensor<2x3xf64>
// CHECK-NEXT: %2 = stablehlo.broadcast_in_dim %1, dims = [1, 0] : (tensor<2x3xf64>) -> tensor<3x2xf64>
// CHECK-NEXT: return %2 : tensor<3x2xf64>
// CHECK-NEXT: }

// A remainder of a negative dividend keeps its sign, so -1 - i rem 8 is not
// 7 - i; negated back it is 1 + i.
func.func @signed_remainder(%x: tensor<8xf64>) -> tensor<4xf64> {
  %i = stablehlo.iota dim = 0 : tensor<4x1xi64>
  %m1 = stablehlo.constant dense<-1> : tensor<4x1xi64>
  %e = stablehlo.constant dense<8> : tensor<4x1xi64>
  %a = stablehlo.subtract %m1, %i : tensor<4x1xi64>
  %r = stablehlo.remainder %a, %e : tensor<4x1xi64>
  %n = stablehlo.negate %r : tensor<4x1xi64>
  %g = "stablehlo.gather"(%x, %n) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<8xf64>, tensor<4x1xi64>) -> tensor<4xf64>
  return %g : tensor<4xf64>
}

// CHECK-LABEL: func.func @signed_remainder(%arg0: tensor<8xf64>) -> tensor<4xf64> {
// CHECK-NEXT: %0 = stablehlo.slice %arg0 [1:5] : (tensor<8xf64>) -> tensor<4xf64>
// CHECK-NEXT: return %0 : tensor<4xf64>
// CHECK-NEXT: }

// An unsigned index: i << 6 is 0, 64, 128, 192 as a ui8, an unsigned divide
// by 64 reads it back as i (a signed one would not).
func.func @unsigned_index(%x: tensor<8xf64>) -> tensor<4xf64> {
  %i = stablehlo.iota dim = 0 : tensor<4x1xui8>
  %six = stablehlo.constant dense<6> : tensor<4x1xui8>
  %d = stablehlo.constant dense<64> : tensor<4x1xui8>
  %l = stablehlo.shift_left %i, %six : tensor<4x1xui8>
  %q = stablehlo.divide %l, %d : tensor<4x1xui8>
  %g = "stablehlo.gather"(%x, %q) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<8xf64>, tensor<4x1xui8>) -> tensor<4xf64>
  return %g : tensor<4xf64>
}

// CHECK-LABEL: func.func @unsigned_index(%arg0: tensor<8xf64>) -> tensor<4xf64> {
// CHECK-NEXT: %0 = stablehlo.slice %arg0 [0:4] : (tensor<8xf64>) -> tensor<4xf64>
// CHECK-NEXT: return %0 : tensor<4xf64>
// CHECK-NEXT: }

// The same as an i8: i << 6 wraps to 0, 64, -128, -64, and a signed divide
// by 64 gives 0, 1, -2, -1, no grid.
func.func @signed_wrap(%x: tensor<8xf64>) -> tensor<4xf64> {
  %i = stablehlo.iota dim = 0 : tensor<4x1xi8>
  %six = stablehlo.constant dense<6> : tensor<4x1xi8>
  %d = stablehlo.constant dense<64> : tensor<4x1xi8>
  %l = stablehlo.shift_left %i, %six : tensor<4x1xi8>
  %q = stablehlo.divide %l, %d : tensor<4x1xi8>
  %g = "stablehlo.gather"(%x, %q) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<8xf64>, tensor<4x1xi8>) -> tensor<4xf64>
  return %g : tensor<4xf64>
}

// CHECK-LABEL: func.func @signed_wrap(%arg0: tensor<8xf64>) -> tensor<4xf64> {
// CHECK-NEXT: %c = stablehlo.constant dense<64> : tensor<4x1xi8>
// CHECK-NEXT: %c_0 = stablehlo.constant dense<6> : tensor<4x1xi8>
// CHECK-NEXT: %0 = stablehlo.iota dim = 0 : tensor<4x1xi8>
// CHECK-NEXT: %1 = stablehlo.shift_left %0, %c_0 : tensor<4x1xi8>
// CHECK-NEXT: %2 = stablehlo.divide %1, %c : tensor<4x1xi8>
// CHECK-NEXT: %3 = "stablehlo.gather"(%arg0, %2) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<8xf64>, tensor<4x1xi8>) -> tensor<4xf64>
// CHECK-NEXT: return %3 : tensor<4xf64>
// CHECK-NEXT: }
