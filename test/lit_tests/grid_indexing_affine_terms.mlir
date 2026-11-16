// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=scatter_op_canon;gather_op_canon" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// The flat index e * 75 + 5 * i + j of BK6's element-major buffers, built as a
// broadcast of a scaled sum: each element writes its own 5x5 box.
func.func @broadcast_of_scaled_sum(%x: tensor<750xf64>, %u: tensor<10x5x5xf64>) -> tensor<750xf64> {
  %e = stablehlo.iota dim = 0 : tensor<10xi64>
  %c15 = stablehlo.constant dense<15> : tensor<10xi64>
  %e15 = stablehlo.multiply %c15, %e : tensor<10xi64>
  %eb = stablehlo.broadcast_in_dim %e15, dims = [0] : (tensor<10xi64>) -> tensor<10x5xi64>
  %i = stablehlo.iota dim = 1 : tensor<10x5xi64>
  %s = stablehlo.add %eb, %i : tensor<10x5xi64>
  %c5 = stablehlo.constant dense<5> : tensor<10x5xi64>
  %s5 = stablehlo.multiply %s, %c5 : tensor<10x5xi64>
  %sb = stablehlo.broadcast_in_dim %s5, dims = [0, 1] : (tensor<10x5xi64>) -> tensor<10x5x5x1xi64>
  %j = stablehlo.iota dim = 2 : tensor<10x5x5x1xi64>
  %idx = stablehlo.add %sb, %j : tensor<10x5x5x1xi64>
  %r = "stablehlo.scatter"(%x, %idx, %u) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 3>}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<750xf64>, tensor<10x5x5x1xi64>, tensor<10x5x5xf64>) -> tensor<750xf64>
  return %r : tensor<750xf64>
}

// CHECK-LABEL: func.func @broadcast_of_scaled_sum(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<750xf64>, %[[V1:[^ ,:)]+]]: tensor<10x5x5xf64>) -> tensor<750xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<750xf64>) -> tensor<10x15x5xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<10x5x5xf64>) -> tensor<10x5x5x1xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.transpose %[[V4]], dims = [0, 1, 2, 3] : (tensor<10x5x5x1xf64>) -> tensor<10x5x5x1xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.reshape %[[V5]] : (tensor<10x5x5x1xf64>) -> tensor<10x5x5xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.dynamic_update_slice %[[V3]], %[[V6]], %[[V2]], %[[V2]], %[[V2]] : (tensor<10x15x5xf64>, tensor<10x5x5xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<10x15x5xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.reshape %[[V7]] : (tensor<10x15x5xf64>) -> tensor<750xf64>
// CHECK-NEXT: return %[[V8]] : tensor<750xf64>
// CHECK-NEXT: }

// A subtracted term, a shift for a factor, and the index computed in i32.
func.func @sub_shift_convert(%x: tensor<64xf64>) -> tensor<4x4xf64> {
  %r = stablehlo.iota dim = 0 : tensor<4x4xi32>
  %c4 = stablehlo.constant dense<4> : tensor<4x4xi32>
  %r16 = stablehlo.shift_left %r, %c4 : tensor<4x4xi32>
  %c = stablehlo.iota dim = 1 : tensor<4x4xi32>
  %c3 = stablehlo.constant dense<3> : tensor<4x4xi32>
  %rc = stablehlo.subtract %c3, %c : tensor<4x4xi32>
  %s = stablehlo.add %r16, %rc : tensor<4x4xi32>
  %idx = stablehlo.convert %s : (tensor<4x4xi32>) -> tensor<4x4xi64>
  %i = stablehlo.reshape %idx : (tensor<4x4xi64>) -> tensor<4x4x1xi64>
  %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1>}> : (tensor<64xf64>, tensor<4x4x1xi64>) -> tensor<4x4xf64>
  return %g : tensor<4x4xf64>
}

// CHECK-LABEL: func.func @sub_shift_convert(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<64xf64>) -> tensor<4x4xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<64xf64>) -> tensor<4x16xf64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.slice %[[V1]] [0:4, 0:4] : (tensor<4x16xf64>) -> tensor<4x4xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reverse %[[V2]], dims = [1] : tensor<4x4xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V3]], dims = [0, 1] : (tensor<4x4xf64>) -> tensor<4x4xf64>
// CHECK-NEXT: return %[[V4]] : tensor<4x4xf64>
// CHECK-NEXT: }

// Stays: the i8 index wraps (5 * 30 = 150 is past 127), so it is not the
// affine form the terms give.
func.func @convert_wraps(%x: tensor<256xf64>) -> tensor<4xf64> {
  %r = stablehlo.iota dim = 0 : tensor<4x1xi8>
  %c = stablehlo.constant dense<30> : tensor<4x1xi8>
  %c5 = stablehlo.constant dense<5> : tensor<4x1xi8>
  %m = stablehlo.multiply %r, %c : tensor<4x1xi8>
  %s = stablehlo.multiply %m, %c5 : tensor<4x1xi8>
  %idx = stablehlo.convert %s : (tensor<4x1xi8>) -> tensor<4x1xi64>
  %g = "stablehlo.gather"(%x, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<256xf64>, tensor<4x1xi64>) -> tensor<4xf64>
  return %g : tensor<4xf64>
}

// CHECK-LABEL: func.func @convert_wraps(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<256xf64>) -> tensor<4xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.constant dense<5> : tensor<4x1xi8>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<30> : tensor<4x1xi8>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.iota dim = 0 : tensor<4x1xi8>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.multiply %[[V3]], %[[V2]] : tensor<4x1xi8>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.multiply %[[V4]], %[[V1]] : tensor<4x1xi8>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.convert %[[V5]] : (tensor<4x1xi8>) -> tensor<4x1xi64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = "stablehlo.gather"(%[[V0]], %[[V6]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<256xf64>, tensor<4x1xi64>) -> tensor<4xf64>
// CHECK-NEXT: return %[[V7]] : tensor<4xf64>
// CHECK-NEXT: }
