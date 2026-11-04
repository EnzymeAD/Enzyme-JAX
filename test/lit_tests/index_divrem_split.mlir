// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt --split-input-file | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=index_divrem_split" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s --check-prefix=SPLIT

// A kernel that writes element i of a vector to position (i / 125) * 125 +
// i % 125, which is i: once the quotient and the remainder are the two iotas
// of the index split into (13824, 125), the scatter covers its operand and
// is its update.
func.func @identity_scatter(%dst: tensor<1728000xf64>, %upd: tensor<1728000xf64>) -> tensor<1728000xf64> {
  %c = stablehlo.constant dense<0> : tensor<1728000x1xi64>
  %c_0 = stablehlo.constant dense<125> : tensor<1728000x1xi64>
  %0 = stablehlo.iota dim = 0 : tensor<1728000x1xi64>
  %1 = stablehlo.remainder %0, %c_0 : tensor<1728000x1xi64>
  %2 = stablehlo.compare LT, %1, %c : (tensor<1728000x1xi64>, tensor<1728000x1xi64>) -> tensor<1728000x1xi1>
  %3 = stablehlo.add %1, %c_0 : tensor<1728000x1xi64>
  %4 = stablehlo.select %2, %3, %1 : tensor<1728000x1xi1>, tensor<1728000x1xi64>
  %5 = stablehlo.divide %0, %c_0 : tensor<1728000x1xi64>
  %6 = stablehlo.multiply %5, %c_0 : tensor<1728000x1xi64>
  %7 = stablehlo.add %4, %6 : tensor<1728000x1xi64>
  %8 = "stablehlo.scatter"(%dst, %7, %upd) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
    stablehlo.return %arg3 : tensor<f64>
  }) : (tensor<1728000xf64>, tensor<1728000x1xi64>, tensor<1728000xf64>) -> tensor<1728000xf64>
  return %8 : tensor<1728000xf64>
}

// CHECK:      func.func @identity_scatter(%arg0: tensor<1728000xf64>, %arg1: tensor<1728000xf64>) -> tensor<1728000xf64> {
// CHECK-NEXT:   return %arg1 : tensor<1728000xf64>

// SPLIT:      func.func @identity_scatter(%arg0: tensor<1728000xf64>, %arg1: tensor<1728000xf64>) -> tensor<1728000xf64> {
// SPLIT-NEXT:   %c = stablehlo.constant dense<125> : tensor<13824x125x1xi64>
// SPLIT-NEXT:   %c_0 = stablehlo.constant dense<0> : tensor<13824x125x1xi64>
// SPLIT-NEXT:   %0 = stablehlo.iota dim = 1 : tensor<13824x125x1xi64>
// SPLIT-NEXT:   %1 = stablehlo.compare LT, %0, %c_0 : (tensor<13824x125x1xi64>, tensor<13824x125x1xi64>) -> tensor<13824x125x1xi1>
// SPLIT-NEXT:   %2 = stablehlo.add %0, %c : tensor<13824x125x1xi64>
// SPLIT-NEXT:   %3 = stablehlo.select %1, %2, %0 : tensor<13824x125x1xi1>, tensor<13824x125x1xi64>
// SPLIT-NEXT:   %4 = stablehlo.iota dim = 0 : tensor<13824x125x1xi64>
// SPLIT-NEXT:   %5 = stablehlo.multiply %4, %c : tensor<13824x125x1xi64>
// SPLIT-NEXT:   %6 = stablehlo.add %3, %5 : tensor<13824x125x1xi64>
// SPLIT-NEXT:   %7 = stablehlo.reshape %arg1 : (tensor<1728000xf64>) -> tensor<13824x125xf64>
// SPLIT-NEXT:   %8 = "stablehlo.scatter"(%arg0, %6, %7) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
// SPLIT-NEXT:   ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// SPLIT-NEXT:     stablehlo.return %arg3 : tensor<f64>
// SPLIT-NEXT:   }) : (tensor<1728000xf64>, tensor<13824x125x1xi64>, tensor<13824x125xf64>) -> tensor<1728000xf64>
// SPLIT-NEXT:   return %8 : tensor<1728000xf64>

// -----



// Element i written to (i / 8) * 24 + i % 8: the first 8 of every 24.
func.func @strided_scatter(%dst: tensor<331776xf64>, %upd: tensor<110592xf64>) -> tensor<331776xf64> {
  %c_2 = stablehlo.constant dense<24> : tensor<110592x1xi64>
  %c_5 = stablehlo.constant dense<8> : tensor<110592x1xi64>
  %0 = stablehlo.iota dim = 0 : tensor<110592x1xi64>
  %1 = stablehlo.remainder %0, %c_5 : tensor<110592x1xi64>
  %2 = stablehlo.divide %0, %c_5 : tensor<110592x1xi64>
  %3 = stablehlo.multiply %2, %c_2 : tensor<110592x1xi64>
  %4 = stablehlo.add %1, %3 : tensor<110592x1xi64>
  %5 = "stablehlo.scatter"(%dst, %4, %upd) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
    stablehlo.return %arg3 : tensor<f64>
  }) : (tensor<331776xf64>, tensor<110592x1xi64>, tensor<110592xf64>) -> tensor<331776xf64>
  return %5 : tensor<331776xf64>
}

// CHECK:      func.func @strided_scatter(%arg0: tensor<331776xf64>, %arg1: tensor<110592xf64>) -> tensor<331776xf64> {
// CHECK-NEXT:   %0 = stablehlo.reshape %arg1 : (tensor<110592xf64>) -> tensor<13824x8xf64>
// CHECK-NEXT:   %1 = stablehlo.reshape %arg0 : (tensor<331776xf64>) -> tensor<13824x24xf64>
// CHECK-NEXT:   %2 = stablehlo.slice %1 [0:13824, 8:24] : (tensor<13824x24xf64>) -> tensor<13824x16xf64>
// CHECK-NEXT:   %3 = stablehlo.concatenate %0, %2, dim = 1 : (tensor<13824x8xf64>, tensor<13824x16xf64>) -> tensor<13824x24xf64>
// CHECK-NEXT:   %4 = stablehlo.reshape %3 : (tensor<13824x24xf64>) -> tensor<331776xf64>
// CHECK-NEXT:   return %4 : tensor<331776xf64>

// SPLIT:      func.func @strided_scatter(%arg0: tensor<331776xf64>, %arg1: tensor<110592xf64>) -> tensor<331776xf64> {
// SPLIT-NEXT:   %c = stablehlo.constant dense<24> : tensor<13824x8x1xi64>
// SPLIT-NEXT:   %0 = stablehlo.iota dim = 1 : tensor<13824x8x1xi64>
// SPLIT-NEXT:   %1 = stablehlo.iota dim = 0 : tensor<13824x8x1xi64>
// SPLIT-NEXT:   %2 = stablehlo.multiply %1, %c : tensor<13824x8x1xi64>
// SPLIT-NEXT:   %3 = stablehlo.add %0, %2 : tensor<13824x8x1xi64>
// SPLIT-NEXT:   %4 = stablehlo.reshape %arg1 : (tensor<110592xf64>) -> tensor<13824x8xf64>
// SPLIT-NEXT:   %5 = "stablehlo.scatter"(%arg0, %3, %4) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
// SPLIT-NEXT:   ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// SPLIT-NEXT:     stablehlo.return %arg3 : tensor<f64>
// SPLIT-NEXT:   }) : (tensor<331776xf64>, tensor<13824x8x1xi64>, tensor<13824x8xf64>) -> tensor<331776xf64>
// SPLIT-NEXT:   return %5 : tensor<331776xf64>

// -----



// The gather of element (i / 216) * 216 + i % 216 of a vector of 2000376,
// through a widening of the i32 index.
func.func @identity_gather(%x: tensor<2000376xf64>) -> tensor<2000376x1xf64> {
  %c = stablehlo.constant dense<216> : tensor<2000376x1xi32>
  %0 = stablehlo.iota dim = 0 : tensor<2000376x1xi32>
  %1 = stablehlo.remainder %0, %c : tensor<2000376x1xi32>
  %2 = stablehlo.divide %0, %c : tensor<2000376x1xi32>
  %3 = stablehlo.multiply %2, %c : tensor<2000376x1xi32>
  %4 = stablehlo.add %1, %3 : tensor<2000376x1xi32>
  %5 = stablehlo.convert %4 : (tensor<2000376x1xi32>) -> tensor<2000376x1xi64>
  %6 = "stablehlo.gather"(%x, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<2000376xf64>, tensor<2000376x1xi64>) -> tensor<2000376x1xf64>
  return %6 : tensor<2000376x1xf64>
}

// CHECK:      func.func @identity_gather(%arg0: tensor<2000376xf64>) -> tensor<2000376x1xf64> {
// CHECK-NEXT:   %0 = stablehlo.reshape %arg0 : (tensor<2000376xf64>) -> tensor<2000376x1xf64>
// CHECK-NEXT:   return %0 : tensor<2000376x1xf64>

// SPLIT:      func.func @identity_gather(%arg0: tensor<2000376xf64>) -> tensor<2000376x1xf64> {
// SPLIT-NEXT:   %c = stablehlo.constant dense<216> : tensor<9261x216x1xi32>
// SPLIT-NEXT:   %0 = stablehlo.iota dim = 1 : tensor<9261x216x1xi32>
// SPLIT-NEXT:   %1 = stablehlo.iota dim = 0 : tensor<9261x216x1xi32>
// SPLIT-NEXT:   %2 = stablehlo.multiply %1, %c : tensor<9261x216x1xi32>
// SPLIT-NEXT:   %3 = stablehlo.add %0, %2 : tensor<9261x216x1xi32>
// SPLIT-NEXT:   %4 = stablehlo.convert %3 : (tensor<9261x216x1xi32>) -> tensor<9261x216x1xi64>
// SPLIT-NEXT:   %5 = "stablehlo.gather"(%arg0, %4) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<2000376xf64>, tensor<9261x216x1xi64>) -> tensor<9261x216x1xf64>
// SPLIT-NEXT:   %6 = stablehlo.reshape %5 : (tensor<9261x216x1xf64>) -> tensor<2000376x1xf64>
// SPLIT-NEXT:   return %6 : tensor<2000376x1xf64>

// -----



// A divisor that does not divide the extent is left alone.
func.func @odd_divisor(%dst: tensor<1000xf64>, %upd: tensor<1000xf64>) -> tensor<1000xf64> {
  %c = stablehlo.constant dense<7> : tensor<1000x1xi64>
  %0 = stablehlo.iota dim = 0 : tensor<1000x1xi64>
  %1 = stablehlo.remainder %0, %c : tensor<1000x1xi64>
  %2 = stablehlo.divide %0, %c : tensor<1000x1xi64>
  %3 = stablehlo.multiply %2, %c : tensor<1000x1xi64>
  %4 = stablehlo.add %1, %3 : tensor<1000x1xi64>
  %5 = "stablehlo.scatter"(%dst, %4, %upd) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
    stablehlo.return %arg3 : tensor<f64>
  }) : (tensor<1000xf64>, tensor<1000x1xi64>, tensor<1000xf64>) -> tensor<1000xf64>
  return %5 : tensor<1000xf64>
}

// SPLIT:      func.func @odd_divisor(%arg0: tensor<1000xf64>, %arg1: tensor<1000xf64>) -> tensor<1000xf64> {
// SPLIT-NEXT:   %c = stablehlo.constant dense<7> : tensor<1000x1xi64>
// SPLIT-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<1000x1xi64>
// SPLIT-NEXT:   %1 = stablehlo.remainder %0, %c : tensor<1000x1xi64>
// SPLIT-NEXT:   %2 = stablehlo.divide %0, %c : tensor<1000x1xi64>
// SPLIT-NEXT:   %3 = stablehlo.multiply %2, %c : tensor<1000x1xi64>
// SPLIT-NEXT:   %4 = stablehlo.add %1, %3 : tensor<1000x1xi64>
// SPLIT-NEXT:   %5 = "stablehlo.scatter"(%arg0, %4, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// SPLIT-NEXT:   ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// SPLIT-NEXT:     stablehlo.return %arg3 : tensor<f64>
// SPLIT-NEXT:   }) : (tensor<1000xf64>, tensor<1000x1xi64>, tensor<1000xf64>) -> tensor<1000xf64>
// SPLIT-NEXT:   return %5 : tensor<1000xf64>

// -----


// The quotient and remainder computed at the element rank and broadcast over
// the lanes: element i's 27 lanes read ((i / 27) * 27 + i % 27) * 27 + lane,
// which is the whole vector in order.
func.func @broadcast_divmod_gather(%x: tensor<5832xf64>) -> tensor<216x27xf64> {
  %lane = stablehlo.iota dim = 0 : tensor<27x1xi64>
  %c = stablehlo.constant dense<27> : tensor<216xi64>
  %0 = stablehlo.iota dim = 0 : tensor<216xi64>
  %1 = stablehlo.divide %0, %c : tensor<216xi64>
  %2 = stablehlo.remainder %0, %c : tensor<216xi64>
  %3 = stablehlo.multiply %1, %c : tensor<216xi64>
  %4 = stablehlo.add %2, %3 : tensor<216xi64>
  %5 = stablehlo.multiply %4, %c : tensor<216xi64>
  %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<216xi64>) -> tensor<216x27x1xi64>
  %7 = stablehlo.broadcast_in_dim %lane, dims = [1, 2] : (tensor<27x1xi64>) -> tensor<216x27x1xi64>
  %8 = stablehlo.add %7, %6 : tensor<216x27x1xi64>
  %9 = "stablehlo.gather"(%x, %8) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<5832xf64>, tensor<216x27x1xi64>) -> tensor<216x27xf64>
  return %9 : tensor<216x27xf64>
}

// CHECK:      func.func @broadcast_divmod_gather(%arg0: tensor<5832xf64>) -> tensor<216x27xf64> {
// CHECK-NEXT:   %0 = stablehlo.reshape %arg0 : (tensor<5832xf64>) -> tensor<216x27xf64>
// CHECK-NEXT:   return %0 : tensor<216x27xf64>

// SPLIT:      func.func @broadcast_divmod_gather(%arg0: tensor<5832xf64>) -> tensor<216x27xf64> {
// SPLIT-NEXT:   %c = stablehlo.constant dense<27> : tensor<8x27xi64>
// SPLIT-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<27x1xi64>
// SPLIT-NEXT:   %1 = stablehlo.broadcast_in_dim %0, dims = [2, 3] : (tensor<27x1xi64>) -> tensor<8x27x27x1xi64>
// SPLIT-NEXT:   %2 = stablehlo.iota dim = 1 : tensor<8x27xi64>
// SPLIT-NEXT:   %3 = stablehlo.iota dim = 0 : tensor<8x27xi64>
// SPLIT-NEXT:   %4 = stablehlo.multiply %3, %c : tensor<8x27xi64>
// SPLIT-NEXT:   %5 = stablehlo.add %2, %4 : tensor<8x27xi64>
// SPLIT-NEXT:   %6 = stablehlo.multiply %5, %c : tensor<8x27xi64>
// SPLIT-NEXT:   %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<8x27xi64>) -> tensor<8x27x27x1xi64>
// SPLIT-NEXT:   %8 = stablehlo.add %1, %7 : tensor<8x27x27x1xi64>
// SPLIT-NEXT:   %9 = "stablehlo.gather"(%arg0, %8) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<5832xf64>, tensor<8x27x27x1xi64>) -> tensor<8x27x27xf64>
// SPLIT-NEXT:   %10 = stablehlo.reshape %9 : (tensor<8x27x27xf64>) -> tensor<216x27xf64>
// SPLIT-NEXT:   return %10 : tensor<216x27xf64>
