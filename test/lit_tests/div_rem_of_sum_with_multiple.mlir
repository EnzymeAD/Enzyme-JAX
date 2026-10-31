// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=div_rem_of_sum_with_multiple" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt --split-input-file | FileCheck %s --check-prefix=OPT

// The lane l of element e, linearized as 36 e + l, split into its row l / 6
// and column l % 6: 36 e holds whole blocks of 6.
func.func @linearized(%x: tensor<952560xf64>) -> tensor<26460x36xf64> {
  %c6 = stablehlo.constant dense<6> : tensor<26460x36x1xi32>
  %c36 = stablehlo.constant dense<36> : tensor<26460x36x1xi32>
  %l = stablehlo.iota dim = 1 : tensor<26460x36x1xi32>
  %e = stablehlo.iota dim = 0 : tensor<26460x36x1xi32>
  %0 = stablehlo.multiply %e, %c36 : tensor<26460x36x1xi32>
  %1 = stablehlo.add %0, %l : tensor<26460x36x1xi32>
  %row = stablehlo.divide %1, %c6 : tensor<26460x36x1xi32>
  %col = stablehlo.remainder %1, %c6 : tensor<26460x36x1xi32>
  %2 = stablehlo.multiply %row, %c6 : tensor<26460x36x1xi32>
  %3 = stablehlo.add %2, %col : tensor<26460x36x1xi32>
  %4 = stablehlo.convert %3 : (tensor<26460x36x1xi32>) -> tensor<26460x36x1xi64>
  %5 = "stablehlo.gather"(%x, %4) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<952560xf64>, tensor<26460x36x1xi64>) -> tensor<26460x36xf64>
  return %5 : tensor<26460x36xf64>
}

// CHECK:      func.func @linearized(%arg0: tensor<952560xf64>) -> tensor<26460x36xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<6> : tensor<26460x36x1xi32>
// CHECK-NEXT:   %0 = stablehlo.iota dim = 1 : tensor<26460x36x1xi32>
// CHECK-NEXT:   %1 = stablehlo.iota dim = 0 : tensor<26460x36x1xi32>
// CHECK-NEXT:   %2 = stablehlo.divide %0, %c : tensor<26460x36x1xi32>
// CHECK-NEXT:   %3 = stablehlo.multiply %1, %c : tensor<26460x36x1xi32>
// CHECK-NEXT:   %4 = stablehlo.add %3, %2 : tensor<26460x36x1xi32>
// CHECK-NEXT:   %5 = stablehlo.remainder %0, %c : tensor<26460x36x1xi32>
// CHECK-NEXT:   %6 = stablehlo.multiply %4, %c : tensor<26460x36x1xi32>
// CHECK-NEXT:   %7 = stablehlo.add %6, %5 : tensor<26460x36x1xi32>
// CHECK-NEXT:   %8 = stablehlo.convert %7 : (tensor<26460x36x1xi32>) -> tensor<26460x36x1xi64>
// CHECK-NEXT:   %9 = "stablehlo.gather"(%arg0, %8) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<952560xf64>, tensor<26460x36x1xi64>) -> tensor<26460x36xf64>
// CHECK-NEXT:   return %9 : tensor<26460x36xf64>

// OPT:      func.func @linearized(%arg0: tensor<952560xf64>) -> tensor<26460x36xf64> {
// OPT-NEXT:   %c = stablehlo.constant dense<6> : tensor<26460x36x1xi32>
// OPT-NEXT:   %0 = stablehlo.iota dim = 1 : tensor<26460x36x1xi32>
// OPT-NEXT:   %1 = stablehlo.iota dim = 0 : tensor<26460x36x1xi32>
// OPT-NEXT:   %2 = stablehlo.divide %0, %c : tensor<26460x36x1xi32>
// OPT-NEXT:   %3 = stablehlo.multiply %1, %c : tensor<26460x36x1xi32>
// OPT-NEXT:   %4 = stablehlo.add %3, %2 : tensor<26460x36x1xi32>
// OPT-NEXT:   %5 = stablehlo.remainder %0, %c : tensor<26460x36x1xi32>
// OPT-NEXT:   %6 = stablehlo.multiply %4, %c : tensor<26460x36x1xi32>
// OPT-NEXT:   %7 = stablehlo.add %6, %5 : tensor<26460x36x1xi32>
// OPT-NEXT:   %8 = stablehlo.convert %7 : (tensor<26460x36x1xi32>) -> tensor<26460x36x1xi64>
// OPT-NEXT:   %9 = "stablehlo.gather"(%arg0, %8) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<952560xf64>, tensor<26460x36x1xi64>) -> tensor<26460x36xf64>
// OPT-NEXT:   return %9 : tensor<26460x36xf64>

// -----


// A multiple the divisor does not divide, and a rest that may be negative,
// are left alone.
func.func @kept(%y: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
  %c4 = stablehlo.constant dense<4> : tensor<8xi32>
  %c6 = stablehlo.constant dense<6> : tensor<8xi32>
  %c36 = stablehlo.constant dense<36> : tensor<8xi32>
  %i = stablehlo.iota dim = 0 : tensor<8xi32>
  %0 = stablehlo.multiply %i, %c6 : tensor<8xi32>
  %1 = stablehlo.add %0, %i : tensor<8xi32>
  %2 = stablehlo.divide %1, %c4 : tensor<8xi32>
  %3 = stablehlo.multiply %i, %c36 : tensor<8xi32>
  %4 = stablehlo.add %3, %y : tensor<8xi32>
  %5 = stablehlo.remainder %4, %c6 : tensor<8xi32>
  return %2, %5 : tensor<8xi32>, tensor<8xi32>
}

// CHECK:      func.func @kept(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
// CHECK-NEXT:   %c = stablehlo.constant dense<4> : tensor<8xi32>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<6> : tensor<8xi32>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<36> : tensor<8xi32>
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<8xi32>
// CHECK-NEXT:   %1 = stablehlo.multiply %0, %c_0 : tensor<8xi32>
// CHECK-NEXT:   %2 = stablehlo.add %1, %0 : tensor<8xi32>
// CHECK-NEXT:   %3 = stablehlo.divide %2, %c : tensor<8xi32>
// CHECK-NEXT:   %4 = stablehlo.multiply %0, %c_1 : tensor<8xi32>
// CHECK-NEXT:   %5 = stablehlo.add %4, %arg0 : tensor<8xi32>
// CHECK-NEXT:   %6 = stablehlo.remainder %5, %c_0 : tensor<8xi32>
// CHECK-NEXT:   return %3, %6 : tensor<8xi32>, tensor<8xi32>

// OPT:      func.func @kept(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
// OPT-NEXT:   %c = stablehlo.constant dense<4> : tensor<8xi32>
// OPT-NEXT:   %c_0 = stablehlo.constant dense<6> : tensor<8xi32>
// OPT-NEXT:   %c_1 = stablehlo.constant dense<36> : tensor<8xi32>
// OPT-NEXT:   %c_2 = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi32>
// OPT-NEXT:   %0 = stablehlo.multiply %c_2, %c_0 : tensor<8xi32>
// OPT-NEXT:   %1 = stablehlo.add %0, %c_2 : tensor<8xi32>
// OPT-NEXT:   %2 = stablehlo.divide %1, %c : tensor<8xi32>
// OPT-NEXT:   %3 = stablehlo.multiply %c_2, %c_1 : tensor<8xi32>
// OPT-NEXT:   %4 = stablehlo.add %3, %arg0 : tensor<8xi32>
// OPT-NEXT:   %5 = stablehlo.remainder %4, %c_0 : tensor<8xi32>
// OPT-NEXT:   return %2, %5 : tensor<8xi32>, tensor<8xi32>
