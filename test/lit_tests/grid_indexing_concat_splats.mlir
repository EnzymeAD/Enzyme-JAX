// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=gather_op_canon" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Indices concatenated from several splat constants and an iota, read through
// a remainder: each splat operand of the concatenation is expanded in turn.
func.func @concat_splats(%x: tensor<9xf64>) -> tensor<6xf64> {
  %a = stablehlo.constant dense<0> : tensor<1x1xi64>
  %i = stablehlo.iota dim = 0 : tensor<2x1xi64>
  %b = stablehlo.constant dense<5> : tensor<2x1xi64>
  %c = stablehlo.constant dense<8> : tensor<1x1xi64>
  %idx = stablehlo.concatenate %a, %i, %b, %c, dim = 0 : (tensor<1x1xi64>, tensor<2x1xi64>, tensor<2x1xi64>, tensor<1x1xi64>) -> tensor<6x1xi64>
  %nine = stablehlo.constant dense<9> : tensor<6x1xi64>
  %r = stablehlo.remainder %idx, %nine : tensor<6x1xi64>
  %g = "stablehlo.gather"(%x, %r) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<9xf64>, tensor<6x1xi64>) -> tensor<6xf64>
  return %g : tensor<6xf64>
}

// CHECK-LABEL: func.func @concat_splats
