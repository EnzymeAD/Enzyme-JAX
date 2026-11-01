// RUN: enzymexlamlir-opt --split-input-file %s | FileCheck %s
// A roundtrip is only complete if the printed form parses back, so run the
// printer output through the parser a second time.
// RUN: enzymexlamlir-opt --split-input-file %s | enzymexlamlir-opt --split-input-file | FileCheck %s

// A contraction axis (k) occurs on both operands but not on the result.
// CHECK-LABEL: func.func @matmul
// CHECK-SAME: (%[[A:.*]]: tensor<4x8xf32>, %[[B:.*]]: tensor<8x16xf32>)
// CHECK: distributed.GrammarTerm "matmul"(%[[A]], %[[B]]) axes <([i, k], [k, j])->([i, j]) {i=4, j=16, k=8}> : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
func.func @matmul(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> tensor<4x16xf32> {
  %0 = distributed.GrammarTerm "matmul" (%a, %b) axes <([i, k], [k, j])->([i, j]) {i=4, j=16, k=8}> : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// -----

// A DFT split as Cooley-Tukey: the input digits (i, j) occur only on the
// operand and the output digits (k, l) only on the result.
// CHECK-LABEL: func.func @dft_cooley_tukey
// CHECK-SAME: (%[[X:.*]]: tensor<32xcomplex<f32>>)
// CHECK: distributed.GrammarTerm "dft"(%[[X]]) axes <([ij])->([kl]) {i=4, j=8, k=8, l=4}> : (tensor<32xcomplex<f32>>) -> tensor<32xcomplex<f32>>
func.func @dft_cooley_tukey(%x: tensor<32xcomplex<f32>>) -> tensor<32xcomplex<f32>> {
  %0 = distributed.GrammarTerm "dft" (%x) axes <([ij])->([kl]) {i=4, j=8, k=8, l=4}> : (tensor<32xcomplex<f32>>) -> tensor<32xcomplex<f32>>
  return %0 : tensor<32xcomplex<f32>>
}

// -----

// A batch axis (b) carried through a DFT on the minor dimension.
// CHECK-LABEL: func.func @batched_dft
// CHECK-SAME: (%[[X:.*]]: tensor<2x16xcomplex<f32>>)
// CHECK: distributed.GrammarTerm "dft"(%[[X]]) axes <([i, j])->([i, k]) {i=2, j=16, k=16}> : (tensor<2x16xcomplex<f32>>) -> tensor<2x16xcomplex<f32>>
func.func @batched_dft(%x: tensor<2x16xcomplex<f32>>) -> tensor<2x16xcomplex<f32>> {
  %0 = distributed.GrammarTerm "dft" (%x) axes <([i, j])->([i, k]) {i=2, j=16, k=16}> : (tensor<2x16xcomplex<f32>>) -> tensor<2x16xcomplex<f32>>
  return %0 : tensor<2x16xcomplex<f32>>
}

// -----

// A rank-0 operand has an empty mapping.
// CHECK-LABEL: func.func @scalar_operand
// CHECK-SAME: (%[[S:.*]]: tensor<f32>, %[[X:.*]]: tensor<4x8xf32>)
// CHECK: distributed.GrammarTerm "scale"(%[[S]], %[[X]]) axes <([], [i, j])->([i, j]) {i=4, j=8}> : (tensor<f32>, tensor<4x8xf32>) -> tensor<4x8xf32>
func.func @scalar_operand(%s: tensor<f32>, %x: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = distributed.GrammarTerm "scale" (%s, %x) axes <([], [i, j])->([i, j]) {i=4, j=8}> : (tensor<f32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----

// A term with no operands has an empty operand mapping list.
// CHECK-LABEL: func.func @no_operands
// CHECK: distributed.GrammarTerm "iota"() axes <()->([i]) {i=4}> : () -> tensor<4xi32>
func.func @no_operands() -> tensor<4xi32> {
  %0 = distributed.GrammarTerm "iota" () axes <()->([i]) {i=4}> : () -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// Kind-specific parameters are kept as given.
// CHECK-LABEL: func.func @params
// CHECK: distributed.GrammarTerm "dft"(%{{.*}}) axes <([i])->([j]) {i=8, j=8}> params {inverse = true} :
func.func @params(%x: tensor<8xcomplex<f32>>) -> tensor<8xcomplex<f32>> {
  %0 = distributed.GrammarTerm "dft" (%x) axes <([i])->([j]) {i=8, j=8}> params {inverse = true} : (tensor<8xcomplex<f32>>) -> tensor<8xcomplex<f32>>
  return %0 : tensor<8xcomplex<f32>>
}

// -----

// A dimension with no operator axes is written `1`, on operands and results.
// CHECK-LABEL: func.func @digitless_dims
// CHECK: distributed.GrammarTerm "reshape"(%{{.*}}) axes <([i, 1])->([1, i]) {i=4}> : (tensor<4x1xf32>) -> tensor<1x4xf32>
func.func @digitless_dims(%x: tensor<4x1xf32>) -> tensor<1x4xf32> {
  %0 = distributed.GrammarTerm "reshape" (%x) axes <([i, 1])->([1, i]) {i=4}> : (tensor<4x1xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

// Redundant `1`s within a dimension are skipped, and print canonically.
// CHECK-LABEL: func.func @redundant_ones
// CHECK: distributed.GrammarTerm "t"(%{{.*}}) axes <([ij, 1])->([ij, 1]) {i=2, j=4}> :
func.func @redundant_ones(%x: tensor<8x1xf32>) -> tensor<8x1xf32> {
  %0 = distributed.GrammarTerm "t" (%x) axes <([1 ij 1, 1 1])->([i 1 j, 1]) {i=2, j=4}> : (tensor<8x1xf32>) -> tensor<8x1xf32>
  return %0 : tensor<8x1xf32>
}
