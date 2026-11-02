// RUN: enzymexlamlir-opt --split-input-file --distributed-print-grammar-operators %s 2>&1 >/dev/null | FileCheck %s
// RUN: enzymexlamlir-opt --split-input-file --distributed-print-grammar-operators=canonical %s 2>&1 >/dev/null | FileCheck %s --check-prefix=CANON

// distributed-print-grammar-operators reports each op's grammar description
// as a remark (remarks go to stderr): its kind, operator axes, and one base
// signature entry per operator axis. Ops described through their Shardy rule
// have kind `sdy.<op name>`, with the rule's factors as operator axes. With
// `canonical`, the remark also gives the axes renumbered by first appearance.

// A contraction axis reduces with add. Canonicalized, the contraction axis is
// numbered second, still shared by both operands, and the signature follows it.
// CHECK: remark: grammar sdy.stablehlo.dot_general axes #distributed.operator_axes<([i, k], [k, j])->([i, j]) {i=4, j=16, k=8}> signature [parallel, parallel, reduce(add)]
// CANON: remark: grammar sdy.stablehlo.dot_general
// CANON-NEXT: canonical: axes #distributed.operator_axes<([i, j], [j, k])->([i, k]) {i=4, j=8, k=16}> signature [parallel, reduce(add), parallel]
func.func @dot_general(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> tensor<4x16xf32> {
  %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// -----

// Elementwise axes are carried through. They are already canonical.
// CHECK: remark: grammar sdy.stablehlo.add axes #distributed.operator_axes<([i, j], [i, j])->([i, j]) {i=4, j=8}> signature [parallel, parallel]
// CANON: remark: grammar sdy.stablehlo.add
// CANON-NEXT: canonical: axes #distributed.operator_axes<([i, j], [i, j])->([i, j]) {i=4, j=8}> signature [parallel, parallel]
func.func @elementwise(%a: tensor<4x8xf32>, %b: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = stablehlo.add %a, %b : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----

// A reduce's reduction axis carries its body's reduce kind.
// CHECK: remark: grammar sdy.stablehlo.reduce axes #distributed.operator_axes<([i, j], [])->([i]) {i=4, j=8}> signature [parallel, reduce(max)]
func.func @reduce_max(%a: tensor<4x8xf32>) -> tensor<4xf32> {
  %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %0 = stablehlo.reduce(%a init: %init) applies stablehlo.maximum across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// A reshape merges two operand dimensions into one result dimension.
// CHECK: remark: grammar sdy.stablehlo.reshape axes #distributed.operator_axes<([i, j])->([ij]) {i=4, j=8}> signature [parallel, parallel]
func.func @reshape(%a: tensor<4x8xf32>) -> tensor<32xf32> {
  %0 = stablehlo.reshape %a : (tensor<4x8xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// -----

// A transpose permutes which dimension each axis is on. Canonicalized, the
// operand's axes come first and the permutation moves to the result.
// CHECK: remark: grammar sdy.stablehlo.transpose axes #distributed.operator_axes<([j, i])->([i, j]) {i=8, j=4}> signature [parallel, parallel]
// CANON: remark: grammar sdy.stablehlo.transpose
// CANON-NEXT: canonical: axes #distributed.operator_axes<([i, j])->([j, i]) {i=4, j=8}> signature [parallel, parallel]
func.func @transpose(%a: tensor<4x8xf32>) -> tensor<8x4xf32> {
  %0 = stablehlo.transpose %a, dims = [1, 0] : (tensor<4x8xf32>) -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// -----

// A sliced dimension keeps Shardy's factor on the operand, whose size it
// matches, and gets an appended local axis on the result.
// CHECK: remark: grammar sdy.stablehlo.slice axes #distributed.operator_axes<([i, j])->([i, k]) {i=4, j=8, k=4}> signature [parallel, local, local]
func.func @slice(%a: tensor<4x8xf32>) -> tensor<4x4xf32> {
  %0 = stablehlo.slice %a [0:4, 2:6] : (tensor<4x8xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// -----

// A padded dimension likewise gets an appended local axis on the result.
// CHECK: remark: grammar sdy.stablehlo.pad axes #distributed.operator_axes<([i], [])->([j]) {i=8, j=12}> signature [local, local]
func.func @pad(%a: tensor<8xf32>, %value: tensor<f32>) -> tensor<12xf32> {
  %0 = stablehlo.pad %a, %value, low = [1], high = [3], interior = [0] : (tensor<8xf32>, tensor<f32>) -> tensor<12xf32>
  return %0 : tensor<12xf32>
}

// -----

// A complex FFT is a dft: the transformed dimension's input and output axes
// are local, and the batch axis is carried through.
// CHECK: remark: grammar dft axes #distributed.operator_axes<([i, j])->([i, k]) {i=2, j=16, k=16}> signature [parallel, local, local]
// CHECK-NOT: params
func.func @fft(%x: tensor<2x16xcomplex<f32>>) -> tensor<2x16xcomplex<f32>> {
  %0 = stablehlo.fft %x, type = FFT, length = [16] : (tensor<2x16xcomplex<f32>>) -> tensor<2x16xcomplex<f32>>
  return %0 : tensor<2x16xcomplex<f32>>
}

// -----

// An IFFT is an inverse dft.
// CHECK: remark: grammar dft axes #distributed.operator_axes<([i])->([j]) {i=8, j=8}> signature [local, local] params {inverse = true}
func.func @ifft(%x: tensor<8xcomplex<f32>>) -> tensor<8xcomplex<f32>> {
  %0 = stablehlo.fft %x, type = IFFT, length = [8] : (tensor<8xcomplex<f32>>) -> tensor<8xcomplex<f32>>
  return %0 : tensor<8xcomplex<f32>>
}
