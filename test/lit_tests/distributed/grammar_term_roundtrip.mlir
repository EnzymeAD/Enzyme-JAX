// RUN: enzymexlamlir-opt --split-input-file %s | FileCheck %s
// A roundtrip is only complete if the printed form parses back, so run the
// printer output through the parser a second time.
// RUN: enzymexlamlir-opt --split-input-file %s | enzymexlamlir-opt --split-input-file | FileCheck %s
// RUN: enzymexlamlir-opt --split-input-file --cse %s | FileCheck %s --check-prefix=CSE

// Derivation IR: grammar terms over layout-free tensors. Each case is a
// derivation container: a func.func with no arguments that creates its axes,
// boundary input tensors and terms, and returns its output tensors.

// A batched DFT: the input holds n and the batch b, the output holds k and b.
// Params are kept as given. This case pins the term's full printed form.
// CHECK-LABEL: func.func @batched_term
// CHECK: %[[N:.*]] = axis.factor
// CHECK: %[[K:.*]] = axis.factor
// CHECK: %[[B:.*]] = axis.factor
// CHECK: %[[NI:.*]] = axis.product
// CHECK: %[[KI:.*]] = axis.product
// CHECK: %[[X:.*]] = distributed.Tensor axes (%[[N]] : {{.*}}, %[[B]] :
// CHECK: %[[O:.*]] = distributed.GrammarTerm "dft"(%[[X]]) axes (%[[NI]] : !axis.factor_group<8>, %[[KI]] : !axis.factor_group<8>) roles <([i])->([j]) {i=8, j=8}> params {inverse = true} : (!distributed.tensor<complex<f32>>) -> !distributed.term_output
// CHECK: %[[Y:.*]] = distributed.Tensor of %[[O]] axes (%[[K]] : {{.*}}, %[[B]] :
// CHECK: return %[[Y]]
!a8 = !distributed.logical_mesh_axis<8>
!a4 = !distributed.logical_mesh_axis<4>
!f8 = !axis.axis_factor<!a8, 8, 1>
!f4 = !axis.axis_factor<!a4, 4, 1>
func.func @batched_term() -> !distributed.tensor<complex<f32>> {
  %n = distributed.LogicalMeshAxes 8 : !a8
  %k = distributed.LogicalMeshAxes 8 : !a8
  %b = distributed.LogicalMeshAxes 4 : !a4
  %nf = axis.factor %n : !distributed.logical_mesh_axis<8><8, 1>
  %kf = axis.factor %k : !distributed.logical_mesh_axis<8><8, 1>
  %bf = axis.factor %b : !distributed.logical_mesh_axis<4><4, 1>
  %ni = axis.product(%nf : !f8)
  %ki = axis.product(%kf : !f8)
  %x = distributed.Tensor axes(%nf : !f8, %bf : !f4) : !distributed.tensor<complex<f32>>
  %o = distributed.GrammarTerm "dft"(%x) axes(%ni : !axis.factor_group<8>, %ki : !axis.factor_group<8>) roles <([i])->([j]) {i=8, j=8}> params {inverse = true} : (!distributed.tensor<complex<f32>>) -> !distributed.term_output
  %y = distributed.Tensor of %o axes(%kf : !f8, %bf : !f4) : !distributed.tensor<complex<f32>>
  return %y : !distributed.tensor<complex<f32>>
}

// -----

// A container chaining two terms returns its final outputs through
// func.return: the DFT's output feeds the twiddle, whose output is returned.
// CHECK-LABEL: func.func @container_returns_outputs()
// CHECK: %[[Y:.*]] = distributed.Tensor of %{{.*}} axes
// CHECK: %[[O:.*]] = distributed.GrammarTerm "twiddle"(%[[Y]])
// CHECK: %[[Z:.*]] = distributed.Tensor of %[[O]] axes
// CHECK: return %[[Z]] :
!a8 = !distributed.logical_mesh_axis<8>
!f8 = !axis.axis_factor<!a8, 8, 1>
func.func @container_returns_outputs() -> !distributed.tensor<complex<f32>> {
  %n = distributed.LogicalMeshAxes 8 : !a8
  %k = distributed.LogicalMeshAxes 8 : !a8
  %nf = axis.factor %n : !distributed.logical_mesh_axis<8><8, 1>
  %kf = axis.factor %k : !distributed.logical_mesh_axis<8><8, 1>
  %ni = axis.product(%nf : !f8)
  %ki = axis.product(%kf : !f8)
  %x = distributed.Tensor axes(%nf : !f8) : !distributed.tensor<complex<f32>>
  %o = distributed.GrammarTerm "dft"(%x) axes(%ni : !axis.factor_group<8>, %ki : !axis.factor_group<8>) roles <([i])->([j]) {i=8, j=8}> : (!distributed.tensor<complex<f32>>) -> !distributed.term_output
  %y = distributed.Tensor of %o axes(%kf : !f8) : !distributed.tensor<complex<f32>>
  %p = distributed.GrammarTerm "twiddle"(%y) axes(%ki : !axis.factor_group<8>) roles <([i])->([i]) {i=8}> : (!distributed.tensor<complex<f32>>) -> !distributed.term_output
  %z = distributed.Tensor of %p axes(%kf : !f8) : !distributed.tensor<complex<f32>>
  return %z : !distributed.tensor<complex<f32>>
}

// -----

// A weight shared across the batch: w holds only role axes, so it is
// broadcast over b, which x and the output carry.
// CHECK-LABEL: func.func @broadcast_input
// CHECK: %[[W:.*]] = distributed.Tensor axes
// CHECK: %[[X:.*]] = distributed.Tensor axes
// CHECK: %[[O:.*]] = distributed.GrammarTerm "matvec"(%[[W]], %[[X]]) {{.*}} roles <([i, j], [j])->([i]) {i=4, j=8}>
// CHECK: distributed.Tensor of %[[O]]
!a8 = !distributed.logical_mesh_axis<8>
!a4 = !distributed.logical_mesh_axis<4>
!a2 = !distributed.logical_mesh_axis<2>
!f8 = !axis.axis_factor<!a8, 8, 1>
!f4 = !axis.axis_factor<!a4, 4, 1>
!f2 = !axis.axis_factor<!a2, 2, 1>
func.func @broadcast_input() -> !distributed.tensor<f32> {
  %i = distributed.LogicalMeshAxes 4 : !a4
  %j = distributed.LogicalMeshAxes 8 : !a8
  %b = distributed.LogicalMeshAxes 2 : !a2
  %if = axis.factor %i : !distributed.logical_mesh_axis<4><4, 1>
  %jf = axis.factor %j : !distributed.logical_mesh_axis<8><8, 1>
  %bf = axis.factor %b : !distributed.logical_mesh_axis<2><2, 1>
  %ii = axis.product(%if : !f4)
  %ji = axis.product(%jf : !f8)
  %w = distributed.Tensor axes(%if : !f4, %jf : !f8) : !distributed.tensor<f32>
  %x = distributed.Tensor axes(%jf : !f8, %bf : !f2) : !distributed.tensor<f32>
  %o = distributed.GrammarTerm "matvec"(%w, %x) axes(%ii : !axis.factor_group<4>, %ji : !axis.factor_group<8>) roles <([i, j], [j])->([i]) {i=4, j=8}> : (!distributed.tensor<f32>, !distributed.tensor<f32>) -> !distributed.term_output
  %y = distributed.Tensor of %o axes(%if : !f4, %bf : !f2) : !distributed.tensor<f32>
  return %y : !distributed.tensor<f32>
}

// -----

// A term reading one digit of a factored axis: x holds the whole of j, the
// term transforms its major digit j1, and the minor digit j2 is batch, which
// the output carries.
// CHECK-LABEL: func.func @factored_axis
// CHECK: %[[J:.*]] = axis.factor
// CHECK: %[[J1:.*]] = axis.factor
// CHECK: %[[J2:.*]] = axis.factor
// CHECK: %[[JI:.*]] = axis.product (%[[J1]] :
// CHECK: %[[X:.*]] = distributed.Tensor axes (%[[J]] :
// CHECK: %[[O:.*]] = distributed.GrammarTerm "dft"(%[[X]]) axes (%[[JI]] : {{.*}}) roles <([i])->([j]) {i=2, j=2}>
// CHECK: distributed.Tensor of %[[O]] axes (%{{.*}} : {{.*}}, %[[J2]] :
!a8 = !distributed.logical_mesh_axis<8>
!a2 = !distributed.logical_mesh_axis<2>
func.func @factored_axis() -> !distributed.tensor<complex<f32>> {
  %j = distributed.LogicalMeshAxes 8 : !a8
  %k = distributed.LogicalMeshAxes 2 : !a2
  %jf = axis.factor %j : !distributed.logical_mesh_axis<8><8, 1>
  %j1 = axis.factor %j : !distributed.logical_mesh_axis<8><2, 4>
  %j2 = axis.factor %j : !distributed.logical_mesh_axis<8><4, 1>
  %kf = axis.factor %k : !distributed.logical_mesh_axis<2><2, 1>
  %ji = axis.product(%j1 : !axis.axis_factor<!a8, 2, 4>)
  %ki = axis.product(%kf : !axis.axis_factor<!a2, 2, 1>)
  %x = distributed.Tensor axes(%jf : !axis.axis_factor<!a8, 8, 1>) : !distributed.tensor<complex<f32>>
  %o = distributed.GrammarTerm "dft"(%x) axes(%ji : !axis.factor_group<2>, %ki : !axis.factor_group<2>) roles <([i])->([j]) {i=2, j=2}> : (!distributed.tensor<complex<f32>>) -> !distributed.term_output
  %y = distributed.Tensor of %o axes(%kf : !axis.axis_factor<!a2, 2, 1>, %j2 : !axis.axis_factor<!a8, 4, 1>) : !distributed.tensor<complex<f32>>
  return %y : !distributed.tensor<complex<f32>>
}

// -----

// One operator axis made of digits of two roots: the major digit of a and the
// minor digit of b. The tensors hold both roots whole; the other digits are
// batch.
// CHECK-LABEL: func.func @product_of_two_roots
// CHECK: %[[I:.*]] = axis.product
// CHECK: distributed.GrammarTerm "twiddle"(%{{.*}}) axes (%[[I]] : {{.*}}) roles <([i])->([i]) {i=4}>
!a4 = !distributed.logical_mesh_axis<4>
!f4 = !axis.axis_factor<!a4, 4, 1>
func.func @product_of_two_roots() -> !distributed.tensor<complex<f32>> {
  %a = distributed.LogicalMeshAxes 4 : !a4
  %b = distributed.LogicalMeshAxes 4 : !a4
  %af = axis.factor %a : !distributed.logical_mesh_axis<4><4, 1>
  %bf = axis.factor %b : !distributed.logical_mesh_axis<4><4, 1>
  %a1 = axis.factor %a : !distributed.logical_mesh_axis<4><2, 2>
  %b0 = axis.factor %b : !distributed.logical_mesh_axis<4><2, 1>
  %i = axis.product(%a1 : !axis.axis_factor<!a4, 2, 2>, %b0 : !axis.axis_factor<!a4, 2, 1>)
  %x = distributed.Tensor axes(%af : !f4, %bf : !f4) : !distributed.tensor<complex<f32>>
  %o = distributed.GrammarTerm "twiddle"(%x) axes(%i : !axis.factor_group<4>) roles <([i])->([i]) {i=4}> : (!distributed.tensor<complex<f32>>) -> !distributed.term_output
  %y = distributed.Tensor of %o axes(%af : !f4, %bf : !f4) : !distributed.tensor<complex<f32>>
  return %y : !distributed.tensor<complex<f32>>
}

// -----

// A transpose reads its two operator axes in opposite orders on its two
// slots, which the axis sets alone can't express.
// CHECK-LABEL: func.func @transpose
// CHECK: %[[I:.*]] = axis.product
// CHECK: %[[J:.*]] = axis.product
// CHECK: distributed.GrammarTerm "transpose"(%{{.*}}) axes (%[[I]] : {{.*}}, %[[J]] : {{.*}}) roles <([i, j])->([j, i]) {i=4, j=8}>
!a8 = !distributed.logical_mesh_axis<8>
!a4 = !distributed.logical_mesh_axis<4>
!f8 = !axis.axis_factor<!a8, 8, 1>
!f4 = !axis.axis_factor<!a4, 4, 1>
func.func @transpose() -> !distributed.tensor<f32> {
  %i = distributed.LogicalMeshAxes 4 : !a4
  %j = distributed.LogicalMeshAxes 8 : !a8
  %if = axis.factor %i : !distributed.logical_mesh_axis<4><4, 1>
  %jf = axis.factor %j : !distributed.logical_mesh_axis<8><8, 1>
  %ii = axis.product(%if : !f4)
  %ji = axis.product(%jf : !f8)
  %x = distributed.Tensor axes(%if : !f4, %jf : !f8) : !distributed.tensor<f32>
  %o = distributed.GrammarTerm "transpose"(%x) axes(%ii : !axis.factor_group<4>, %ji : !axis.factor_group<8>) roles <([i, j])->([j, i]) {i=4, j=8}> : (!distributed.tensor<f32>) -> !distributed.term_output
  %y = distributed.Tensor of %o axes(%jf : !f8, %if : !f4) : !distributed.tensor<f32>
  return %y : !distributed.tensor<f32>
}

// -----

// A term with no inputs, and a tensor with no axes read by an empty role.
// CHECK-LABEL: func.func @empty_slots
// CHECK: %[[S:.*]] = distributed.Tensor axes ()
// CHECK: distributed.GrammarTerm "iota"() axes ({{.*}}) roles <()->([i]) {i=4}> : () -> !distributed.term_output
// CHECK: distributed.GrammarTerm "scale"(%[[S]], %{{.*}}) {{.*}} roles <([], [i])->([i]) {i=4}>
!a4 = !distributed.logical_mesh_axis<4>
!f4 = !axis.axis_factor<!a4, 4, 1>
func.func @empty_slots() -> !distributed.tensor<f32> {
  %i = distributed.LogicalMeshAxes 4 : !a4
  %if = axis.factor %i : !distributed.logical_mesh_axis<4><4, 1>
  %ii = axis.product(%if : !f4)
  %s = distributed.Tensor axes() : !distributed.tensor<f32>
  %o = distributed.GrammarTerm "iota"() axes(%ii : !axis.factor_group<4>) roles <()->([i]) {i=4}> : () -> !distributed.term_output
  %x = distributed.Tensor of %o axes(%if : !f4) : !distributed.tensor<f32>
  %p = distributed.GrammarTerm "scale"(%s, %x) axes(%ii : !axis.factor_group<4>) roles <([], [i])->([i]) {i=4}> : (!distributed.tensor<f32>, !distributed.tensor<f32>) -> !distributed.term_output
  %y = distributed.Tensor of %p axes(%if : !f4) : !distributed.tensor<f32>
  return %y : !distributed.tensor<f32>
}

// -----

// Roles keep their dimension syntax: a dimension with no operator axes is
// written `1`, and redundant `1`s within a dimension print canonically.
// CHECK-LABEL: func.func @role_dimension_syntax
// CHECK: distributed.GrammarTerm "reshape"({{.*}} roles <([i, 1])->([1, i]) {i=4}>
// CHECK: distributed.GrammarTerm "t"({{.*}} roles <([ij, 1])->([ij, 1]) {i=2, j=4}>
!a4 = !distributed.logical_mesh_axis<4>
!a2 = !distributed.logical_mesh_axis<2>
!f4 = !axis.axis_factor<!a4, 4, 1>
!f2 = !axis.axis_factor<!a2, 2, 1>
func.func @role_dimension_syntax() -> (!distributed.tensor<f32>, !distributed.tensor<f32>) {
  %i = distributed.LogicalMeshAxes 4 : !a4
  %j = distributed.LogicalMeshAxes 2 : !a2
  %if = axis.factor %i : !distributed.logical_mesh_axis<4><4, 1>
  %jf = axis.factor %j : !distributed.logical_mesh_axis<2><2, 1>
  %ii = axis.product(%if : !f4)
  %ji = axis.product(%jf : !f2)
  %x = distributed.Tensor axes(%if : !f4) : !distributed.tensor<f32>
  %o = distributed.GrammarTerm "reshape"(%x) axes(%ii : !axis.factor_group<4>) roles <([i, 1])->([1, i]) {i=4}> : (!distributed.tensor<f32>) -> !distributed.term_output
  %y = distributed.Tensor of %o axes(%if : !f4) : !distributed.tensor<f32>
  %z = distributed.Tensor axes(%jf : !f2, %if : !f4) : !distributed.tensor<f32>
  %p = distributed.GrammarTerm "t"(%z) axes(%ji : !axis.factor_group<2>, %ii : !axis.factor_group<4>) roles <([1 ij 1, 1 1])->([i 1 j, 1]) {i=2, j=4}> : (!distributed.tensor<f32>) -> !distributed.term_output
  %w = distributed.Tensor of %p axes(%jf : !f2, %if : !f4) : !distributed.tensor<f32>
  return %y, %w : !distributed.tensor<f32>, !distributed.tensor<f32>
}

// -----

// Two boundary inputs over the same axes are two tensors, and CSE keeps both.
// CSE-LABEL: func.func @distinct_boundary_inputs
// CSE: %[[X:.*]] = distributed.Tensor axes
// CSE: %[[Y:.*]] = distributed.Tensor axes
// CSE: distributed.GrammarTerm "add"(%[[X]], %[[Y]])
!a4 = !distributed.logical_mesh_axis<4>
!f4 = !axis.axis_factor<!a4, 4, 1>
func.func @distinct_boundary_inputs() -> !distributed.tensor<f32> {
  %i = distributed.LogicalMeshAxes 4 : !a4
  %if = axis.factor %i : !distributed.logical_mesh_axis<4><4, 1>
  %ii = axis.product(%if : !f4)
  %x = distributed.Tensor axes(%if : !f4) : !distributed.tensor<f32>
  %y = distributed.Tensor axes(%if : !f4) : !distributed.tensor<f32>
  %o = distributed.GrammarTerm "add"(%x, %y) axes(%ii : !axis.factor_group<4>) roles <([i], [i])->([i]) {i=4}> : (!distributed.tensor<f32>, !distributed.tensor<f32>) -> !distributed.term_output
  %z = distributed.Tensor of %o axes(%if : !f4) : !distributed.tensor<f32>
  return %z : !distributed.tensor<f32>
}
