// RUN: enzymexlamlir-opt --split-input-file --verify-diagnostics %s

// Derivation IR's verifiers. Each case is a derivation container, a func.func
// with no arguments, except where a case needs a value no op of derivation IR
// defines, which it takes as a block argument: @negative_tensor_of_block_argument
// (a term_output) and @negative_input_not_tensor (a tensor).

//===----------------------------------------------------------------------===//
// distributed.Tensor
//===----------------------------------------------------------------------===//

// A tensor's axes are factors of LogicalMeshAxes roots.
!r4 = !distributed.replication_axis<4>
func.func @negative_tensor_axis_root() {
  %r = distributed.ReplicationAxis 4 : !r4
  %rf = axis.factor %r : !distributed.replication_axis<4><4, 1>
  // expected-error @+1 {{axis 0 must be an axis.factor of a distributed.LogicalMeshAxes root}}
  %x = distributed.Tensor axes(%rf : !axis.axis_factor<!r4, 4, 1>) : !distributed.tensor<f32>
  return
}

// -----

// A tensor can't hold a digit twice: a whole axis and one of its factors
// overlap.
!a8 = !distributed.logical_mesh_axis<8>
func.func @negative_tensor_axes_overlap() {
  %n = distributed.LogicalMeshAxes 8 : !a8
  %nf = axis.factor %n : !distributed.logical_mesh_axis<8><8, 1>
  %n1 = axis.factor %n : !distributed.logical_mesh_axis<8><2, 4>
  // expected-error @+1 {{axes 0 and 1 share a digit}}
  %x = distributed.Tensor axes(%nf : !axis.axis_factor<!a8, 8, 1>, %n1 : !axis.axis_factor<!a8, 2, 4>) : !distributed.tensor<f32>
  return
}

// -----

// Factors of one root whose cuts don't nest (2 and 3 on an axis of 6) have no
// common atoms.
!a6 = !distributed.logical_mesh_axis<6>
func.func @negative_tensor_cuts_dont_nest() {
  %n = distributed.LogicalMeshAxes 6 : !a6
  %n2 = axis.factor %n : !distributed.logical_mesh_axis<6><2, 1>
  %n3 = axis.factor %n : !distributed.logical_mesh_axis<6><3, 1>
  // expected-error @+1 {{has axes whose cuts on one root don't nest}}
  %x = distributed.Tensor axes(%n2 : !axis.axis_factor<!a6, 2, 1>, %n3 : !axis.axis_factor<!a6, 3, 1>) : !distributed.tensor<f32>
  return
}

// -----

// A tensor's `of` is a term output.
func.func @negative_tensor_of_not_term_output() {
  %x = distributed.Tensor axes() : !distributed.tensor<f32>
  // expected-error @+1 {{operand #0 must be Output of a grammar term}}
  %y = "distributed.Tensor"(%x) <{operandSegmentSizes = array<i32: 1, 0>}> : (!distributed.tensor<f32>) -> !distributed.tensor<f32>
  return
}

// -----

// A tensor's `of` is an output of a term, not any term_output value.
func.func @negative_tensor_of_block_argument(%o: !distributed.term_output) {
  // expected-error @+1 {{`of` must be an output of a distributed.GrammarTerm}}
  %y = distributed.Tensor of %o axes() : !distributed.tensor<f32>
  return
}

// -----

//===----------------------------------------------------------------------===//
// distributed.GrammarTerm
//===----------------------------------------------------------------------===//

// The kind must be non-empty.
func.func @negative_empty_kind() {
  %x = distributed.Tensor axes() : !distributed.tensor<f32>
  // expected-error @+1 {{requires a non-empty kind}}
  distributed.GrammarTerm ""(%x) axes() roles <([])->()> : (!distributed.tensor<f32>) -> ()
  return
}

// -----

// An input is a distributed.Tensor.
func.func @negative_input_not_tensor(%x: !distributed.tensor<f32>) {
  // expected-error @+1 {{input 0 must be a distributed.Tensor}}
  distributed.GrammarTerm "t"(%x) axes() roles <([])->()> : (!distributed.tensor<f32>) -> ()
  return
}

// -----

// A slot tensor with a factor that isn't derivation IR is reported by that
// Tensor alone: the term leaves its digit checks to valid tensors, so the
// output's missing role digit is not reported on the term.
!a4 = !distributed.logical_mesh_axis<4>
!r4 = !distributed.replication_axis<4>
!f4 = !axis.axis_factor<!a4, 4, 1>
func.func @negative_term_skips_malformed_tensor() {
  %i = distributed.LogicalMeshAxes 4 : !a4
  %r = distributed.ReplicationAxis 4 : !r4
  %if = axis.factor %i : !distributed.logical_mesh_axis<4><4, 1>
  %rf = axis.factor %r : !distributed.replication_axis<4><4, 1>
  %ii = axis.product(%if : !f4)
  %x = distributed.Tensor axes(%if : !f4) : !distributed.tensor<f32>
  %o = distributed.GrammarTerm "t"(%x) axes(%ii : !axis.factor_group<4>) roles <([i])->([i]) {i=4}> : (!distributed.tensor<f32>) -> !distributed.term_output
  // expected-error @+1 {{axis 0 must be an axis.factor of a distributed.LogicalMeshAxes root}}
  %y = distributed.Tensor of %o axes(%rf : !axis.axis_factor<!r4, 4, 1>) : !distributed.tensor<f32>
  return
}

// -----

// Likewise for a slot tensor whose own axes don't nest (2 and 3 on an axis of
// 6): only the Tensor reports it, not the term's digit checks.
!a6 = !distributed.logical_mesh_axis<6>
!f6 = !axis.axis_factor<!a6, 6, 1>
func.func @negative_term_skips_tensor_with_non_nesting_axes() {
  %n = distributed.LogicalMeshAxes 6 : !a6
  %nf = axis.factor %n : !distributed.logical_mesh_axis<6><6, 1>
  %n2 = axis.factor %n : !distributed.logical_mesh_axis<6><2, 1>
  %n3 = axis.factor %n : !distributed.logical_mesh_axis<6><3, 1>
  %ni = axis.product(%nf : !f6)
  %x = distributed.Tensor axes(%nf : !f6) : !distributed.tensor<f32>
  %o = distributed.GrammarTerm "t"(%x) axes(%ni : !axis.factor_group<6>) roles <([i])->([i]) {i=6}> : (!distributed.tensor<f32>) -> !distributed.term_output
  // expected-error @+1 {{has axes whose cuts on one root don't nest}}
  %y = distributed.Tensor of %o axes(%n2 : !axis.axis_factor<!a6, 2, 1>, %n3 : !axis.axis_factor<!a6, 3, 1>) : !distributed.tensor<f32>
  return
}

// -----

// An operator axis is a product of factors of LogicalMeshAxes roots.
!r4 = !distributed.replication_axis<4>
func.func @negative_operator_axis_root() {
  %r = distributed.ReplicationAxis 4 : !r4
  %rf = axis.factor %r : !distributed.replication_axis<4><4, 1>
  %ri = axis.product(%rf : !axis.axis_factor<!r4, 4, 1>)
  %x = distributed.Tensor axes() : !distributed.tensor<f32>
  // expected-error @+1 {{operator axis i must be an axis.product of axis factors of distributed.LogicalMeshAxes roots}}
  distributed.GrammarTerm "t"(%x) axes(%ri : !axis.factor_group<4>) roles <([i])->() {i=4}> : (!distributed.tensor<f32>) -> ()
  return
}

// -----

// There is one input slot per input.
func.func @negative_input_slot_count() {
  %x = distributed.Tensor axes() : !distributed.tensor<f32>
  // expected-error @+1 {{has roles for 2 input slots but 1 inputs}}
  distributed.GrammarTerm "t"(%x) axes() roles <([], [])->()> : (!distributed.tensor<f32>) -> ()
  return
}

// -----

// There is one output slot per output.
func.func @negative_output_slot_count() {
  %x = distributed.Tensor axes() : !distributed.tensor<f32>
  // expected-error @+1 {{has roles for 1 output slots but 0 outputs}}
  distributed.GrammarTerm "t"(%x) axes() roles <([])->([])> : (!distributed.tensor<f32>) -> ()
  return
}

// -----

// The roles name exactly the term's operator axes.
!a4 = !distributed.logical_mesh_axis<4>
func.func @negative_operator_axis_count() {
  %i = distributed.LogicalMeshAxes 4 : !a4
  %if = axis.factor %i : !distributed.logical_mesh_axis<4><4, 1>
  %ii = axis.product(%if : !axis.axis_factor<!a4, 4, 1>)
  %x = distributed.Tensor axes(%if : !axis.axis_factor<!a4, 4, 1>) : !distributed.tensor<f32>
  // expected-error @+1 {{has roles over 2 operator axes but 1 axis operands}}
  distributed.GrammarTerm "t"(%x) axes(%ii : !axis.factor_group<4>) roles <([i, j])->() {i=4, j=2}> : (!distributed.tensor<f32>) -> ()
  return
}

// -----

// A role axis has its operator axis's extent.
!a4 = !distributed.logical_mesh_axis<4>
func.func @negative_role_extent() {
  %i = distributed.LogicalMeshAxes 4 : !a4
  %if = axis.factor %i : !distributed.logical_mesh_axis<4><4, 1>
  %ii = axis.product(%if : !axis.axis_factor<!a4, 4, 1>)
  %x = distributed.Tensor axes(%if : !axis.axis_factor<!a4, 4, 1>) : !distributed.tensor<f32>
  // expected-error @+1 {{operator axis i has extent 8 in the roles but its product has extent 4}}
  distributed.GrammarTerm "t"(%x) axes(%ii : !axis.factor_group<4>) roles <([i])->() {i=8}> : (!distributed.tensor<f32>) -> ()
  return
}

// -----

// Each output has exactly one use, its Tensor; here it has two.
!a4 = !distributed.logical_mesh_axis<4>
!f4 = !axis.axis_factor<!a4, 4, 1>
func.func @negative_term_output_two_uses() {
  %i = distributed.LogicalMeshAxes 4 : !a4
  %if = axis.factor %i : !distributed.logical_mesh_axis<4><4, 1>
  %ii = axis.product(%if : !f4)
  %x = distributed.Tensor axes(%if : !f4) : !distributed.tensor<f32>
  // expected-error @+1 {{output 0 must have exactly one use, a distributed.Tensor of it}}
  %o = distributed.GrammarTerm "t"(%x) axes(%ii : !axis.factor_group<4>) roles <([i])->([i]) {i=4}> : (!distributed.tensor<f32>) -> !distributed.term_output
  %y = distributed.Tensor of %o axes(%if : !f4) : !distributed.tensor<f32>
  %z = distributed.Tensor of %o axes(%if : !f4) : !distributed.tensor<f32>
  return
}

// -----

// Two operator axes can't share a digit.
!a4 = !distributed.logical_mesh_axis<4>
!f4 = !axis.axis_factor<!a4, 4, 1>
func.func @negative_operator_axes_overlap() {
  %i = distributed.LogicalMeshAxes 4 : !a4
  %if = axis.factor %i : !distributed.logical_mesh_axis<4><4, 1>
  %ii = axis.product(%if : !f4)
  %jj = axis.product(%if : !f4)
  %x = distributed.Tensor axes(%if : !f4) : !distributed.tensor<f32>
  // expected-error @+1 {{operator axes i and j share a digit}}
  %o = distributed.GrammarTerm "t"(%x) axes(%ii : !axis.factor_group<4>, %jj : !axis.factor_group<4>) roles <([i])->([j]) {i=4, j=4}> : (!distributed.tensor<f32>) -> !distributed.term_output
  %y = distributed.Tensor of %o axes(%if : !f4) : !distributed.tensor<f32>
  return
}

// -----

// One operator axis can't hold a digit twice.
!a4 = !distributed.logical_mesh_axis<4>
!a2 = !distributed.logical_mesh_axis<2>
!f2 = !axis.axis_factor<!a2, 2, 1>
func.func @negative_operator_axis_repeats_digit() {
  %i = distributed.LogicalMeshAxes 2 : !a2
  %if = axis.factor %i : !distributed.logical_mesh_axis<2><2, 1>
  %ii = axis.product(%if : !f2, %if : !f2)
  %x = distributed.Tensor axes(%if : !f2) : !distributed.tensor<f32>
  // expected-error @+1 {{operator axis i holds a digit twice}}
  %o = distributed.GrammarTerm "t"(%x) axes(%ii : !axis.factor_group<4>) roles <([i])->([i]) {i=4}> : (!distributed.tensor<f32>) -> !distributed.term_output
  %y = distributed.Tensor of %o axes(%if : !f2) : !distributed.tensor<f32>
  return
}

// -----

// An operator axis and a tensor whose cuts on one root don't nest: the term
// reads the minor 2 of an axis of 6, the tensor holds its minor 3.
!a6 = !distributed.logical_mesh_axis<6>
func.func @negative_term_cuts_dont_nest() {
  %n = distributed.LogicalMeshAxes 6 : !a6
  %n2 = axis.factor %n : !distributed.logical_mesh_axis<6><2, 1>
  %n3 = axis.factor %n : !distributed.logical_mesh_axis<6><3, 1>
  %ii = axis.product(%n2 : !axis.axis_factor<!a6, 2, 1>)
  %x = distributed.Tensor axes(%n3 : !axis.axis_factor<!a6, 3, 1>) : !distributed.tensor<f32>
  // expected-error @+1 {{has operator axes and tensors whose cuts on one root don't nest}}
  distributed.GrammarTerm "t"(%x) axes(%ii : !axis.factor_group<2>) roles <([i])->() {i=2}> : (!distributed.tensor<f32>) -> ()
  return
}

// -----

// A slot's tensor holds its role digits; this input holds only half of i.
!a4 = !distributed.logical_mesh_axis<4>
!f4 = !axis.axis_factor<!a4, 4, 1>
func.func @negative_tensor_lacks_role_digit() {
  %i = distributed.LogicalMeshAxes 4 : !a4
  %if = axis.factor %i : !distributed.logical_mesh_axis<4><4, 1>
  %i1 = axis.factor %i : !distributed.logical_mesh_axis<4><2, 2>
  %ii = axis.product(%if : !f4)
  %x = distributed.Tensor axes(%i1 : !axis.axis_factor<!a4, 2, 2>) : !distributed.tensor<f32>
  // expected-error @+1 {{input 0's tensor lacks some digit of its roles}}
  %o = distributed.GrammarTerm "t"(%x) axes(%ii : !axis.factor_group<4>) roles <([i])->([i]) {i=4}> : (!distributed.tensor<f32>) -> !distributed.term_output
  %y = distributed.Tensor of %o axes(%if : !f4) : !distributed.tensor<f32>
  return
}

// -----

// A slot's tensor can't carry another slot's role digit: the DFT's output
// holds its input digit n as if it were batch.
!a8 = !distributed.logical_mesh_axis<8>
!f8 = !axis.axis_factor<!a8, 8, 1>
func.func @negative_tensor_holds_other_role_digit() {
  %n = distributed.LogicalMeshAxes 8 : !a8
  %k = distributed.LogicalMeshAxes 8 : !a8
  %nf = axis.factor %n : !distributed.logical_mesh_axis<8><8, 1>
  %kf = axis.factor %k : !distributed.logical_mesh_axis<8><8, 1>
  %ni = axis.product(%nf : !f8)
  %ki = axis.product(%kf : !f8)
  %x = distributed.Tensor axes(%nf : !f8) : !distributed.tensor<complex<f32>>
  // expected-error @+1 {{output 0's tensor holds a role digit of another slot}}
  %o = distributed.GrammarTerm "dft"(%x) axes(%ni : !axis.factor_group<8>, %ki : !axis.factor_group<8>) roles <([i])->([j]) {i=8, j=8}> : (!distributed.tensor<complex<f32>>) -> !distributed.term_output
  %y = distributed.Tensor of %o axes(%kf : !f8, %nf : !f8) : !distributed.tensor<complex<f32>>
  return
}

// -----

// An input's batch digit reaches some output; dropping it would be a
// reduction.
!a4 = !distributed.logical_mesh_axis<4>
!a2 = !distributed.logical_mesh_axis<2>
!f4 = !axis.axis_factor<!a4, 4, 1>
!f2 = !axis.axis_factor<!a2, 2, 1>
func.func @negative_batch_dropped() {
  %i = distributed.LogicalMeshAxes 4 : !a4
  %b = distributed.LogicalMeshAxes 2 : !a2
  %if = axis.factor %i : !distributed.logical_mesh_axis<4><4, 1>
  %bf = axis.factor %b : !distributed.logical_mesh_axis<2><2, 1>
  %ii = axis.product(%if : !f4)
  %x = distributed.Tensor axes(%if : !f4, %bf : !f2) : !distributed.tensor<f32>
  // expected-error @+1 {{input 0 has a batch digit that no output tensor holds}}
  %o = distributed.GrammarTerm "t"(%x) axes(%ii : !axis.factor_group<4>) roles <([i])->([i]) {i=4}> : (!distributed.tensor<f32>) -> !distributed.term_output
  %y = distributed.Tensor of %o axes(%if : !f4) : !distributed.tensor<f32>
  return
}

// -----

// A dft dimension is batch or transformed as a whole: [b n] mixes the
// both-sided b with the input-only n.
!a8 = !distributed.logical_mesh_axis<8>
!a2 = !distributed.logical_mesh_axis<2>
!f8 = !axis.axis_factor<!a8, 8, 1>
!f2 = !axis.axis_factor<!a2, 2, 1>
func.func @negative_dft_mixed_dimension() {
  %b = distributed.LogicalMeshAxes 2 : !a2
  %n = distributed.LogicalMeshAxes 8 : !a8
  %k = distributed.LogicalMeshAxes 8 : !a8
  %bf = axis.factor %b : !distributed.logical_mesh_axis<2><2, 1>
  %nf = axis.factor %n : !distributed.logical_mesh_axis<8><8, 1>
  %kf = axis.factor %k : !distributed.logical_mesh_axis<8><8, 1>
  %bi = axis.product(%bf : !f2)
  %ni = axis.product(%nf : !f8)
  %ki = axis.product(%kf : !f8)
  %x = distributed.Tensor axes(%bf : !f2, %nf : !f8) : !distributed.tensor<complex<f32>>
  // expected-error @+1 {{input 0 dimension 0 mixes axes on both sides with axes on one side, which a dft doesn't allow}}
  %o = distributed.GrammarTerm "dft"(%x) axes(%bi : !axis.factor_group<2>, %ni : !axis.factor_group<8>, %ki : !axis.factor_group<8>) roles <([ij])->([i, k]) {i=2, j=8, k=8}> : (!distributed.tensor<complex<f32>>) -> !distributed.term_output
  %y = distributed.Tensor of %o axes(%bf : !f2, %kf : !f8) : !distributed.tensor<complex<f32>>
  return
}

// -----

//===----------------------------------------------------------------------===//
// #distributed.operator_axes (the roles)
//===----------------------------------------------------------------------===//

// An operator axis must have extent at least 1.
func.func @negative_zero_extent() {
  %x = distributed.Tensor axes() : !distributed.tensor<f32>
  // expected-error @+1 {{operator axis j has extent 0, expected at least 1}}
  distributed.GrammarTerm "t"(%x) axes() roles <([i])->() {i=4, j=0}> : (!distributed.tensor<f32>) -> ()
  return
}

// -----

// Axis indices must name a declared axis.
func.func @negative_axis_out_of_range() {
  %x = distributed.Tensor axes() : !distributed.tensor<f32>
  // expected-error @+1 {{result 0 dimension 0 maps operator axis index 1, expected less than 1}}
  distributed.GrammarTerm "t"(%x) axes() roles <([i])->([j]) {i=4}> : (!distributed.tensor<f32>) -> ()
  return
}

// -----

// Every operator axis must occur on some slot.
func.func @negative_unused_axis() {
  %x = distributed.Tensor axes() : !distributed.tensor<f32>
  // expected-error @+1 {{operator axis j is mapped by no operand or result}}
  distributed.GrammarTerm "t"(%x) axes() roles <([i])->() {i=4, j=2}> : (!distributed.tensor<f32>) -> ()
  return
}

// -----

// One slot may not use an axis in two dimensions.
func.func @negative_axis_reused_across_dims() {
  %x = distributed.Tensor axes() : !distributed.tensor<f32>
  // expected-error @+1 {{operand 0 maps operator axis i more than once}}
  distributed.GrammarTerm "t"(%x) axes() roles <([i, i])->() {i=4}> : (!distributed.tensor<f32>) -> ()
  return
}

// -----

// One slot may not use an axis twice within a dimension.
func.func @negative_axis_reused_in_dim() {
  %x = distributed.Tensor axes() : !distributed.tensor<f32>
  // expected-error @+1 {{operand 0 maps operator axis i more than once}}
  distributed.GrammarTerm "t"(%x) axes() roles <([ii])->() {i=4}> : (!distributed.tensor<f32>) -> ()
  return
}

// -----

// A dimension split into several axes may not include an extent-1 axis.
func.func @negative_unit_digit() {
  %x = distributed.Tensor axes() : !distributed.tensor<f32>
  // expected-error @+1 {{operand 0 dimension 0 splits into several operator axes, but i has extent 1}}
  distributed.GrammarTerm "t"(%x) axes() roles <([ij])->() {i=1, j=4}> : (!distributed.tensor<f32>) -> ()
  return
}

// -----

// The only integer allowed in a dimension is 1.
func.func @negative_dim_integer() {
  %x = distributed.Tensor axes() : !distributed.tensor<f32>
  // expected-error @+1 {{expected operator axes or 1 for a dimension, got 2}}
  distributed.GrammarTerm "t"(%x) axes() roles <([i, 2])->() {i=4}> : (!distributed.tensor<f32>) -> ()
  return
}

// -----

// A dimension needs at least one token.
func.func @negative_empty_dim() {
  %x = distributed.Tensor axes() : !distributed.tensor<f32>
  // expected-error @+1 {{expected operator axes or 1 for a dimension}}
  distributed.GrammarTerm "t"(%x) axes() roles <([i, ])->() {i=4}> : (!distributed.tensor<f32>) -> ()
  return
}
