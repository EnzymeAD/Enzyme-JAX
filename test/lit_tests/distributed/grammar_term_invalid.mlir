// RUN: enzymexlamlir-opt --split-input-file --verify-diagnostics %s

// An operator axis must have extent at least 1.
func.func @negative_zero_extent(%x: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{operator axis j has extent 0, expected at least 1}}
  %0 = distributed.GrammarTerm "t" (%x) axes <([i])->([i]) {i=4, j=0}> : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// Axis indices must name a declared axis.
func.func @negative_axis_out_of_range(%x: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{result 0 dimension 0 maps operator axis index 1, expected less than 1}}
  %0 = distributed.GrammarTerm "t" (%x) axes <([i])->([j]) {i=4}> : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// Every operator axis must occur on some operand or result.
func.func @negative_unused_axis(%x: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{operator axis j is mapped by no operand or result}}
  %0 = distributed.GrammarTerm "t" (%x) axes <([i])->([i]) {i=4, j=2}> : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// One tensor may not use an axis in two dimensions.
func.func @negative_axis_reused_across_dims(%x: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // expected-error @+1 {{operand 0 maps operator axis i more than once}}
  %0 = distributed.GrammarTerm "t" (%x) axes <([i, i])->([i, j]) {i=4, j=4}> : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// -----

// One tensor may not use an axis twice within a dimension.
func.func @negative_axis_reused_in_dim(%x: tensor<16xf32>) -> tensor<16xf32> {
  // expected-error @+1 {{operand 0 maps operator axis i more than once}}
  %0 = distributed.GrammarTerm "t" (%x) axes <([ii])->([i]) {i=4}> : (tensor<16xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// -----

// A dimension split into several axes may not include an extent-1 axis.
func.func @negative_unit_digit(%x: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{operand 0 dimension 0 splits into several operator axes, but i has extent 1}}
  %0 = distributed.GrammarTerm "t" (%x) axes <([ij])->([j]) {i=1, j=4}> : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// The kind must be non-empty.
func.func @negative_empty_kind(%x: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{requires a non-empty kind}}
  %0 = distributed.GrammarTerm "" (%x) axes <([i])->([i]) {i=4}> : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// There is one operand mapping per operand.
func.func @negative_operand_mapping_count(%x: tensor<4xf32>, %y: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{has 1 operand mappings for 2 operands}}
  %0 = distributed.GrammarTerm "t" (%x, %y) axes <([i])->([i]) {i=4}> : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// There is one result mapping per result.
func.func @negative_result_mapping_count(%x: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{has 2 result mappings for 1 results}}
  %0 = distributed.GrammarTerm "t" (%x) axes <([i])->([i], [i]) {i=4}> : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// A mapping's rank must equal its tensor's rank.
func.func @negative_rank_mismatch(%x: tensor<4x2xf32>) -> tensor<4x2xf32> {
  // expected-error @+1 {{operand 0 has rank 2 but its mapping has rank 1}}
  %0 = distributed.GrammarTerm "t" (%x) axes <([ij])->([i, j]) {i=4, j=2}> : (tensor<4x2xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// -----

// Tensors must have static shapes.
func.func @negative_dynamic_shape(%x: tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{operand #0 must be variadic of statically shaped tensor}}
  %0 = distributed.GrammarTerm "t" (%x) axes <([i])->([i]) {i=4}> : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// A dimension's size must equal the product of its axes' extents.
func.func @negative_extent_product(%x: tensor<32xf32>) -> tensor<32xf32> {
  // expected-error @+1 {{result 0 dimension 0 has size 32 but its operator axes' extents multiply to 16}}
  %0 = distributed.GrammarTerm "t" (%x) axes <([ij])->([j]) {i=2, j=16}> : (tensor<32xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// -----

// A dimension with no operator axes has size 1.
func.func @negative_digitless_dim_size(%x: tensor<4x2xf32>) -> tensor<4x2xf32> {
  // expected-error @+1 {{operand 0 dimension 1 has size 2 but its operator axes' extents multiply to 1}}
  %0 = distributed.GrammarTerm "t" (%x) axes <([i, 1])->([i, j]) {i=4, j=2}> : (tensor<4x2xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// -----

// The only integer allowed in a dimension is 1.
func.func @negative_dim_integer(%x: tensor<4x2xf32>) -> tensor<4x2xf32> {
  // expected-error @+1 {{expected operator axes or 1 for a dimension, got 2}}
  %0 = distributed.GrammarTerm "t" (%x) axes <([i, 2])->([i, j]) {i=4, j=2}> : (tensor<4x2xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// -----

// A dimension needs at least one token.
func.func @negative_empty_dim(%x: tensor<4x1xf32>) -> tensor<4x1xf32> {
  // expected-error @+1 {{expected operator axes or 1 for a dimension}}
  %0 = distributed.GrammarTerm "t" (%x) axes <([i, ])->([i, 1]) {i=4}> : (tensor<4x1xf32>) -> tensor<4x1xf32>
  return %0 : tensor<4x1xf32>
}
