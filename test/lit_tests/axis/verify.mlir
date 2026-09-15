// RUN: enzymexlamlir-opt --split-input-file --verify-diagnostics %s

func.func @getaxis_valid() -> !axis.shape_axis<tensor<8x4xf32>, 1> {
  %axis = axis.getaxis tensor<8x4xf32> 1
  return %axis : !axis.shape_axis<tensor<8x4xf32>, 1>
}

// -----

func.func @getaxis_requires_shaped_type() {
  // expected-error @+1 {{requires shape_type to be a shaped type}}
  %axis = axis.getaxis i32 0
  return
}

// -----

func.func @getaxis_requires_ranked_type() {
  // expected-error @+1 {{requires shape_type to be ranked}}
  %axis = axis.getaxis tensor<*xf32> 0
  return
}

// -----

func.func @getaxis_requires_in_range_axis() {
  // expected-error @+1 {{requires axis_index in [0, rank), got 2 for rank 2}}
  %axis = axis.getaxis tensor<8x4xf32> 2
  return
}

// -----

func.func @getaxis_requires_static_dim() {
  // expected-error @+1 {{requires static shape dimension at axis_index 0}}
  %axis = axis.getaxis tensor<?x4xf32> 0
  return
}

// -----

func.func @factor_requires_axis_op_result(%arg0: !axis.shape_axis<tensor<6xf32>, 0>) {
  // expected-error @+1 {{requires axis operand to be traceable to an op result}}
  %f0, %f1 = axis.factor %arg0 [2, 3] : !axis.shape_axis<tensor<6xf32>, 0>
  return
}

// -----

func.func @factor_requires_positive_extents() {
  %axis = axis.getaxis tensor<6xf32> 0
  // expected-error @+1 {{requires all factor extents to be > 0}}
  %f0, %f1 = axis.factor %axis [0, 6] : !axis.shape_axis<tensor<6xf32>, 0>
  return
}

// -----

func.func @factor_requires_extent_product_match() {
  %axis = axis.getaxis tensor<6xf32> 0
  // expected-error @+1 {{requires product(factor_extents) == axis extent (4 != 6)}}
  %f0, %f1 = axis.factor %axis [2, 2] : !axis.shape_axis<tensor<6xf32>, 0>
  return
}

// -----

func.func @factor_requires_stride_convention() {
  %axis = axis.getaxis tensor<6xf32> 0
  // expected-error @+1 {{requires result #0 stride to follow leftmost-major convention}}
  %f0, %f1 = "axis.factor"(%axis) {factor_extents = array<i32: 2, 3>} : (!axis.shape_axis<tensor<6xf32>, 0>) -> (!axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 2, 1>, !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 3, 1>)
  return
}

// -----

func.func @group_requires_factor_op_results(%arg0: !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 2, 3>) {
  // expected-error @+1 {{requires factor operands to be op results}}
  %g = axis.group %arg0 : !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 2, 3>
  return
}

// -----

func.func @group_requires_axis_factor_producer() {
  %axis = axis.getaxis tensor<6xf32> 0
  %f0, %f1 = axis.factor %axis [2, 3] : !axis.shape_axis<tensor<6xf32>, 0>
  %fake = builtin.unrealized_conversion_cast %f0 : !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 2, 3> to !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 2, 3>
  // expected-error @+1 {{requires factor operands to be produced by axis.factor}}
  %g = axis.group %fake, %f1 : !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 2, 3>, !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 3, 1>
  return
}

// -----

func.func @group_requires_extent_product_match() {
  %axis = axis.getaxis tensor<6xf32> 0
  %f0, %f1 = axis.factor %axis [2, 3] : !axis.shape_axis<tensor<6xf32>, 0>
  // expected-error @+1 {{requires group extent to equal product of factor extents}}
  %g = "axis.group"(%f0, %f1) : (!axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 2, 3>, !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 3, 1>) -> !axis.factor_group<5>
  return
}
