// RUN: enzymexlamlir-opt --split-input-file --verify-diagnostics %s

%axis = axis.getaxis tensor<8x4xf32> 1

// -----

// expected-error @+1 {{requires shape_type to be a shaped type}}
%axis0 = axis.getaxis i32 0

// -----

// expected-error @+1 {{requires shape_type to be ranked}}
%axis1 = axis.getaxis tensor<*xf32> 0

// -----

// expected-error @+1 {{requires axis_index in [0, rank), got 2 for rank 2}}
%axis2 = axis.getaxis tensor<8x4xf32> 2

// -----

// expected-error @+1 {{requires static shape dimension at axis_index 0}}
%axis3 = axis.getaxis tensor<?x4xf32> 0

// -----

// A value fed to axis.factor from a block argument is neither traceable to
// an op result nor declared at module scope; the module-scope check fires
// first.
func.func @factor_requires_axis_op_result(%arg0: !axis.shape_axis<tensor<6xf32>, 0>) {
  // expected-error @+1 {{is axis-algebra metadata and must be declared directly in a module body}}
  %f0 = axis.factor %arg0 : !axis.shape_axis<tensor<6xf32>, 0> <2, 3>
  return
}

// -----

%axis4 = axis.getaxis tensor<6xf32> 0
// expected-error @+1 {{requires extent to be positive, got 0}}
%f0 = axis.factor %axis4 : !axis.shape_axis<tensor<6xf32>, 0> <0, 1>

// -----

%axis5 = axis.getaxis tensor<6xf32> 0
// expected-error @+1 {{requires stride to be positive, got 0}}
%f0b = axis.factor %axis5 : !axis.shape_axis<tensor<6xf32>, 0> <2, 0>

// -----

%axis6 = axis.getaxis tensor<6xf32> 0
// expected-error @+1 {{requires factor to divide source axis}}
%f0c = axis.factor %axis6 : !axis.shape_axis<tensor<6xf32>, 0> <2, 4>

// -----

// Same reasoning as factor_requires_axis_op_result above: a block-argument
// operand trips the module-scope check before the op-specific one.
func.func @product_requires_factor_op_results(%arg0: !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 2, 3>) {
  // expected-error @+1 {{is axis-algebra metadata and must be declared directly in a module body}}
  %g = axis.product (%arg0 : !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 2, 3>)
  return
}

// -----

%axis7 = axis.getaxis tensor<6xf32> 0
%f0d = axis.factor %axis7 : !axis.shape_axis<tensor<6xf32>, 0> <2, 3>
%f1 = axis.factor %axis7 : !axis.shape_axis<tensor<6xf32>, 0> <3, 1>
%fake = builtin.unrealized_conversion_cast %f0d : !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 2, 3> to !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 2, 3>
// expected-error @+1 {{requires factor operands to be produced by axis.factor}}
%g0 = axis.product (%fake : !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 2, 3>, %f1 : !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 3, 1>)

// -----

%axis8 = axis.getaxis tensor<6xf32> 0
%f0e = axis.factor %axis8 : !axis.shape_axis<tensor<6xf32>, 0> <2, 3>
%f1b = axis.factor %axis8 : !axis.shape_axis<tensor<6xf32>, 0> <3, 1>
// expected-error @+1 {{requires product extent to equal product of factor extents}}
%g1 = "axis.product"(%f0e, %f1b) : (!axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 2, 3>, !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 3, 1>) -> !axis.factor_group<5>

// -----

%axis9 = axis.getaxis tensor<6xf32> 0
// expected-error @+1 {{requires all segment extents to be > 0}}
%s0, %s1 = axis.segment %axis9 [0, 6] : !axis.shape_axis<tensor<6xf32>, 0>

// -----

%axis10 = axis.getaxis tensor<6xf32> 0
// expected-error @+1 {{requires sum(segment_extents) == axis extent (4 != 6)}}
%s0b, %s1b = axis.segment %axis10 [2, 2] : !axis.shape_axis<tensor<6xf32>, 0>

// -----

%axis11 = axis.getaxis tensor<6xf32> 0
// expected-error @+1 {{requires result #1 offset to match cumulative segment layout (low result index maps to low axis values)}}
%s0c, %s1c = "axis.segment"(%axis11) {segment_extents = array<i64: 2, 4>} : (!axis.shape_axis<tensor<6xf32>, 0>) -> (!axis.axis_segment<!axis.shape_axis<tensor<6xf32>, 0>, 2, 0>, !axis.axis_segment<!axis.shape_axis<tensor<6xf32>, 0>, 4, 1>)

// -----

%axis12 = axis.getaxis tensor<6xf32> 0
// expected-error @+1 {{requires result #0 offset to match cumulative segment layout (low result index maps to low axis values)}}
%s0d, %s1d = "axis.segment"(%axis12) {segment_extents = array<i64: 2, 4>} : (!axis.shape_axis<tensor<6xf32>, 0>) -> (!axis.axis_segment<!axis.shape_axis<tensor<6xf32>, 0>, 2, 1>, !axis.axis_segment<!axis.shape_axis<tensor<6xf32>, 0>, 4, 3>)
