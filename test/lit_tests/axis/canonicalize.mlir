// RUN: enzymexlamlir-opt --canonicalize %s | FileCheck %s

// General home for testing --canonicalize behavior on the axis dialect's
// own ops (as opposed to a specific pass like canonicalize-axis-maps.mlir).

// Extent-1 factors are multiplicatively neutral -- they never divide
// anything and never change which real index space a product covers (see
// areFactorIndexSpacesEqual, which ignores them for the same reason) -- so
// axis.product drops them unconditionally. This is what lets a
// distributed.Collective mapping combining a real mesh factor with an
// incidental extent-1 tensor-shape factor (as
// MaterializeDistributedCollectives.cpp's toLocallyTypedAxisProduct always
// emits, one local-axis factor per tensor dimension regardless of extent)
// reach axis::split_divisible with only real content left.
// CHECK-LABEL: func.func @product_drops_unit_factor
// CHECK: %[[REAL:.*]] = axis.factor %{{.*}} : !axis.shape_axis<tensor<2xf32>, 0><2, 1>
// CHECK: axis.product (%[[REAL]] : !axis.axis_factor<!axis.shape_axis<tensor<2xf32>, 0>, 2, 1>)
// CHECK-NOT: shape_axis<tensor<1xf32>
func.func @product_drops_unit_factor() -> !axis.factor_group<2> {
  %axis = axis.getaxis tensor<2xf32> 0
  %unit_axis = axis.getaxis tensor<1xf32> 0
  %real = axis.factor %axis : !axis.shape_axis<tensor<2xf32>, 0> <2, 1>
  %unit = axis.factor %unit_axis : !axis.shape_axis<tensor<1xf32>, 0> <1, 1>
  %p = axis.product (%real : !axis.axis_factor<!axis.shape_axis<tensor<2xf32>, 0>, 2, 1>, %unit : !axis.axis_factor<!axis.shape_axis<tensor<1xf32>, 0>, 1, 1>)
  return %p : !axis.factor_group<2>
}

// A product made up entirely of extent-1 factors becomes the empty product
// -- AxisProductOp's own documented "denotes a fully-sharded axis (extent
// 1)" empty-list case.
// CHECK-LABEL: func.func @product_all_unit_factors_becomes_empty
// CHECK: axis.product ()
func.func @product_all_unit_factors_becomes_empty() -> !axis.factor_group<1> {
  %axis = axis.getaxis tensor<1xf32> 0
  %unit = axis.factor %axis : !axis.shape_axis<tensor<1xf32>, 0> <1, 1>
  %p = axis.product (%unit : !axis.axis_factor<!axis.shape_axis<tensor<1xf32>, 0>, 1, 1>)
  return %p : !axis.factor_group<1>
}
