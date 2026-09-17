// RUN: enzymexlamlir-opt --canonicalize %s | FileCheck %s

// General home for testing --canonicalize behavior on the axis dialect's
// own ops (as opposed to a specific pass like canonicalize-axis-maps.mlir).
// Each factor group below is used as a distributed.DistributedKernel's own
// partitioning axis, both to give --canonicalize a real (non-Pure) consumer
// to keep it from folding the whole chain away, and because that's exactly
// how these values get used in practice.

// Extent-1 factors are multiplicatively neutral -- they never divide
// anything and never change which real index space a product covers (see
// areFactorIndexSpacesEqual, which ignores them for the same reason) -- so
// axis.product drops them unconditionally. This is what lets a
// distributed.Collective mapping combining a real mesh factor with an
// incidental extent-1 tensor-shape factor (as
// MaterializeDistributedCollectives.cpp's toLocallyTypedAxisProduct always
// emits, one local-axis factor per tensor dimension regardless of extent)
// reach axis::split_divisible with only real content left.
// CHECK-LABEL: module @product_drops_unit_factor
// CHECK: %[[REAL:.*]] = axis.factor %{{.*}} : !axis.shape_axis<tensor<2xf32>, 0><2, 1>
// CHECK: %[[PRODUCT:.*]] = axis.product (%[[REAL]] : !axis.axis_factor<!axis.shape_axis<tensor<2xf32>, 0>, 2, 1>)
// CHECK-NOT: shape_axis<tensor<1xf32>
// CHECK: axes (%[[PRODUCT]] : !axis.factor_group<2>)
module @product_drops_unit_factor {
  func.func @main() {
    return
  }
  %axis = axis.getaxis tensor<2xf32> 0
  %unit_axis = axis.getaxis tensor<1xf32> 0
  %real = axis.factor %axis : !axis.shape_axis<tensor<2xf32>, 0> <2, 1>
  %unit = axis.factor %unit_axis : !axis.shape_axis<tensor<1xf32>, 0> <1, 1>
  %p = axis.product (%real : !axis.axis_factor<!axis.shape_axis<tensor<2xf32>, 0>, 2, 1>, %unit : !axis.axis_factor<!axis.shape_axis<tensor<1xf32>, 0>, 1, 1>)
  %input = tensor.empty() : tensor<1xf32>
  %keep = distributed.DistributedKernel (%input : tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>
    -> (tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    axes (%p : !axis.factor_group<2>) {
  ^bb0(%arg0: tensor<2xf32>):
    %r = stablehlo.constant dense<0.0> : tensor<1xf32>
    distributed.DistributedYield (%r : tensor<1xf32>)
  }
}

// -----

// A product made up entirely of extent-1 factors becomes the empty product
// -- AxisProductOp's own documented "denotes a fully-sharded axis (extent
// 1)" empty-list case.
// CHECK-LABEL: module @product_all_unit_factors_becomes_empty
// CHECK: %[[PRODUCT:.*]] = axis.product ()
// CHECK: axes (%[[PRODUCT]] : !axis.factor_group<1>)
module @product_all_unit_factors_becomes_empty {
  func.func @main() {
    return
  }
  %axis = axis.getaxis tensor<1xf32> 0
  %unit = axis.factor %axis : !axis.shape_axis<tensor<1xf32>, 0> <1, 1>
  %p = axis.product (%unit : !axis.axis_factor<!axis.shape_axis<tensor<1xf32>, 0>, 1, 1>)
  %input = tensor.empty() : tensor<1xf32>
  %keep = distributed.DistributedKernel (%input : tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>
    -> (tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    axes (%p : !axis.factor_group<1>) {
  ^bb0(%arg0: tensor<1xf32>):
    distributed.DistributedYield (%arg0 : tensor<1xf32>)
  }
}
