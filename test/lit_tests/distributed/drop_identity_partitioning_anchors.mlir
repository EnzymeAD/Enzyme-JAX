// RUN: enzymexlamlir-opt %s --drop-identity-partitioning-anchors -split-input-file | FileCheck %s

// AnchorPartitioning's input/output types are always identical
// (AllTypesMatch), so every instance is dropped unconditionally regardless
// of what its partitioning_axes claim.
// CHECK-LABEL: module @drops_bare_anchor
// CHECK: %[[INPUT:.*]] = tensor.empty
// CHECK-NOT: distributed.AnchorPartitioning
// CHECK: distributed.DistributedKernel (%[[INPUT]] : tensor<4xf32>)
module @drops_bare_anchor {
  func.func @main() {
    return
  }
  %l = distributed.LogicalMeshAxes 4 : !distributed.logical_mesh_axis<4>
  %lf = axis.factor %l : !distributed.logical_mesh_axis<4><4, 1>
  %g = axis.product (%lf : !axis.axis_factor<!distributed.logical_mesh_axis<4>, 4, 1>)
  %input = tensor.empty() : tensor<4xf32>
  %out = distributed.AnchorPartitioning %input axes (%g : !axis.factor_group<4>) : tensor<4xf32>
  %keep = distributed.DistributedKernel (%out : tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    -> (tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    axes () {
  ^bb0(%arg0: tensor<4xf32>):
    distributed.DistributedYield (%arg0 : tensor<4xf32>)
  }
}

// -----

// A Cast whose input/output types already happen to be equal (a trivial,
// no-op-shape cast) folds away the same as a bare anchor.
// CHECK-LABEL: module @drops_trivial_cast
// CHECK: %[[INPUT:.*]] = tensor.empty
// CHECK-NOT: distributed.CastGlobalToLocal
// CHECK-NOT: distributed.AnchorPartitioning
// CHECK: distributed.DistributedKernel (%[[INPUT]] : tensor<4xf32>)
module @drops_trivial_cast {
  func.func @main() {
    return
  }
  %g = axis.product ()
  %input = tensor.empty() : tensor<4xf32>
  %out = distributed.CastGlobalToLocal %input axes (%g : !axis.factor_group<1>) : tensor<4xf32> -> tensor<4xf32>
  %keep = distributed.DistributedKernel (%out : tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    -> (tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    axes () {
  ^bb0(%arg0: tensor<4xf32>):
    distributed.DistributedYield (%arg0 : tensor<4xf32>)
  }
}

// -----

// A Global->Local->Global round trip over the same axes first collapses (via
// the existing shared FoldCastRoundTrip canonicalizer) into an
// AnchorPartitioning, which this pass then drops in the same run.
// CHECK-LABEL: module @drops_roundtrip_cast
// CHECK: %[[GLOBAL:.*]] = tensor.empty
// CHECK-NOT: distributed.Cast
// CHECK-NOT: distributed.AnchorPartitioning
// CHECK: distributed.DistributedKernel (%[[GLOBAL]] : tensor<4xf32>)
module @drops_roundtrip_cast {
  func.func @main() {
    return
  }
  %l = distributed.LogicalMeshAxes 4 : !distributed.logical_mesh_axis<4>
  %lf = axis.factor %l : !distributed.logical_mesh_axis<4><4, 1>
  %g = axis.product (%lf : !axis.axis_factor<!distributed.logical_mesh_axis<4>, 4, 1>)
  %global = tensor.empty() : tensor<4xf32>
  %local = distributed.CastGlobalToLocal %global axes (%g : !axis.factor_group<4>) : tensor<4xf32> -> tensor<1xf32>
  %back = distributed.CastLocalToGlobal %local axes (%g : !axis.factor_group<4>) : tensor<1xf32> -> tensor<4xf32>
  %keep = distributed.DistributedKernel (%back : tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    -> (tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    axes () {
  ^bb0(%arg0: tensor<4xf32>):
    distributed.DistributedYield (%arg0 : tensor<4xf32>)
  }
}
