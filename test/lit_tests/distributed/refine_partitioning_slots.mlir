// RUN: enzymexlamlir-opt --refine-partitioning-slots %s | FileCheck %s

// A slot mixing a sharded and a local factor is replaced by one slot per run,
// and every index list (kernel boundary and body-op attributes) is remapped to
// the run of new slots. Slots that are purely local or empty are kept as is.
module @mixed_slot {
  func.func @main() {
    return
  }
  %logical = distributed.LogicalMeshAxes 2 : !distributed.logical_mesh_axis<2>
  %devloc = distributed.DeviceLocalAxis 3 : !distributed.device_local_axis<3>
  %lf = axis.factor %logical : !distributed.logical_mesh_axis<2><2, 1>
  %df = axis.factor %devloc : !distributed.device_local_axis<3><3, 1>
  %mixed = axis.product (%df : !axis.axis_factor<!distributed.device_local_axis<3>, 3, 1>, %lf : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>)
  %empty = axis.product ()
  %input = stablehlo.constant dense<1.0> : tensor<1x1xf32>
  %r = distributed.DistributedKernel (%input : tensor<1x1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0], [1]] : unreduced_axes = []>]>
    -> (tensor<1x1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[1], [0]] : unreduced_axes = []>]>
    axes (%mixed : !axis.factor_group<6>, %empty : !axis.factor_group<1>) {
  ^bb0(%arg0: tensor<6x1xf32>):
    %t = stablehlo.transpose %arg0, dims = [1, 0] {distributed.argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0], [1]] : unreduced_axes = []>]>, distributed.output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[1], [0]] : unreduced_axes = []>]>} : (tensor<6x1xf32>) -> tensor<1x6xf32>
    distributed.DistributedYield (%t : tensor<1x6xf32>)
  }
}

// CHECK-LABEL: module @mixed_slot {
// CHECK-DAG: %[[LOCAL:.*]] = axis.product (%{{.*}} : !axis.axis_factor<!distributed.device_local_axis<3>, 3, 1>)
// CHECK-DAG: %[[SHARD:.*]] = axis.product (%{{.*}} : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>)
// CHECK-DAG: %[[EMPTY:.*]] = axis.product ()
// CHECK: distributed.DistributedKernel (%{{.*}} : tensor<1x1xf32>) <[<dim_partitioning_axes = {{\[\[0, 1\], \[2\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: -> (tensor<1x1xf32>) <[<dim_partitioning_axes = {{\[\[2\], \[0, 1\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: axes (%[[LOCAL]] : !axis.factor_group<3>, %[[SHARD]] : !axis.factor_group<2>, %[[EMPTY]] : !axis.factor_group<1>) {
// CHECK: stablehlo.transpose {{.*}}dim_partitioning_axes = {{\[\[0, 1\], \[2\]\]}}
