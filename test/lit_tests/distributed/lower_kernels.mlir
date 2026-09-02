// RUN: enzymexlamlir-opt --distributed-lower-kernels -split-input-file %s | FileCheck %s --check-prefix=NOLOWER
// RUN: enzymexlamlir-opt --distributed-lower-kernels="lower-logical-axes=true" -split-input-file %s | FileCheck %s --check-prefix=LOWER

// Idempotency: applying the pass twice must be a no-op the second time,
// for both the default and lower-logical-axes=true configurations.
// RUN: enzymexlamlir-opt --distributed-lower-kernels -split-input-file %s -o %t.default.once
// RUN: enzymexlamlir-opt --distributed-lower-kernels -split-input-file %t.default.once -o %t.default.twice
// RUN: diff -u %t.default.once %t.default.twice
// RUN: enzymexlamlir-opt --distributed-lower-kernels="lower-logical-axes=true" -split-input-file %s -o %t.lower.once
// RUN: enzymexlamlir-opt --distributed-lower-kernels="lower-logical-axes=true" -split-input-file %t.lower.once -o %t.lower.twice
// RUN: diff -u %t.lower.once %t.lower.twice

// A kernel partitioned over a physical axis. Physical axes are never
// shardable through this pass (regardless of lower-logical-axes), so nothing
// about the kernel should change: not the external operand/result types, not
// the internal block-arg/yield types, and not the per-dim axis product.
module @physical_kernel {
  func.func @main() {
    return
  }
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 1>]
  %phys = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 1>
  %pf = axis.factor %phys : !distributed.physical_comm_axis<4, 1><4, 1>
  %pg = axis.product (%pf : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %input = tensor.empty() : tensor<4xf32>
  %r = distributed.DistributedKernel %input : tensor<4xf32> #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>
    -> tensor<4xf32> #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>
    axes %pg : !axis.factor_group<4> {
  ^bb0(%arg0: tensor<4xf32>):
    distributed.DistributedYield %arg0 tensor<4xf32>
  }
}

// NOLOWER-LABEL: module @physical_kernel {
// NOLOWER: %[[PG:.*]] = axis.product (%{{.*}} : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
// NOLOWER: distributed.DistributedKernel %{{.*}} : tensor<4xf32> {{.*}}
// NOLOWER-NEXT: -> tensor<4xf32> {{.*}}
// NOLOWER-NEXT: axes %[[PG]] : !axis.factor_group<4> {
// NOLOWER-NEXT: ^bb0(%arg0: tensor<4xf32>):
// NOLOWER-NEXT: distributed.DistributedYield %arg0 tensor<4xf32>

// LOWER-LABEL: module @physical_kernel {
// LOWER: %[[PG:.*]] = axis.product (%{{.*}} : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
// LOWER: distributed.DistributedKernel %{{.*}} : tensor<4xf32> {{.*}}
// LOWER-NEXT: -> tensor<4xf32> {{.*}}
// LOWER-NEXT: axes %[[PG]] : !axis.factor_group<4> {
// LOWER-NEXT: ^bb0(%arg0: tensor<4xf32>):
// LOWER-NEXT: distributed.DistributedYield %arg0 tensor<4xf32>

// -----

// A kernel partitioned over a logical mesh axis. Logical axes are only
// shardable when lower-logical-axes=true. Once fully sharded over, the
// single-factor product is replaced by an empty product ("()"), not removed
// entirely (that would desync the sharding attrs' axis indices). The
// external operand/result types never change; only the internal block-arg
// and yield types (and the factor_group extent) step closer to local.
module @logical_kernel {
  func.func @main() {
    return
  }
  %logical = distributed.LogicalMeshAxes [4] : !distributed.logical_mesh_axis<4>
  %lf = axis.factor %logical : !distributed.logical_mesh_axis<4><4, 1>
  %lg = axis.product (%lf : !axis.axis_factor<!distributed.logical_mesh_axis<4>, 4, 1>)
  %input = tensor.empty() : tensor<4xf32>
  %r = distributed.DistributedKernel %input : tensor<4xf32> #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>
    -> tensor<4xf32> #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>
    axes %lg : !axis.factor_group<4> {
  ^bb0(%arg0: tensor<4xf32>):
    distributed.DistributedYield %arg0 tensor<4xf32>
  }
}

// Unlowered: nothing changes.
// NOLOWER-LABEL: module @logical_kernel {
// NOLOWER: %[[LG:.*]] = axis.product (%{{.*}} : !axis.axis_factor<!distributed.logical_mesh_axis<4>, 4, 1>)
// NOLOWER: distributed.DistributedKernel %{{.*}} : tensor<4xf32> {{.*}}
// NOLOWER-NEXT: -> tensor<4xf32> {{.*}}
// NOLOWER-NEXT: axes %[[LG]] : !axis.factor_group<4> {
// NOLOWER-NEXT: ^bb0(%arg0: tensor<4xf32>):
// NOLOWER-NEXT: distributed.DistributedYield %arg0 tensor<4xf32>

// Lowered: external types unchanged; the factor is removed from the product
// (which becomes empty rather than disappearing), and the internal
// block-arg/yield types and factor_group extent shrink from 4 to 1.
// LOWER-LABEL: module @logical_kernel {
// LOWER: %[[LG:.*]] = axis.product ()
// LOWER: distributed.DistributedKernel %{{.*}} : tensor<4xf32> {{.*}}
// LOWER-NEXT: -> tensor<4xf32> {{.*}}
// LOWER-NEXT: axes %[[LG]] : !axis.factor_group<1> {
// LOWER-NEXT: ^bb0(%arg0: tensor<1xf32>):
// LOWER-NEXT: distributed.DistributedYield %arg0 tensor<1xf32>

// -----

// A kernel partitioned over a replication axis. Replication axes are never
// shardable through this pass, so nothing changes regardless of the flag.
module @replication_kernel {
  func.func @main() {
    return
  }
  %repl = distributed.ReplicationAxis 4 : !distributed.replication_axis<4>
  %rf = axis.factor %repl : !distributed.replication_axis<4><4, 1>
  %rg = axis.product (%rf : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
  %input = tensor.empty() : tensor<4xf32>
  %r = distributed.DistributedKernel %input : tensor<4xf32> #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>
    -> tensor<4xf32> #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>
    axes %rg : !axis.factor_group<4> {
  ^bb0(%arg0: tensor<4xf32>):
    distributed.DistributedYield %arg0 tensor<4xf32>
  }
}

// NOLOWER-LABEL: module @replication_kernel {
// NOLOWER: %[[RG:.*]] = axis.product (%{{.*}} : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
// NOLOWER: axes %[[RG]] : !axis.factor_group<4> {
// NOLOWER-NEXT: ^bb0(%arg0: tensor<4xf32>):

// LOWER-LABEL: module @replication_kernel {
// LOWER: %[[RG:.*]] = axis.product (%{{.*}} : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
// LOWER: axes %[[RG]] : !axis.factor_group<4> {
// LOWER-NEXT: ^bb0(%arg0: tensor<4xf32>):

// -----

// A kernel partitioned over a device-local axis. Device-local axes are never
// shardable through this pass (they are always left for a later pass to
// tile into larger local blocks), so nothing changes regardless of the flag.
module @devicelocal_kernel {
  func.func @main() {
    return
  }
  %devloc = distributed.DeviceLocalAxis 4 : !distributed.device_local_axis<4>
  %df = axis.factor %devloc : !distributed.device_local_axis<4><4, 1>
  %dg = axis.product (%df : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>)
  %input = tensor.empty() : tensor<4xf32>
  %r = distributed.DistributedKernel %input : tensor<4xf32> #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>
    -> tensor<4xf32> #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>
    axes %dg : !axis.factor_group<4> {
  ^bb0(%arg0: tensor<4xf32>):
    distributed.DistributedYield %arg0 tensor<4xf32>
  }
}

// NOLOWER-LABEL: module @devicelocal_kernel {
// NOLOWER: %[[DG:.*]] = axis.product (%{{.*}} : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>)
// NOLOWER: axes %[[DG]] : !axis.factor_group<4> {
// NOLOWER-NEXT: ^bb0(%arg0: tensor<4xf32>):

// LOWER-LABEL: module @devicelocal_kernel {
// LOWER: %[[DG:.*]] = axis.product (%{{.*}} : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>)
// LOWER: axes %[[DG]] : !axis.factor_group<4> {
// LOWER-NEXT: ^bb0(%arg0: tensor<4xf32>):

// -----

// A single sharding axis composed of two factors: a logical factor (extent
// 2) and a device-local factor (extent 2). With lower-logical-axes=true,
// only the logical factor is removed from the product -- the device-local
// factor stays -- and the factor_group/internal types shrink by exactly the
// logical factor's extent (4 -> 2), not all the way to 1.
module @composite_kernel {
  func.func @main() {
    return
  }
  %logical = distributed.LogicalMeshAxes [2] : !distributed.logical_mesh_axis<2>
  %devloc = distributed.DeviceLocalAxis 2 : !distributed.device_local_axis<2>
  %lf = axis.factor %logical : !distributed.logical_mesh_axis<2><2, 1>
  %df = axis.factor %devloc : !distributed.device_local_axis<2><2, 1>
  %cg = axis.product (%lf : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>, %df : !axis.axis_factor<!distributed.device_local_axis<2>, 2, 1>)
  %input = tensor.empty() : tensor<4xf32>
  %r = distributed.DistributedKernel %input : tensor<4xf32> #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>
    -> tensor<4xf32> #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>
    axes %cg : !axis.factor_group<4> {
  ^bb0(%arg0: tensor<4xf32>):
    distributed.DistributedYield %arg0 tensor<4xf32>
  }
}

// NOLOWER-LABEL: module @composite_kernel {
// NOLOWER: %[[CG:.*]] = axis.product (%{{.*}} : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>, %{{.*}} : !axis.axis_factor<!distributed.device_local_axis<2>, 2, 1>)
// NOLOWER: axes %[[CG]] : !axis.factor_group<4> {
// NOLOWER-NEXT: ^bb0(%arg0: tensor<4xf32>):

// LOWER-LABEL: module @composite_kernel {
// LOWER: %[[DF:.*]] = axis.factor %{{.*}} : !distributed.device_local_axis<2><2, 1>
// LOWER: %[[CG:.*]] = axis.product (%[[DF]] : !axis.axis_factor<!distributed.device_local_axis<2>, 2, 1>)
// LOWER: axes %[[CG]] : !axis.factor_group<2> {
// LOWER-NEXT: ^bb0(%arg0: tensor<2xf32>):
// LOWER-NEXT: distributed.DistributedYield %arg0 tensor<2xf32>
