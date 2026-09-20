// RUN: enzymexlamlir-opt --refine-partitioning-slots --canonicalize-sharded-factor-order --cse -split-input-file %s | FileCheck %s

// A kernel whose sole dimension is already sharded-major/local-minor within
// one composite axis.product slot (the same shape as lower_kernels.mlir's
// composite_kernel). refine-partitioning-slots splits the slot into a sharded
// slot and a local slot, listed in that order; canonicalization then has
// nothing to reorder.
module @already_canonical {
  func.func @main() {
    return
  }
  %logical = distributed.LogicalMeshAxes 2 : !distributed.logical_mesh_axis<2>
  %devloc = distributed.DeviceLocalAxis 2 : !distributed.device_local_axis<2>
  %lf = axis.factor %logical : !distributed.logical_mesh_axis<2><2, 1>
  %df = axis.factor %devloc : !distributed.device_local_axis<2><2, 1>
  %cg = axis.product (%lf : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>, %df : !axis.axis_factor<!distributed.device_local_axis<2>, 2, 1>)
  %input = stablehlo.constant dense<1.0> : tensor<1xf32>
  %r = distributed.DistributedKernel (%input : tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>
    -> (tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>
    axes (%cg : !axis.factor_group<4>) {
  ^bb0(%arg0: tensor<4xf32>):
    distributed.DistributedYield (%arg0 : tensor<4xf32>)
  }
}

// CHECK-LABEL: module @already_canonical {
// CHECK-DAG: %[[SHARD:.*]] = axis.product (%{{.*}} : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>)
// CHECK-DAG: %[[LOCAL:.*]] = axis.product (%{{.*}} : !axis.axis_factor<!distributed.device_local_axis<2>, 2, 1>)
// CHECK: distributed.DistributedKernel (%{{.*}} : tensor<1xf32>) <[<dim_partitioning_axes = {{\[\[0, 1\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: -> (tensor<1xf32>) <[<dim_partitioning_axes = {{\[\[0, 1\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: axes (%[[SHARD]] : !axis.factor_group<2>, %[[LOCAL]] : !axis.factor_group<2>) {
// CHECK-NEXT: ^bb0(%arg0: tensor<4xf32>):
// CHECK-NEXT: distributed.DistributedYield (%arg0 : tensor<4xf32>)

// -----

// Same shape, but the one slot's own factor order is sandwiched the other
// way (local major, shard minor). Part L case (1): this is a
// DistributedKernelOp's own boundary, so it's a PURE METADATA fix. The slot is
// refined into a local slot and a sharded slot, and the dimension's slot list
// is permuted to [sharded, local]; no slot is created by canonicalization.
// The kernel body (block argument, yield) is untouched.
module @sandwiched_within_slot {
  func.func @main() {
    return
  }
  %logical = distributed.LogicalMeshAxes 2 : !distributed.logical_mesh_axis<2>
  %devloc = distributed.DeviceLocalAxis 2 : !distributed.device_local_axis<2>
  %lf = axis.factor %logical : !distributed.logical_mesh_axis<2><2, 1>
  %df = axis.factor %devloc : !distributed.device_local_axis<2><2, 1>
  %cg = axis.product (%df : !axis.axis_factor<!distributed.device_local_axis<2>, 2, 1>, %lf : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>)
  %input = stablehlo.constant dense<1.0> : tensor<1xf32>
  %r = distributed.DistributedKernel (%input : tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>
    -> (tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>
    axes (%cg : !axis.factor_group<4>) {
  ^bb0(%arg0: tensor<4xf32>):
    distributed.DistributedYield (%arg0 : tensor<4xf32>)
  }
}

// CHECK-LABEL: module @sandwiched_within_slot {
// CHECK-DAG: %[[LOCAL:.*]] = axis.product (%{{.*}} : !axis.axis_factor<!distributed.device_local_axis<2>, 2, 1>)
// CHECK-DAG: %[[SHARD:.*]] = axis.product (%{{.*}} : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>)
// CHECK: distributed.DistributedKernel (%{{.*}} : tensor<1xf32>) <[<dim_partitioning_axes = {{\[\[1, 0\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: -> (tensor<1xf32>) <[<dim_partitioning_axes = {{\[\[1, 0\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: axes (%[[LOCAL]] : !axis.factor_group<2>, %[[SHARD]] : !axis.factor_group<2>) {
// CHECK-NEXT: ^bb0(%arg0: tensor<4xf32>):
// CHECK-NEXT: distributed.DistributedYield (%arg0 : tensor<4xf32>)

// -----

// A 32-element dimension spread across three *separate* single-factor
// slots, DeviceLocal(4) -> LogicalMesh(2) -> DeviceLocal(4) (Local x Shard
// x Local, sandwiched across slots rather than within one) -- exactly the
// pattern InlineDeviceLocalAxes.cpp's doc comment describes as unsupported
// downstream, and the motivating case for this whole pass. Part L case (1)
// again: pure metadata -- dim_partitioning_axes is repointed to [1, 0, 2]
// (the existing shard slot, now listed first), reusing the three EXISTING
// slots (no new ones needed, since each was already single-factor), with
// zero change to the kernel body.
module @sandwiched_across_slots {
  func.func @main() {
    return
  }
  %a1 = distributed.DeviceLocalAxis 4 : !distributed.device_local_axis<4>
  %a2 = distributed.LogicalMeshAxes 2 : !distributed.logical_mesh_axis<2>
  %a3 = distributed.DeviceLocalAxis 4 : !distributed.device_local_axis<4>
  %f1 = axis.factor %a1 : !distributed.device_local_axis<4><4, 1>
  %f2 = axis.factor %a2 : !distributed.logical_mesh_axis<2><2, 1>
  %f3 = axis.factor %a3 : !distributed.device_local_axis<4><4, 1>
  %g1 = axis.product (%f1 : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>)
  %g2 = axis.product (%f2 : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>)
  %g3 = axis.product (%f3 : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>)
  %input = stablehlo.constant dense<0.0> : tensor<1xf32>
  %r = distributed.DistributedKernel (%input : tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0, 1, 2]] : unreduced_axes = []>]>
    -> (tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0, 1, 2]] : unreduced_axes = []>]>
    axes (%g1 : !axis.factor_group<4>, %g2 : !axis.factor_group<2>, %g3 : !axis.factor_group<4>) {
  ^bb0(%arg0: tensor<32xf32>):
    distributed.DistributedYield (%arg0 : tensor<32xf32>)
  }
}

// CHECK-LABEL: module @sandwiched_across_slots {
// CHECK: distributed.DistributedKernel (%{{.*}} : tensor<1xf32>) <[<dim_partitioning_axes = {{\[\[1, 0, 2\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: -> (tensor<1xf32>) <[<dim_partitioning_axes = {{\[\[1, 0, 2\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: axes (%{{.*}} : !axis.factor_group<4>, %{{.*}} : !axis.factor_group<2>, %{{.*}} : !axis.factor_group<4>) {
// CHECK-NEXT: ^bb0(%arg0: tensor<32xf32>):
// CHECK-NEXT: distributed.DistributedYield (%arg0 : tensor<32xf32>)

// -----

// DistributedCastGlobalToLocalOp with a sandwiched (local major, shard
// minor) factor group on its sole dimension. Part L case (2): the cast's
// own `axes` operand is repointed at a fresh, canonically-ordered
// axis.product (declaring G/L order in place of g/l) -- pure axis-algebra,
// no stablehlo op inserted anywhere, no change to the cast's result or any
// of its uses. (The bookending reconciling collective this design also
// calls for is not yet implemented -- see the pass's own top-level comment.)
module @cast_global_to_local_sandwiched {
  func.func @main() {
    return
  }
  %logical = distributed.LogicalMeshAxes 2 : !distributed.logical_mesh_axis<2>
  %devloc = distributed.DeviceLocalAxis 2 : !distributed.device_local_axis<2>
  %lf = axis.factor %logical : !distributed.logical_mesh_axis<2><2, 1>
  %df = axis.factor %devloc : !distributed.device_local_axis<2><2, 1>
  %cg = axis.product (%df : !axis.axis_factor<!distributed.device_local_axis<2>, 2, 1>, %lf : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>)
  %input = stablehlo.constant dense<1.0> : tensor<16xf32>
  %out = distributed.CastGlobalToLocal %input axes (%cg : !axis.factor_group<4>) : tensor<16xf32> -> tensor<4xf32>
  %keep = distributed.DistributedKernel (%out : tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    -> (tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    axes () {
  ^bb0(%arg0: tensor<4xf32>):
    distributed.DistributedYield (%arg0 : tensor<4xf32>)
  }
}

// CHECK-LABEL: module @cast_global_to_local_sandwiched {
// CHECK: %[[NEWCG:.*]] = axis.product (%{{.*}} : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>, %{{.*}} : !axis.axis_factor<!distributed.device_local_axis<2>, 2, 1>)
// CHECK: %[[CST:.*]] = stablehlo.constant
// CHECK-NEXT: %[[OUT:.*]] = distributed.CastGlobalToLocal %[[CST]] axes (%[[NEWCG]] : !axis.factor_group<4>) : tensor<16xf32> -> tensor<4xf32>
// CHECK-NEXT: distributed.DistributedKernel (%[[OUT]] : tensor<4xf32>)

// -----

// DistributedCastLocalToGlobalOp, same sandwiched shape -- same pure
// metadata fix, symmetric direction.
module @cast_local_to_global_sandwiched {
  func.func @main() {
    return
  }
  %logical = distributed.LogicalMeshAxes 2 : !distributed.logical_mesh_axis<2>
  %devloc = distributed.DeviceLocalAxis 2 : !distributed.device_local_axis<2>
  %lf = axis.factor %logical : !distributed.logical_mesh_axis<2><2, 1>
  %df = axis.factor %devloc : !distributed.device_local_axis<2><2, 1>
  %cg = axis.product (%df : !axis.axis_factor<!distributed.device_local_axis<2>, 2, 1>, %lf : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>)
  %input = stablehlo.constant dense<1.0> : tensor<4xf32>
  %out = distributed.CastLocalToGlobal %input axes (%cg : !axis.factor_group<4>) : tensor<4xf32> -> tensor<16xf32>
  %keep = distributed.DistributedKernel (%out : tensor<16xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    -> (tensor<16xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    axes () {
  ^bb0(%arg0: tensor<16xf32>):
    distributed.DistributedYield (%arg0 : tensor<16xf32>)
  }
}

// CHECK-LABEL: module @cast_local_to_global_sandwiched {
// CHECK: %[[NEWCG:.*]] = axis.product (%{{.*}} : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>, %{{.*}} : !axis.axis_factor<!distributed.device_local_axis<2>, 2, 1>)
// CHECK: %[[CST:.*]] = stablehlo.constant
// CHECK-NEXT: %[[OUT:.*]] = distributed.CastLocalToGlobal %[[CST]] axes (%[[NEWCG]] : !axis.factor_group<4>) : tensor<4xf32> -> tensor<16xf32>
// CHECK-NEXT: distributed.DistributedKernel (%[[OUT]] : tensor<16xf32>)

// -----

// DistributedAnchorPartitioningOp, same sandwiched shape -- same pure
// metadata fix as the two real casts above (case (2) also covers this op:
// see this file's top-level comment). AllTypesMatch means there's no
// direction to get wrong and no type to change either way.
module @anchor_partitioning_sandwiched {
  func.func @main() {
    return
  }
  %logical = distributed.LogicalMeshAxes 2 : !distributed.logical_mesh_axis<2>
  %devloc = distributed.DeviceLocalAxis 2 : !distributed.device_local_axis<2>
  %lf = axis.factor %logical : !distributed.logical_mesh_axis<2><2, 1>
  %df = axis.factor %devloc : !distributed.device_local_axis<2><2, 1>
  %cg = axis.product (%df : !axis.axis_factor<!distributed.device_local_axis<2>, 2, 1>, %lf : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>)
  %input = stablehlo.constant dense<1.0> : tensor<4xf32>
  %out = distributed.AnchorPartitioning %input axes (%cg : !axis.factor_group<4>) : tensor<4xf32>
  %keep = distributed.DistributedKernel (%out : tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    -> (tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    axes () {
  ^bb0(%arg0: tensor<4xf32>):
    distributed.DistributedYield (%arg0 : tensor<4xf32>)
  }
}

// CHECK-LABEL: module @anchor_partitioning_sandwiched {
// CHECK: %[[NEWCG:.*]] = axis.product (%{{.*}} : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>, %{{.*}} : !axis.axis_factor<!distributed.device_local_axis<2>, 2, 1>)
// CHECK: %[[CST:.*]] = stablehlo.constant
// CHECK-NEXT: %[[OUT:.*]] = distributed.AnchorPartitioning %[[CST]] axes (%[[NEWCG]] : !axis.factor_group<4>) : tensor<4xf32>
// CHECK-NEXT: distributed.DistributedKernel (%[[OUT]] : tensor<4xf32>)

// -----

// A reshape strictly inside a kernel body, fed directly by the kernel's own
// block argument, whose own (stale) distributed.argument_shardings still
// declares the pre-canonicalization order [0, 1, 2] -- exactly as
// ClusterDistributedKernels.cpp originally set it. Because the kernel's own
// boundary gets canonicalized to [1, 0, 2] FIRST (pre-order walk, Part L
// case (1)), and this reshape's operand IS that same block argument,
// resolveCurrentSharding re-derives the operand's *current* (canonical)
// sharding from the kernel's own (already-updated) argument_shardings rather
// than trusting the reshape's own stale copy -- and since [1, 0, 2] is
// already canonical, nothing further needs fixing: no split/transpose/merge
// at all, just the reshape's own attrs corrected to match reality. This is
// exactly the shape validated end-to-end against the real
// --distributed-lower-kernels pipeline (see the design doc's Part L/M).
module @kernel_internal_reshape_sandwiched {
  func.func @main() {
    return
  }
  %a1 = distributed.DeviceLocalAxis 4 : !distributed.device_local_axis<4>
  %a2 = distributed.LogicalMeshAxes 2 : !distributed.logical_mesh_axis<2>
  %a3 = distributed.DeviceLocalAxis 4 : !distributed.device_local_axis<4>
  %f1 = axis.factor %a1 : !distributed.device_local_axis<4><4, 1>
  %f2 = axis.factor %a2 : !distributed.logical_mesh_axis<2><2, 1>
  %f3 = axis.factor %a3 : !distributed.device_local_axis<4><4, 1>
  %g1 = axis.product (%f1 : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>)
  %g2 = axis.product (%f2 : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>)
  %g3 = axis.product (%f3 : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>)
  %input = stablehlo.constant dense<0.0> : tensor<1xf32>
  %r = distributed.DistributedKernel (%input : tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0, 1, 2]] : unreduced_axes = []>]>
    -> (tensor<1x1x1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0], [1], [2]] : unreduced_axes = []>]>
    axes (%g1 : !axis.factor_group<4>, %g2 : !axis.factor_group<2>, %g3 : !axis.factor_group<4>) {
  ^bb0(%arg0: tensor<32xf32>):
    %reshaped = stablehlo.reshape %arg0 {distributed.argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0, 1, 2]] : unreduced_axes = []>]>, distributed.output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0], [1], [2]] : unreduced_axes = []>]>} : (tensor<32xf32>) -> tensor<4x2x4xf32>
    distributed.DistributedYield (%reshaped : tensor<4x2x4xf32>)
  }
}

// CHECK-LABEL: module @kernel_internal_reshape_sandwiched {
// CHECK: distributed.DistributedKernel (%{{.*}} : tensor<1xf32>) <[<dim_partitioning_axes = {{\[\[1, 0, 2\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: -> (tensor<1x1x1xf32>) <[<dim_partitioning_axes = {{\[\[0\], \[1\], \[2\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: axes (%{{.*}} : !axis.factor_group<4>, %{{.*}} : !axis.factor_group<2>, %{{.*}} : !axis.factor_group<4>) {
// CHECK-NEXT: ^bb0(%arg0: tensor<32xf32>):
// CHECK-NEXT: %[[RESHAPED:.*]] = stablehlo.reshape %arg0 {distributed.argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = {{\[\[1, 0, 2\]\]}} : unreduced_axes = []>]>, distributed.output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = {{\[\[0\], \[1\], \[2\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: distributed.DistributedYield (%[[RESHAPED]] : tensor<4x2x4xf32>)

// -----

// A merge (join) reshape strictly inside a kernel body: two already
// individually-canonical dims (X = Xshard(2) major/Xlocal(3) minor,
// Y = Yshard(2) major/Ylocal(2) minor) get merged into one dimension whose
// natural order [Xshard,Xlocal,Yshard,Ylocal] has a local factor between two
// sharded ones. The reshape is replaced by one ManualComputation over both
// sharded slots whose region merges the local 3x2 tile into 6, and whose result
// declares the canonical order [Xshard,Yshard,Xlocal,Ylocal].
module @kernel_internal_merge_sandwiched {
  func.func @main() {
    return
  }
  %xs = distributed.LogicalMeshAxes 2 : !distributed.logical_mesh_axis<2>
  %xl = distributed.DeviceLocalAxis 3 : !distributed.device_local_axis<3>
  %ys = distributed.LogicalMeshAxes 2 : !distributed.logical_mesh_axis<2>
  %yl = distributed.DeviceLocalAxis 2 : !distributed.device_local_axis<2>
  %fxs = axis.factor %xs : !distributed.logical_mesh_axis<2><2, 1>
  %fxl = axis.factor %xl : !distributed.device_local_axis<3><3, 1>
  %fys = axis.factor %ys : !distributed.logical_mesh_axis<2><2, 1>
  %fyl = axis.factor %yl : !distributed.device_local_axis<2><2, 1>
  %slotX = axis.product (%fxs : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>, %fxl : !axis.axis_factor<!distributed.device_local_axis<3>, 3, 1>)
  %slotY = axis.product (%fys : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>, %fyl : !axis.axis_factor<!distributed.device_local_axis<2>, 2, 1>)
  %cst = stablehlo.constant dense<0.0> : tensor<1x1xf32>
  %r = distributed.DistributedKernel (%cst : tensor<1x1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0], [1]] : unreduced_axes = []>]>
    -> (tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0, 1]] : unreduced_axes = []>]>
    axes (%slotX : !axis.factor_group<6>, %slotY : !axis.factor_group<4>) {
  ^bb0(%arg0: tensor<6x4xf32>):
    %merged = stablehlo.reshape %arg0 {distributed.argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0], [1]] : unreduced_axes = []>]>, distributed.output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0, 1]] : unreduced_axes = []>]>} : (tensor<6x4xf32>) -> tensor<24xf32>
    distributed.DistributedYield (%merged : tensor<24xf32>)
  }
}

// CHECK-LABEL: module @kernel_internal_merge_sandwiched {
// CHECK: distributed.DistributedKernel (%{{.*}} : tensor<1x1xf32>) <[<dim_partitioning_axes = {{\[\[0, 1\], \[2, 3\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: -> (tensor<1xf32>) <[<dim_partitioning_axes = {{\[\[0, 2, 1, 3\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: axes
// CHECK-NEXT: ^bb0(%arg0: tensor<6x4xf32>):
// CHECK-NEXT: %[[MANUAL:.*]] = distributed.ManualComputation (%arg0 : tensor<6x4xf32>) <[<dim_partitioning_axes = {{\[\[0, 1\], \[2, 3\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: manual_axes [0, 2]
// CHECK-NEXT: -> (tensor<24xf32>) <[<dim_partitioning_axes = {{\[\[0, 2, 1, 3\]\]}} : unreduced_axes = []>]> {
// CHECK-NEXT: ^bb0(%[[LARG:.*]]: tensor<3x2xf32>):
// CHECK-NEXT: %[[LMERGE:.*]] = stablehlo.reshape %[[LARG]] {canonicalize_sharded_factor_order.internal} : (tensor<3x2xf32>) -> tensor<6xf32>
// CHECK-NEXT: distributed.DistributedYield (%[[LMERGE]] : tensor<6xf32>)
// CHECK-NEXT: }
// CHECK-NEXT: distributed.DistributedYield (%[[MANUAL]] : tensor<24xf32>)

// -----

// A concatenate whose own concatenated dimension (a Shardy need_replication
// factor) is a single trivial DeviceLocalAxis slot -- never a real
// cross-device split -- while its OTHER (pass-through) dimension is
// sandwiched (local-major, shard-minor) and genuinely needs reordering. The
// two dimensions are independent slot-index lists, so the special factor's
// dimension is left untouched (classified per factor, not for the op as a
// whole -- see isFactorFullyLocal) while the pass-through dimension gets the
// ordinary pure-metadata fix, exactly as it would for any Conforming op: no
// "not yet supported" remark at all.
// The sandwiched slot is refined once into a local and a sharded slot, so
// both operands and the result of the concatenate share those slots.
module @special_factor_but_local {
  func.func @main() {
    return
  }
  %devloc0 = distributed.DeviceLocalAxis 2 : !distributed.device_local_axis<2>
  %logical0 = distributed.LogicalMeshAxes 2 : !distributed.logical_mesh_axis<2>
  %df0 = axis.factor %devloc0 : !distributed.device_local_axis<2><2, 1>
  %lf0 = axis.factor %logical0 : !distributed.logical_mesh_axis<2><2, 1>
  %sandwiched = axis.product (%df0 : !axis.axis_factor<!distributed.device_local_axis<2>, 2, 1>, %lf0 : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>)
  %devloc1 = distributed.DeviceLocalAxis 1 : !distributed.device_local_axis<1>
  %df1 = axis.factor %devloc1 : !distributed.device_local_axis<1><1, 1>
  %local_dim = axis.product (%df1 : !axis.axis_factor<!distributed.device_local_axis<1>, 1, 1>)
  %cst = stablehlo.constant dense<0.0> : tensor<1x1xf32>
  %r = distributed.DistributedKernel (%cst : tensor<1x1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[], []] : unreduced_axes = []>]>
      -> (tensor<1x2xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0], [1]] : unreduced_axes = []>]>
      axes (%sandwiched : !axis.factor_group<4>, %local_dim : !axis.factor_group<1>) {
  ^bb0(%arg0: tensor<4x1xf32>):
    %cat = stablehlo.concatenate %arg0, %arg0, dim = 1 {distributed.argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0], [1]] : unreduced_axes = []>, <dim_partitioning_axes = [[0], [1]] : unreduced_axes = []>]>, distributed.output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0], [1]] : unreduced_axes = []>]>} : (tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<4x2xf32>
    distributed.DistributedYield (%cat : tensor<4x2xf32>)
  }
}

// CHECK-NOT: not yet supported
// CHECK-LABEL: module @special_factor_but_local {
// CHECK-DAG: %[[LOCAL2:.*]] = axis.product (%{{.*}} : !axis.axis_factor<!distributed.device_local_axis<2>, 2, 1>)
// CHECK-DAG: %[[SHARD:.*]] = axis.product (%{{.*}} : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>)
// CHECK-DAG: %[[LOCAL1:.*]] = axis.product (%{{.*}} : !axis.axis_factor<!distributed.device_local_axis<1>, 1, 1>)
// CHECK: distributed.DistributedKernel (%{{.*}} : tensor<1x1xf32>) <[<dim_partitioning_axes = {{\[\[\], \[\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: -> (tensor<1x2xf32>) <[<dim_partitioning_axes = {{\[\[1, 0\], \[2\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: axes (%[[LOCAL2]] : !axis.factor_group<2>, %[[SHARD]] : !axis.factor_group<2>, %[[LOCAL1]] : !axis.factor_group<1>) {
// CHECK-NEXT: ^bb0(%arg0: tensor<4x1xf32>):
// CHECK-NEXT: %[[CAT:.*]] = stablehlo.concatenate %arg0, %arg0, dim = 1 {distributed.argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = {{\[\[1, 0\], \[2\]\]}} : unreduced_axes = []>, <dim_partitioning_axes = {{\[\[1, 0\], \[2\]\]}} : unreduced_axes = []>]>, {{.*}}
// CHECK-NEXT: distributed.DistributedYield (%[[CAT]] : tensor<4x2xf32>)

// -----

// A merge on a tensor with another sharded dimension (the batch slot 0). Every
// sharded slot is manual, so the region reshapes the fully local 1x3x2 tile to
// 1x6.
module @merge_with_sharded_batch_dim {
  func.func @main() {
    return
  }
  %zs = distributed.LogicalMeshAxes 2 : !distributed.logical_mesh_axis<2>
  %xs = distributed.LogicalMeshAxes 2 : !distributed.logical_mesh_axis<2>
  %xl = distributed.DeviceLocalAxis 3 : !distributed.device_local_axis<3>
  %ys = distributed.LogicalMeshAxes 2 : !distributed.logical_mesh_axis<2>
  %yl = distributed.DeviceLocalAxis 2 : !distributed.device_local_axis<2>
  %fzs = axis.factor %zs : !distributed.logical_mesh_axis<2><2, 1>
  %fxs = axis.factor %xs : !distributed.logical_mesh_axis<2><2, 1>
  %fxl = axis.factor %xl : !distributed.device_local_axis<3><3, 1>
  %fys = axis.factor %ys : !distributed.logical_mesh_axis<2><2, 1>
  %fyl = axis.factor %yl : !distributed.device_local_axis<2><2, 1>
  %slotZ = axis.product (%fzs : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>)
  %slotX = axis.product (%fxs : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>, %fxl : !axis.axis_factor<!distributed.device_local_axis<3>, 3, 1>)
  %slotY = axis.product (%fys : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>, %fyl : !axis.axis_factor<!distributed.device_local_axis<2>, 2, 1>)
  %cst = stablehlo.constant dense<0.0> : tensor<1x1x1xf32>
  %r = distributed.DistributedKernel (%cst : tensor<1x1x1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0], [1], [2]] : unreduced_axes = []>]>
    -> (tensor<1x1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0], [1, 2]] : unreduced_axes = []>]>
    axes (%slotZ : !axis.factor_group<2>, %slotX : !axis.factor_group<6>, %slotY : !axis.factor_group<4>) {
  ^bb0(%arg0: tensor<2x6x4xf32>):
    %merged = stablehlo.reshape %arg0 {distributed.argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0], [1], [2]] : unreduced_axes = []>]>, distributed.output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0], [1, 2]] : unreduced_axes = []>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, jk]) {i=2, j=6, k=4}>} : (tensor<2x6x4xf32>) -> tensor<2x24xf32>
    distributed.DistributedYield (%merged : tensor<2x24xf32>)
  }
}

// CHECK-LABEL: module @merge_with_sharded_batch_dim {
// CHECK: distributed.ManualComputation (%arg0 : tensor<2x6x4xf32>)
// CHECK-NEXT: manual_axes [0, 1, 3]
// CHECK-NEXT: -> (tensor<2x24xf32>) <[<dim_partitioning_axes = {{\[\[0\], \[1, 3, 2, 4\]\]}} : unreduced_axes = []>]> {
// CHECK-NEXT: ^bb0(%[[L:.*]]: tensor<1x3x2xf32>):
// CHECK-NEXT: stablehlo.reshape %[[L]] {canonicalize_sharded_factor_order.internal} : (tensor<1x3x2xf32>) -> tensor<1x6xf32>
