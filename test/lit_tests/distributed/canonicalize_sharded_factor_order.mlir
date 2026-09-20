// RUN: enzymexlamlir-opt --canonicalize-sharded-factor-order --cse -split-input-file %s | FileCheck %s

// A kernel whose sole dimension is already sharded-major/local-minor within
// one composite axis.product slot (the same shape as lower_kernels.mlir's
// composite_kernel) -- must be a complete no-op: this is the pure-metadata
// path (Part L case (1)), and it already sees a canonical order.
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
// CHECK: %[[CG:.*]] = axis.product (%{{.*}} : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>, %{{.*}} : !axis.axis_factor<!distributed.device_local_axis<2>, 2, 1>)
// CHECK: %[[CST:.*]] = stablehlo.constant
// CHECK: distributed.DistributedKernel (%[[CST]] : tensor<1xf32>) <[<dim_partitioning_axes = {{\[\[0\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: -> (tensor<1xf32>) <[<dim_partitioning_axes = {{\[\[0\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: axes (%[[CG]] : !axis.factor_group<4>) {
// CHECK-NEXT: ^bb0(%arg0: tensor<4xf32>):
// CHECK-NEXT: distributed.DistributedYield (%arg0 : tensor<4xf32>)

// -----

// Same shape, but the one slot's own factor order is sandwiched the other
// way (local major, shard minor). Part L case (1): this is a
// DistributedKernelOp's own boundary, so it's a PURE METADATA fix -- a new
// canonical axis.product slot is appended and argument_shardings is
// repointed at it. output_shardings is derived separately, from the
// DistributedYieldOp's own operand (here, %arg0 itself, an identity kernel)
// via resolveCurrentSharding -- since that resolves to the SAME
// already-canonicalized argument_shardings slot, output_shardings ends up
// referencing that same slot too, rather than a redundant second one. The
// kernel body (block argument, yield) is completely untouched: no
// stablehlo.reshape/transpose anywhere.
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
// CHECK-DAG: %[[ORIGCG:.*]] = axis.product (%{{.*}} : !axis.axis_factor<!distributed.device_local_axis<2>, 2, 1>, %{{.*}} : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>)
// CHECK-DAG: %[[NEWCG:.*]] = axis.product (%{{.*}} : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>, %{{.*}} : !axis.axis_factor<!distributed.device_local_axis<2>, 2, 1>)
// CHECK: distributed.DistributedKernel (%{{.*}} : tensor<1xf32>) <[<dim_partitioning_axes = {{\[\[1\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: -> (tensor<1xf32>) <[<dim_partitioning_axes = {{\[\[1\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: axes (%[[ORIGCG]] : !axis.factor_group<4>, %[[NEWCG]] : !axis.factor_group<4>) {
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

// A merge (join) reshape strictly inside a kernel body: two ALREADY
// individually-canonical dims (X = Xshard(2) major/Xlocal(3) minor,
// Y = Yshard(2) major/Ylocal(2) minor) get merged into one compound
// dimension. Unlike the split case above, this genuinely needs the real
// split+transpose+merge recipe (per sanity_check/09/10: a merge's natural
// order [Xshard,Xlocal,Yshard,Yshard] cannot be relabeled into canonical
// [Xshard,Yshard,Xlocal,Ylocal] order without physically rearranging bytes).
// Critically, the ORIGINAL reshape's own distributed.output_shardings must
// stay describing its TRUE (natural, non-canonical) structural output --
// only the chain's own final merge op declares the canonical order, matching
// the kernel's own (separately-derived, via the yield) output_shardings.
// Getting this backwards (overwriting the original reshape's own declared
// output to the post-chain value) was a real bug caught by this test: Shardy
// mechanically derives a reshape's own local type from its structural rule
// PLUS its declared operand sharding, so a mismatched self-declaration on
// the reshape itself, not just at a boundary, causes a real lowering
// failure, independent of whatever the kernel or a downstream consumer
// separately declares.
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
// CHECK: distributed.DistributedKernel (%{{.*}} : tensor<1x1xf32>) <[<dim_partitioning_axes = {{\[\[0\], \[1\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: -> (tensor<1xf32>) <[<dim_partitioning_axes = {{\[\[2, 4, 3, 5\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: axes
// CHECK-NEXT: ^bb0(%arg0: tensor<6x4xf32>):
// CHECK-NEXT: %[[MERGED:.*]] = stablehlo.reshape %arg0 {distributed.argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = {{\[\[0\], \[1\]\]}} : unreduced_axes = []>]>, distributed.output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = {{\[\[0, 1\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: %[[SPLIT:.*]] = stablehlo.reshape %[[MERGED]] {canonicalize_sharded_factor_order.internal, distributed.argument_shardings = {{.*}}, distributed.output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = {{\[\[2\], \[3\], \[4\], \[5\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: %[[MANUAL:.*]] = distributed.ManualComputation (%[[SPLIT]] : tensor<2x3x2x2xf32>) <[<dim_partitioning_axes = {{\[\[2\], \[3\], \[4\], \[5\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: manual_axes [2, 4]
// CHECK-NEXT: -> (tensor<24xf32>) <[<dim_partitioning_axes = {{\[\[2, 4, 3, 5\]\]}} : unreduced_axes = []>]> {
// CHECK-NEXT: ^bb0(%[[LARG:.*]]: tensor<1x3x1x2xf32>):
// CHECK-NEXT: %[[LMERGE:.*]] = stablehlo.reshape %[[LARG]] {canonicalize_sharded_factor_order.internal} : (tensor<1x3x1x2xf32>) -> tensor<6xf32>
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
// The sandwiched slot is split once and every use of it (both operands and
// the result of the concatenate) shares that one new slot: LowerKernels names
// mesh axes by slot index, so separate slots for the same factors would look
// like different shardings.
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
// CHECK-DAG: %[[SANDWICHED:.*]] = axis.product (%{{.*}} : !axis.axis_factor<!distributed.device_local_axis<2>, 2, 1>, %{{.*}} : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>)
// CHECK-DAG: %[[CANON:.*]] = axis.product (%{{.*}} : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>, %{{.*}} : !axis.axis_factor<!distributed.device_local_axis<2>, 2, 1>)
// CHECK-DAG: %[[LOCAL:.*]] = axis.product (%{{.*}} : !axis.axis_factor<!distributed.device_local_axis<1>, 1, 1>)
// CHECK: distributed.DistributedKernel (%{{.*}} : tensor<1x1xf32>) <[<dim_partitioning_axes = {{\[\[\], \[\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: -> (tensor<1x2xf32>) <[<dim_partitioning_axes = {{\[\[[0-9]+\], \[[0-9]+\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: axes (%[[SANDWICHED]] : !axis.factor_group<4>, %[[LOCAL]] : !axis.factor_group<1>, %[[CANON]] : !axis.factor_group<4>) {
// CHECK-NEXT: ^bb0(%arg0: tensor<4x1xf32>):
// CHECK-NEXT: %[[CAT:.*]] = stablehlo.concatenate %arg0, %arg0, dim = 1 {distributed.argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = {{\[\[[0-9]+\], \[1\]\]}} : unreduced_axes = []>, <dim_partitioning_axes = {{\[\[[0-9]+\], \[1\]\]}} : unreduced_axes = []>]>, distributed.output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = {{\[\[[0-9]+\], \[1\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: distributed.DistributedYield (%[[CAT]] : tensor<4x2xf32>)
