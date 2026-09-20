// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives %s | FileCheck %s
// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives %s 2>&1 >/dev/null | FileCheck %s --check-prefix=REMARK

// The pass rewrites every collective's input_mesh/output_mesh/
// reduction_groups/mapping operands into atomic form: one fresh factor per
// atom, one atom-to-atom mapping pair per atom position. The CHECK-DAG lines
// below pin the real rewritten structure (regenerated from the actual
// binary's output, per this repo's convention -- do not trust a remembered
// shape over the tool); they tolerate the old, now-dead operands the rewrite
// leaves behind (cse/canonicalize's job, not this pass's). The REMARK lines
// pin the same atom analysis as before, re-run against the REWRITTEN
// collective: it is a standing debugging aid, not scaffolding, so it stays
// live across this rewrite and its content changes exactly where the
// rewrite itself introduces finer mapping pairs (splitting one pair that
// spanned several atoms into one pair per atom). The REMARK RUN line
// discards stdout (the module dump) and feeds FileCheck stderr only, since a
// remark is a diagnostic, not part of the module text.
//
// P(e,s) in comments is a factor of extent e and stride s over axis P. An
// atom is printed as `<space><axis>.<atom>` (mesh/in/out/replicate), e.g.
// `mesh0.1` is mesh axis 0's atom #1; axis summary lines list atom extents
// major-first. A replicate atom has no persistent identity across factors
// (see CollectiveAtoms.h), so an atomized replicate factor is always a
// fresh, exactly-sized ReplicationAxis rather than a strided slice of the
// original, possibly coarser one.

// An already-atomic collective's own atoms are exactly its two whole mesh
// axes, one atom apiece, so atomizing rebuilds an equally trivial,
// structurally identical-shaped map: same atom coverage, freshly built ops.
// CHECK-LABEL: module @already_atomic
// CHECK-DAG: %[[P:.*]]:2 = distributed.GetPhysicalMeshAxes @mesh0
// CHECK-DAG: %[[MESH_IN:[0-9]+]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
// CHECK-DAG: %[[MESH_OUT:[0-9]+]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
// CHECK-DAG: %[[MAP:[0-9]+]] = axis.map %[[L0:[0-9]+]], %[[L1:[0-9]+]] to %[[R0:[0-9]+]], %[[R1:[0-9]+]] : [!axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<2>, !axis.factor_group<2>]
// CHECK-DAG: %[[L0]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
// CHECK-DAG: %[[R0]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
// CHECK-DAG: %[[L1]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
// CHECK-DAG: %[[R1]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
// CHECK-DAG: distributed.Collective %{{[0-9]+}} : tensor<1xf32> on %[[MESH_IN]] : <4> to tensor<1xf32> on %[[MESH_OUT]] : <4> reduces () maps %[[MAP]] : !axis.map
// REMARK: remark: mesh0: 2
// REMARK-NEXT: mesh1: 2
// REMARK-NEXT: in0: 1
// REMARK-NEXT: out0: 1
// REMARK-NEXT: pair0: mesh0.0 -> mesh0.0
// REMARK-NEXT: pair1: mesh1.0 -> mesh1.0
module @already_atomic {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %f0 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f1 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>

  %mesh_in = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %mesh_out = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)

  %lhs0 = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %rhs0 = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %lhs1 = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %rhs1 = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<2>, !axis.factor_group<2>]

  %global = tensor.empty() : tensor<4xf32>
  %local = distributed.CastGlobalToLocal %global axes (%mesh_in : !axis.factor_group<4>) : tensor<4xf32> -> tensor<1xf32>
  %h = distributed.Collective %local : tensor<1xf32> on %mesh_in : !axis.factor_group<4> to tensor<1xf32> on %mesh_out : !axis.factor_group<4> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<1xf32>> -> tensor<1xf32>
}

// -----

// A reduction group cutting P(8,1) at 4 splits a mapped P(8,1) and, through
// its pair, the tile factor. The atomize rewrite (Step 4) still needs to
// build this split; here the pass only reports it: mesh axis 0 and tile dim
// 0 both refine to atoms [2, 4], the reduction covers the extent-4 atom, and
// the tile<->mesh pair covers both atoms while the mesh<->replicate pair
// covers only the extent-2 one.
// CHECK-LABEL: module @reduction_cut_splits_rhs_factor
// CHECK-DAG: %[[MESH_IN:[0-9]+]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 4>, %{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 4, 1>)
// CHECK-DAG: %[[MESH_OUT:[0-9]+]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 4>, %{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 4, 1>)
// CHECK-DAG: %[[RED:[0-9]+]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 4, 1>)
// CHECK-DAG: %[[MAP:[0-9]+]] = axis.map %[[L0:[0-9]+]], %[[L1:[0-9]+]], %[[L2:[0-9]+]] to %[[R0:[0-9]+]], %[[R1:[0-9]+]], %[[R2:[0-9]+]] : [!axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<2>]
// CHECK-DAG: %[[L0]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 2, 4>)
// CHECK-DAG: %[[R0]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 4>)
// CHECK-DAG: %[[L1]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 4, 1>)
// CHECK-DAG: %[[R1]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 4, 1>)
// CHECK-DAG: %[[L2]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 4>)
// CHECK-DAG: %[[R2]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
// CHECK-DAG: distributed.Collective %{{[0-9]+}} : tensor<8xf32> on %[[MESH_IN]] : <8> to tensor<1xf32> on %[[MESH_OUT]] : <8> reduces (%[[RED]] : !axis.factor_group<4>) maps %[[MAP]] : !axis.map
// REMARK: remark: mesh0: 2, 4
// REMARK-NEXT: in0: 2, 4
// REMARK-NEXT: out0: 1
// REMARK-NEXT: reduction0: mesh0.1
// REMARK-NEXT: pair0: in0.0 -> mesh0.0
// REMARK-NEXT: pair1: in0.1 -> mesh0.1
// REMARK-NEXT: pair2: mesh0.0 -> replicate0.0
module @reduction_cut_splits_rhs_factor {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<8, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<8, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %tdim0 = axis.getaxis tensor<8xf32> 0

  %P_8_1 = axis.factor %p0 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %P_4_1 = axis.factor %p0 : !distributed.physical_comm_axis<8, 1> <4, 1>
  %P_2_4 = axis.factor %p0 : !distributed.physical_comm_axis<8, 1> <2, 4>
  %T_8_1 = axis.factor %tdim0 : !axis.shape_axis<tensor<8xf32>, 0> <8, 1>
  %Rep_2_1 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %mesh_in = axis.product (%P_8_1 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %mesh_out = axis.product (%P_8_1 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %red = axis.product (%P_4_1 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 4, 1>)
  %lhs0 = axis.product (%T_8_1 : !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 8, 1>)
  %rhs0 = axis.product (%P_8_1 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %lhs1 = axis.product (%P_2_4 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 4>)
  %rhs1 = axis.product (%Rep_2_1 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<8>, !axis.factor_group<2>] [!axis.factor_group<8>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<8xf32>
  %h = distributed.Collective %in : tensor<8xf32> on %mesh_in : !axis.factor_group<8> to tensor<1xf32> on %mesh_out : !axis.factor_group<8> reduces (%red : !axis.factor_group<4>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %sum = stablehlo.add %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<1xf32>> -> tensor<1xf32>
}

// -----

// Splitting a one-to-many pair cascades to a second pair that uses the same
// factor: the mesh axis's full-extent factor is forced to split [2, 4] to
// match the two-dim output tile, and that same split then propagates onto
// the replicate factor the other pair maps against it.
// CHECK-LABEL: module @pair_split_cascades_to_other_pair
// CHECK-DAG: %[[MESH_IN:[0-9]+]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 4>, %{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 4, 1>)
// CHECK-DAG: %[[MESH_OUT:[0-9]+]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 4>, %{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 4, 1>)
// CHECK-DAG: %[[MAP:[0-9]+]] = axis.map %[[L0:[0-9]+]], %[[L1:[0-9]+]], %[[L2:[0-9]+]], %[[L3:[0-9]+]] to %[[R0:[0-9]+]], %[[R1:[0-9]+]], %[[R2:[0-9]+]], %[[R3:[0-9]+]] : [!axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<4>] [!axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<4>]
// CHECK-DAG: %[[L0]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 4>)
// CHECK-DAG: %[[R0]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!axis.shape_axis<tensor<2x4xf32>, 0>, 2, 1>)
// CHECK-DAG: %[[L1]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 4, 1>)
// CHECK-DAG: %[[R1]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!axis.shape_axis<tensor<2x4xf32>, 1>, 4, 1>)
// CHECK-DAG: %[[L2]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
// CHECK-DAG: %[[R2]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 4>)
// CHECK-DAG: %[[L3]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
// CHECK-DAG: %[[R3]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 4, 1>)
// CHECK-DAG: distributed.Collective %{{[0-9]+}} : tensor<1xf32> on %[[MESH_IN]] : <8> to tensor<2x4xf32> on %[[MESH_OUT]] : <8> reduces () maps %[[MAP]] : !axis.map
// REMARK: remark: mesh0: 2, 4
// REMARK-NEXT: in0: 1
// REMARK-NEXT: out0: 2
// REMARK-NEXT: out1: 4
// REMARK-NEXT: pair0: mesh0.0 -> out0.0
// REMARK-NEXT: pair1: mesh0.1 -> out1.0
// REMARK-NEXT: pair2: replicate0.0 -> mesh0.0
// REMARK-NEXT: pair3: replicate1.0 -> mesh0.1
module @pair_split_cascades_to_other_pair {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<8, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<8, 1>
  %rep8 = distributed.ReplicationAxis 8 : !distributed.replication_axis<8>
  %tdim0 = axis.getaxis tensor<2x4xf32> 0
  %tdim1 = axis.getaxis tensor<2x4xf32> 1

  %P_8_1 = axis.factor %p0 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %D0_2_1 = axis.factor %tdim0 : !axis.shape_axis<tensor<2x4xf32>, 0> <2, 1>
  %D1_4_1 = axis.factor %tdim1 : !axis.shape_axis<tensor<2x4xf32>, 1> <4, 1>
  %Rep_8_1 = axis.factor %rep8 : !distributed.replication_axis<8> <8, 1>
  %mesh_in = axis.product (%P_8_1 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %mesh_out = axis.product (%P_8_1 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %lhs0 = axis.product (%P_8_1 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %rhs0 = axis.product (%D0_2_1 : !axis.axis_factor<!axis.shape_axis<tensor<2x4xf32>, 0>, 2, 1>, %D1_4_1 : !axis.axis_factor<!axis.shape_axis<tensor<2x4xf32>, 1>, 4, 1>)
  %lhs1 = axis.product (%Rep_8_1 : !axis.axis_factor<!distributed.replication_axis<8>, 8, 1>)
  %rhs1 = axis.product (%P_8_1 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<8>, !axis.factor_group<8>] [!axis.factor_group<8>, !axis.factor_group<8>]

  %in = tensor.empty() : tensor<1xf32>
  %h = distributed.Collective %in : tensor<1xf32> on %mesh_in : !axis.factor_group<8> to tensor<2x4xf32> on %mesh_out : !axis.factor_group<8> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<2x4xf32>> -> tensor<2x4xf32>
}

// -----

// Pairs in reverse dependency order: the cascade crosses two axes and needs
// a fixpoint. The output tile forces mesh axis 0 to split [2, 4]; that
// forces mesh axis 1 to split the same way (through the mesh-to-mesh pair);
// that in turn forces the replicate factor to split the same way too.
// CHECK-LABEL: module @cascade_reaches_fixpoint_across_axes
// CHECK-DAG: %[[MESH_IN:[0-9]+]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 8>, 2, 4>, %{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 8>, 4, 1>, %{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 4>, %{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 4, 1>)
// CHECK-DAG: %[[MESH_OUT:[0-9]+]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 8>, 2, 4>, %{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 8>, 4, 1>, %{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 4>, %{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 4, 1>)
// CHECK-DAG: %[[MAP:[0-9]+]] = axis.map %[[L0:[0-9]+]], %[[L1:[0-9]+]], %[[L2:[0-9]+]], %[[L3:[0-9]+]], %[[L4:[0-9]+]], %[[L5:[0-9]+]] to %[[R0:[0-9]+]], %[[R1:[0-9]+]], %[[R2:[0-9]+]], %[[R3:[0-9]+]], %[[R4:[0-9]+]], %[[R5:[0-9]+]] : [!axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<4>] [!axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<4>]
// CHECK-DAG: %[[L0]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
// CHECK-DAG: %[[R0]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 4>)
// CHECK-DAG: %[[L1]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
// CHECK-DAG: %[[R1]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 4, 1>)
// CHECK-DAG: %[[L2]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 4>)
// CHECK-DAG: %[[R2]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 8>, 2, 4>)
// CHECK-DAG: %[[L3]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 4, 1>)
// CHECK-DAG: %[[R3]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 8>, 4, 1>)
// CHECK-DAG: %[[L4]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 8>, 2, 4>)
// CHECK-DAG: %[[R4]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!axis.shape_axis<tensor<2x4xf32>, 0>, 2, 1>)
// CHECK-DAG: %[[L5]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 8>, 4, 1>)
// CHECK-DAG: %[[R5]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!axis.shape_axis<tensor<2x4xf32>, 1>, 4, 1>)
// CHECK-DAG: distributed.Collective %{{[0-9]+}} : tensor<1xf32> on %[[MESH_IN]] : <64> to tensor<2x4xf32> on %[[MESH_OUT]] : <64> reduces () maps %[[MAP]] : !axis.map
// REMARK: remark: mesh0: 2, 4
// REMARK-NEXT: mesh1: 2, 4
// REMARK-NEXT: in0: 1
// REMARK-NEXT: out0: 2
// REMARK-NEXT: out1: 4
// REMARK-NEXT: pair0: replicate0.0 -> mesh1.0
// REMARK-NEXT: pair1: replicate1.0 -> mesh1.1
// REMARK-NEXT: pair2: mesh1.0 -> mesh0.0
// REMARK-NEXT: pair3: mesh1.1 -> mesh0.1
// REMARK-NEXT: pair4: mesh0.0 -> out0.0
// REMARK-NEXT: pair5: mesh0.1 -> out1.0
module @cascade_reaches_fixpoint_across_axes {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<8, 8>, !distributed.physical_comm_axis<8, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<8, 8>, !distributed.physical_comm_axis<8, 1>
  %rep8 = distributed.ReplicationAxis 8 : !distributed.replication_axis<8>
  %tdim0 = axis.getaxis tensor<2x4xf32> 0
  %tdim1 = axis.getaxis tensor<2x4xf32> 1

  %P_8_1 = axis.factor %p0 : !distributed.physical_comm_axis<8, 8> <8, 1>
  %Q_8_1 = axis.factor %p1 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %D0_2_1 = axis.factor %tdim0 : !axis.shape_axis<tensor<2x4xf32>, 0> <2, 1>
  %D1_4_1 = axis.factor %tdim1 : !axis.shape_axis<tensor<2x4xf32>, 1> <4, 1>
  %Rep_8_1 = axis.factor %rep8 : !distributed.replication_axis<8> <8, 1>
  %mesh_in = axis.product (%P_8_1 : !axis.axis_factor<!distributed.physical_comm_axis<8, 8>, 8, 1>, %Q_8_1 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %mesh_out = axis.product (%P_8_1 : !axis.axis_factor<!distributed.physical_comm_axis<8, 8>, 8, 1>, %Q_8_1 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %lhs0 = axis.product (%Rep_8_1 : !axis.axis_factor<!distributed.replication_axis<8>, 8, 1>)
  %rhs0 = axis.product (%Q_8_1 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %lhs1 = axis.product (%Q_8_1 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %rhs1 = axis.product (%P_8_1 : !axis.axis_factor<!distributed.physical_comm_axis<8, 8>, 8, 1>)
  %lhs2 = axis.product (%P_8_1 : !axis.axis_factor<!distributed.physical_comm_axis<8, 8>, 8, 1>)
  %rhs2 = axis.product (%D0_2_1 : !axis.axis_factor<!axis.shape_axis<tensor<2x4xf32>, 0>, 2, 1>, %D1_4_1 : !axis.axis_factor<!axis.shape_axis<tensor<2x4xf32>, 1>, 4, 1>)
  %map = axis.map %lhs0, %lhs1, %lhs2 to %rhs0, %rhs1, %rhs2 : [!axis.factor_group<8>, !axis.factor_group<8>, !axis.factor_group<8>] [!axis.factor_group<8>, !axis.factor_group<8>, !axis.factor_group<8>]

  %in = tensor.empty() : tensor<1xf32>
  %h = distributed.Collective %in : tensor<1xf32> on %mesh_in : !axis.factor_group<64> to tensor<2x4xf32> on %mesh_out : !axis.factor_group<64> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<2x4xf32>> -> tensor<2x4xf32>
}

// -----

// Two collectives share one map and reduce over different groups; each
// atomizes into its own, fully independent set of new mesh/reduction/map
// ops -- neither the rewrite of one nor the other ever reuses the other's
// freshly built values, even though both originally shared one axis.map.
// CHECK-LABEL: module @shared_map_different_reductions
// CHECK-DAG: %[[MESH_IN1:[0-9]+]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 4>, %{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 4, 1>)
// CHECK-DAG: %[[MESH_OUT1:[0-9]+]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 4>, %{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 4, 1>)
// CHECK-DAG: %[[RED1:[0-9]+]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 4, 1>)
// CHECK-DAG: %[[MAP1:[0-9]+]] = axis.map %[[L10:[0-9]+]], %[[L11:[0-9]+]], %[[L12:[0-9]+]] to %[[R10:[0-9]+]], %[[R11:[0-9]+]], %[[R12:[0-9]+]] : [!axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<2>]
// CHECK-DAG: %[[L10]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 2, 4>)
// CHECK-DAG: %[[R10]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 4>)
// CHECK-DAG: %[[L11]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 4, 1>)
// CHECK-DAG: %[[R11]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 4, 1>)
// CHECK-DAG: %[[L12]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 4>)
// CHECK-DAG: %[[R12]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
// CHECK-DAG: distributed.Collective %{{[0-9]+}} : tensor<8xf32> on %[[MESH_IN1]] : <8> to tensor<1xf32> on %[[MESH_OUT1]] : <8> reduces (%[[RED1]] : !axis.factor_group<4>) maps %[[MAP1]] : !axis.map
// CHECK-DAG: %[[MESH_IN2:[0-9]+]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 4>, %{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 2>, %{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 1>)
// CHECK-DAG: %[[MESH_OUT2:[0-9]+]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 4>, %{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 2>, %{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 1>)
// CHECK-DAG: %[[RED2:[0-9]+]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 1>, %{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 2>)
// CHECK-DAG: %[[MAP2:[0-9]+]] = axis.map %[[L20:[0-9]+]], %[[L21:[0-9]+]], %[[L22:[0-9]+]], %[[L23:[0-9]+]] to %[[R20:[0-9]+]], %[[R21:[0-9]+]], %[[R22:[0-9]+]], %[[R23:[0-9]+]] : [!axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>]
// CHECK-DAG: %[[L20]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 2, 4>)
// CHECK-DAG: %[[R20]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 4>)
// CHECK-DAG: %[[L21]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 2, 2>)
// CHECK-DAG: %[[R21]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 2>)
// CHECK-DAG: %[[L22]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 2, 1>)
// CHECK-DAG: %[[R22]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 1>)
// CHECK-DAG: %[[L23]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 4>)
// CHECK-DAG: %[[R23]] = axis.product (%{{[0-9]+}} : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
// CHECK-DAG: distributed.Collective %{{[0-9]+}} : tensor<8xf32> on %[[MESH_IN2]] : <8> to tensor<1xf32> on %[[MESH_OUT2]] : <8> reduces (%[[RED2]] : !axis.factor_group<4>) maps %[[MAP2]] : !axis.map
// The first collective's reduction (P(4,1) alone) matches case A exactly.
// REMARK: remark: mesh0: 2, 4
// REMARK-NEXT: in0: 2, 4
// REMARK-NEXT: out0: 1
// REMARK-NEXT: reduction0: mesh0.1
// REMARK-NEXT: pair0: in0.0 -> mesh0.0
// REMARK-NEXT: pair1: in0.1 -> mesh0.1
// REMARK-NEXT: pair2: mesh0.0 -> replicate0.0
// The second's reduction (P(2,1) and P(2,2)) forces a finer 3-atom split of
// the same mesh axis; reduction0 lists its two factors in their own
// (minor-first) order, mesh0.2 then mesh0.1.
// REMARK: remark: mesh0: 2, 2, 2
// REMARK-NEXT: in0: 2, 2, 2
// REMARK-NEXT: out0: 1
// REMARK-NEXT: reduction0: mesh0.2, mesh0.1
// REMARK-NEXT: pair0: in0.0 -> mesh0.0
// REMARK-NEXT: pair1: in0.1 -> mesh0.1
// REMARK-NEXT: pair2: in0.2 -> mesh0.2
// REMARK-NEXT: pair3: mesh0.0 -> replicate0.0
module @shared_map_different_reductions {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<8, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<8, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %tdim0 = axis.getaxis tensor<8xf32> 0

  %P_8_1 = axis.factor %p0 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %P_4_1 = axis.factor %p0 : !distributed.physical_comm_axis<8, 1> <4, 1>
  %P_2_4 = axis.factor %p0 : !distributed.physical_comm_axis<8, 1> <2, 4>
  %P_2_1 = axis.factor %p0 : !distributed.physical_comm_axis<8, 1> <2, 1>
  %P_2_2 = axis.factor %p0 : !distributed.physical_comm_axis<8, 1> <2, 2>
  %T_8_1 = axis.factor %tdim0 : !axis.shape_axis<tensor<8xf32>, 0> <8, 1>
  %Rep_2_1 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %mesh = axis.product (%P_8_1 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %red1 = axis.product (%P_4_1 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 4, 1>)
  %red2 = axis.product (%P_2_1 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 1>, %P_2_2 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 2>)
  %lhs0 = axis.product (%T_8_1 : !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 8, 1>)
  %rhs0 = axis.product (%P_8_1 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %lhs1 = axis.product (%P_2_4 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 2, 4>)
  %rhs1 = axis.product (%Rep_2_1 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<8>, !axis.factor_group<2>] [!axis.factor_group<8>, !axis.factor_group<2>]

  %in1 = tensor.empty() : tensor<8xf32>
  %h1 = distributed.Collective %in1 : tensor<8xf32> on %mesh : !axis.factor_group<8> to tensor<1xf32> on %mesh : !axis.factor_group<8> reduces (%red1 : !axis.factor_group<4>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %sum = stablehlo.add %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }
  %v1 = distributed.Await %h1 : !distributed.asynch_handle<tensor<1xf32>> -> tensor<1xf32>
  %in2 = tensor.empty() : tensor<8xf32>
  %h2 = distributed.Collective %in2 : tensor<8xf32> on %mesh : !axis.factor_group<8> to tensor<1xf32> on %mesh : !axis.factor_group<8> reduces (%red2 : !axis.factor_group<4>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %sum = stablehlo.add %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }
  %v2 = distributed.Await %h2 : !distributed.asynch_handle<tensor<1xf32>> -> tensor<1xf32>
}
