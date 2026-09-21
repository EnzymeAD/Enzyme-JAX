// RUN: enzymexlamlir-opt --split-input-file --distributed-fuse-collectives %s 2>&1 >/dev/null | FileCheck %s --check-prefix=REMARK
// RUN: enzymexlamlir-opt --split-input-file --distributed-fuse-collectives %s | FileCheck %s --check-prefix=IR
// RUN: enzymexlamlir-opt --split-input-file --distributed-fuse-collectives --distributed-atomize-collectives --distributed-print-collective-plan %s 2>&1 >/dev/null | FileCheck %s --check-prefix=PLAN
// RUN: enzymexlamlir-opt --split-input-file --distributed-fuse-collectives %s -o %t.once
// RUN: enzymexlamlir-opt --split-input-file --distributed-fuse-collectives %t.once -o %t.twice
// RUN: diff -u %t.once %t.twice

// distributed-fuse-collectives fuses a collective C1 with the collective C2
// that consumes its result when nothing else uses that result. Every module
// below is a chain built the way distributed-make-replications-explicit
// leaves collectives (each mesh atom on both sides of both collectives). The
// four RUN groups check, in order: the pass's own remarks (one per skipped
// pair, on C2); the IR shape (how many collectives are left); the role table
// row of each resulting collective, read through the print pass after
// atomizing the unfused remainder; and idempotence.
//
// A mapping pair (L -> R) sends input digit L to output digit R, so on the
// fused collective a mesh atom's `in` role is where its input digit goes and
// its `out` role is where its output digit comes from (see
// print_collective_plan.mlir). Reading a pair chain by hand:
//   C1: L1 -> m and C2: m -> R2 fuse to L1 -> R2, and a mesh atom C2
//   reduces is reduced at the C1 digit that fed it.

// All-reduce over a followed by a slice of the replicated result. Alone, the reduction is an all-reduce (reduced -> replicate) and the slice a local slice
// (replicate -> tile); fused, the reduced atom's output comes from the input tile, which is the reduce-scatter row. The replicate -> replicate pair the
// composition produces relates no real digits and is dropped. pair0 says the reduced atom's output digit is the major half of the input tile (in0.0),
// matching the slice's major-first (mesh, tile) split.
// IR-LABEL: module @reduce_then_slice
// IR: distributed.Collective {{.*}} : tensor<4xf32> on {{.*}} to tensor<2xf32> on {{.*}} reduces (
// IR-NOT: distributed.Collective
// PLAN: remark: mesh0.0: extent 2 stride 1 in reduced out tile(in0.0) => reduce-scatter
// PLAN-NEXT: in0.0: extent 2 pairs mesh0.0
// PLAN-NEXT: in0.1: extent 2 pairs out0.0
// PLAN-NEXT: out0.0: extent 2 pairs in0.1
// PLAN-NEXT: reduction: add
// PLAN-NEXT: payload bytes: 16
module @reduce_then_slice {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>

  %ax1 = axis.getaxis tensor<4xf32> 0
  %ax2 = axis.getaxis tensor<4xf32> 0
  %ax3 = axis.getaxis tensor<2xf32> 0
  %in = tensor.empty() : tensor<4xf32>
  %f4 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh5 = axis.product (%f4 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f6 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh7 = axis.product (%f6 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f8 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %red9 = axis.product (%f8 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f10 = axis.factor %ax1 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %f11 = axis.factor %ax2 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %rep12 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %f13 = axis.factor %rep12 : !distributed.replication_axis<2> <2, 1>
  %f14 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs15 = axis.product (%f10 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %rhs16 = axis.product (%f11 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %lhs17 = axis.product (%f13 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %rhs18 = axis.product (%f14 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map19 = axis.map %lhs15, %lhs17 to %rhs16, %rhs18 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]
  %h20 = distributed.Collective %in : tensor<4xf32> on %mesh5 : !axis.factor_group<2> to tensor<4xf32> on %mesh7 : !axis.factor_group<2> reduces (%red9 : !axis.factor_group<2>) maps %map19 : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }
  %v21 = distributed.Await %h20 : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
  %f22 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh23 = axis.product (%f22 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f24 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh25 = axis.product (%f24 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f26 = axis.factor %ax2 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %f27 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f28 = axis.factor %ax3 : !axis.shape_axis<tensor<2xf32>, 0> <2, 1>
  %f29 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rep30 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %f31 = axis.factor %rep30 : !distributed.replication_axis<2> <2, 1>
  %lhs32 = axis.product (%f26 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %rhs33 = axis.product (%f27 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %f28 : !axis.axis_factor<!axis.shape_axis<tensor<2xf32>, 0>, 2, 1>)
  %lhs34 = axis.product (%f29 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %rhs35 = axis.product (%f31 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %map36 = axis.map %lhs32, %lhs34 to %rhs33, %rhs35 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]
  %h37 = distributed.Collective %v21 : tensor<4xf32> on %mesh23 : !axis.factor_group<2> to tensor<2xf32> on %mesh25 : !axis.factor_group<2> reduces () maps %map36 : !axis.map
  %v38 = distributed.Await %h37 : !distributed.asynch_handle<tensor<2xf32>> -> tensor<2xf32>
}

// -----

// All-reduce over mesh axis 0 followed by a swap of the two mesh axes. Composing (b -> b) with the swap's (b -> a) gives b -> a, and the reduced atom a is
// followed through the swap's (a -> b): the output digit of b is the broadcast of the reduced sum. The fused collective's rows are (reduced, mesh) and (mesh,
// replicate): a reduction then a permute.
// IR-LABEL: module @reduce_then_permute
// IR: distributed.Collective
// IR-NOT: distributed.Collective
// PLAN: remark: mesh0.0: extent 2 stride 1 in reduced out mesh(mesh1.0) => reduce-then-permute
// PLAN-NEXT: mesh1.0: extent 2 stride 1 in mesh(mesh0.0) out replicate => permute
// PLAN-NEXT: in0.0: extent 4 pairs out0.0
// PLAN-NEXT: out0.0: extent 4 pairs in0.0
// PLAN-NEXT: reduction: add
// PLAN-NEXT: payload bytes: 16
module @reduce_then_permute {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>

  %ax1 = axis.getaxis tensor<4xf32> 0
  %ax2 = axis.getaxis tensor<4xf32> 0
  %in = tensor.empty() : tensor<4xf32>
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f4 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh5 = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f4 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f6 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f7 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh8 = axis.product (%f6 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f9 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %red10 = axis.product (%f9 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f11 = axis.factor %ax1 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %f12 = axis.factor %ax2 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %rep13 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %f14 = axis.factor %rep13 : !distributed.replication_axis<2> <2, 1>
  %f15 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f16 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f17 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs18 = axis.product (%f11 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %rhs19 = axis.product (%f12 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %lhs20 = axis.product (%f14 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %rhs21 = axis.product (%f15 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %lhs22 = axis.product (%f16 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %rhs23 = axis.product (%f17 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map24 = axis.map %lhs18, %lhs20, %lhs22 to %rhs19, %rhs21, %rhs23 : [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>]
  %h25 = distributed.Collective %in : tensor<4xf32> on %mesh5 : !axis.factor_group<4> to tensor<4xf32> on %mesh8 : !axis.factor_group<4> reduces (%red10 : !axis.factor_group<2>) maps %map24 : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }
  %v26 = distributed.Await %h25 : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
  %f27 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f28 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh29 = axis.product (%f27 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f28 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f30 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f31 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh32 = axis.product (%f30 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f31 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f33 = axis.factor %ax2 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %f34 = axis.factor %ax2 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %f35 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f36 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f37 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f38 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %lhs39 = axis.product (%f33 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %rhs40 = axis.product (%f34 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %lhs41 = axis.product (%f35 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %rhs42 = axis.product (%f36 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %lhs43 = axis.product (%f37 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %rhs44 = axis.product (%f38 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %map45 = axis.map %lhs39, %lhs41, %lhs43 to %rhs40, %rhs42, %rhs44 : [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>]
  %h46 = distributed.Collective %v26 : tensor<4xf32> on %mesh29 : !axis.factor_group<4> to tensor<4xf32> on %mesh32 : !axis.factor_group<4> reduces () maps %map45 : !axis.map
  %v47 = distributed.Await %h46 : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
}

// -----

// All-gather along tensor dim 0 then slice along dim 1, both on the same mesh atom: the gathered atom's input digit ends up in out dim 0 and its output
// digit is taken from in dim 1, an all-to-all (tile -> tile).
// IR-LABEL: module @gather_then_slice
// IR: distributed.Collective
// IR-NOT: distributed.Collective
// PLAN: remark: mesh0.0: extent 2 stride 1 in tile(out0.0) out tile(in1.0) => tile-to-tile
// PLAN-NEXT: in0.0: extent 2 pairs out0.1
// PLAN-NEXT: in1.0: extent 2 pairs mesh0.0
// PLAN-NEXT: in1.1: extent 2 pairs out1.0
// PLAN-NEXT: out0.0: extent 2 pairs mesh0.0
// PLAN-NEXT: out0.1: extent 2 pairs in0.0
// PLAN-NEXT: out1.0: extent 2 pairs in1.1
// PLAN-NEXT: payload bytes: 32
module @gather_then_slice {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>

  %ax1 = axis.getaxis tensor<2x4xf32> 0
  %ax2 = axis.getaxis tensor<2x4xf32> 1
  %ax3 = axis.getaxis tensor<4x4xf32> 0
  %ax4 = axis.getaxis tensor<4x4xf32> 1
  %ax5 = axis.getaxis tensor<4x2xf32> 0
  %ax6 = axis.getaxis tensor<4x2xf32> 1
  %in = tensor.empty() : tensor<2x4xf32>
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh8 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f9 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh10 = axis.product (%f9 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f11 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f12 = axis.factor %ax1 : !axis.shape_axis<tensor<2x4xf32>, 0> <2, 1>
  %f13 = axis.factor %ax3 : !axis.shape_axis<tensor<4x4xf32>, 0> <4, 1>
  %f14 = axis.factor %ax2 : !axis.shape_axis<tensor<2x4xf32>, 1> <4, 1>
  %f15 = axis.factor %ax4 : !axis.shape_axis<tensor<4x4xf32>, 1> <4, 1>
  %rep16 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %f17 = axis.factor %rep16 : !distributed.replication_axis<2> <2, 1>
  %f18 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs19 = axis.product (%f11 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %f12 : !axis.axis_factor<!axis.shape_axis<tensor<2x4xf32>, 0>, 2, 1>)
  %rhs20 = axis.product (%f13 : !axis.axis_factor<!axis.shape_axis<tensor<4x4xf32>, 0>, 4, 1>)
  %lhs21 = axis.product (%f14 : !axis.axis_factor<!axis.shape_axis<tensor<2x4xf32>, 1>, 4, 1>)
  %rhs22 = axis.product (%f15 : !axis.axis_factor<!axis.shape_axis<tensor<4x4xf32>, 1>, 4, 1>)
  %lhs23 = axis.product (%f17 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %rhs24 = axis.product (%f18 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map25 = axis.map %lhs19, %lhs21, %lhs23 to %rhs20, %rhs22, %rhs24 : [!axis.factor_group<4>, !axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<4>, !axis.factor_group<2>]
  %h26 = distributed.Collective %in : tensor<2x4xf32> on %mesh8 : !axis.factor_group<2> to tensor<4x4xf32> on %mesh10 : !axis.factor_group<2> reduces () maps %map25 : !axis.map
  %v27 = distributed.Await %h26 : !distributed.asynch_handle<tensor<4x4xf32>> -> tensor<4x4xf32>
  %f28 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh29 = axis.product (%f28 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f30 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh31 = axis.product (%f30 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f32 = axis.factor %ax3 : !axis.shape_axis<tensor<4x4xf32>, 0> <4, 1>
  %f33 = axis.factor %ax5 : !axis.shape_axis<tensor<4x2xf32>, 0> <4, 1>
  %f34 = axis.factor %ax4 : !axis.shape_axis<tensor<4x4xf32>, 1> <4, 1>
  %f35 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f36 = axis.factor %ax6 : !axis.shape_axis<tensor<4x2xf32>, 1> <2, 1>
  %f37 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rep38 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %f39 = axis.factor %rep38 : !distributed.replication_axis<2> <2, 1>
  %lhs40 = axis.product (%f32 : !axis.axis_factor<!axis.shape_axis<tensor<4x4xf32>, 0>, 4, 1>)
  %rhs41 = axis.product (%f33 : !axis.axis_factor<!axis.shape_axis<tensor<4x2xf32>, 0>, 4, 1>)
  %lhs42 = axis.product (%f34 : !axis.axis_factor<!axis.shape_axis<tensor<4x4xf32>, 1>, 4, 1>)
  %rhs43 = axis.product (%f35 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %f36 : !axis.axis_factor<!axis.shape_axis<tensor<4x2xf32>, 1>, 2, 1>)
  %lhs44 = axis.product (%f37 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %rhs45 = axis.product (%f39 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %map46 = axis.map %lhs40, %lhs42, %lhs44 to %rhs41, %rhs43, %rhs45 : [!axis.factor_group<4>, !axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<4>, !axis.factor_group<2>]
  %h47 = distributed.Collective %v27 : tensor<4x4xf32> on %mesh29 : !axis.factor_group<2> to tensor<4x2xf32> on %mesh31 : !axis.factor_group<2> reduces () maps %map46 : !axis.map
  %v48 = distributed.Await %h47 : !distributed.asynch_handle<tensor<4x2xf32>> -> tensor<4x2xf32>
}

// -----

// A swap of the two mesh axes followed by an all-reduce over axis 0. Only the second collective reduces; the fused collective takes its reduction body, and the
// reduced atom is traced back through the swap's (b -> a) to mesh axis 1 (reduction0: mesh1.0).
// IR-LABEL: module @permute_then_reduce
// IR: distributed.Collective
// IR-NOT: distributed.Collective
// PLAN: remark: mesh0.0: extent 2 stride 1 in mesh(mesh1.0) out replicate => permute
// PLAN-NEXT: mesh1.0: extent 2 stride 1 in reduced out mesh(mesh0.0) => reduce-then-permute
// PLAN-NEXT: in0.0: extent 4 pairs out0.0
// PLAN-NEXT: out0.0: extent 4 pairs in0.0
// PLAN-NEXT: reduction: add
// PLAN-NEXT: payload bytes: 16
module @permute_then_reduce {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>

  %ax1 = axis.getaxis tensor<4xf32> 0
  %ax2 = axis.getaxis tensor<4xf32> 0
  %in = tensor.empty() : tensor<4xf32>
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f4 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh5 = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f4 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f6 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f7 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh8 = axis.product (%f6 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f9 = axis.factor %ax1 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %f10 = axis.factor %ax2 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %f11 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f12 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f13 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f14 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %lhs15 = axis.product (%f9 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %rhs16 = axis.product (%f10 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %lhs17 = axis.product (%f11 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %rhs18 = axis.product (%f12 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %lhs19 = axis.product (%f13 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %rhs20 = axis.product (%f14 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %map21 = axis.map %lhs15, %lhs17, %lhs19 to %rhs16, %rhs18, %rhs20 : [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>]
  %h22 = distributed.Collective %in : tensor<4xf32> on %mesh5 : !axis.factor_group<4> to tensor<4xf32> on %mesh8 : !axis.factor_group<4> reduces () maps %map21 : !axis.map
  %v23 = distributed.Await %h22 : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
  %f24 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f25 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh26 = axis.product (%f24 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f25 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f27 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f28 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh29 = axis.product (%f27 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f28 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f30 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %red31 = axis.product (%f30 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f32 = axis.factor %ax2 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %f33 = axis.factor %ax2 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %rep34 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %f35 = axis.factor %rep34 : !distributed.replication_axis<2> <2, 1>
  %f36 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f37 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f38 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs39 = axis.product (%f32 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %rhs40 = axis.product (%f33 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %lhs41 = axis.product (%f35 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %rhs42 = axis.product (%f36 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %lhs43 = axis.product (%f37 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %rhs44 = axis.product (%f38 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map45 = axis.map %lhs39, %lhs41, %lhs43 to %rhs40, %rhs42, %rhs44 : [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>]
  %h46 = distributed.Collective %v23 : tensor<4xf32> on %mesh26 : !axis.factor_group<4> to tensor<4xf32> on %mesh29 : !axis.factor_group<4> reduces (%red31 : !axis.factor_group<2>) maps %map45 : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }
  %v47 = distributed.Await %h46 : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
}

// -----

// Chain of three: all-reduce, slice, all-gather over the same atom. The first two fuse to a reduce-scatter, which then fuses with the gather; the whole chain
// collapses to an all-reduce whose tile pairs are the identity (reduce-scatter followed by all-gather is an all-reduce).
// IR-LABEL: module @reduce_slice_gather_chain
// IR: distributed.Collective {{.*}} : tensor<4xf32> on {{.*}} to tensor<4xf32> on {{.*}} reduces (
// IR-NOT: distributed.Collective
// PLAN: remark: mesh0.0: extent 2 stride 1 in reduced out replicate => all-reduce
// PLAN-NEXT: in0.0: extent 2 pairs out0.0
// PLAN-NEXT: in0.1: extent 2 pairs out0.1
// PLAN-NEXT: out0.0: extent 2 pairs in0.0
// PLAN-NEXT: out0.1: extent 2 pairs in0.1
// PLAN-NEXT: reduction: add
// PLAN-NEXT: payload bytes: 16
module @reduce_slice_gather_chain {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>

  %ax1 = axis.getaxis tensor<4xf32> 0
  %ax2 = axis.getaxis tensor<4xf32> 0
  %ax3 = axis.getaxis tensor<2xf32> 0
  %ax4 = axis.getaxis tensor<2xf32> 0
  %ax5 = axis.getaxis tensor<4xf32> 0
  %in = tensor.empty() : tensor<4xf32>
  %f6 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh7 = axis.product (%f6 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f8 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh9 = axis.product (%f8 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f10 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %red11 = axis.product (%f10 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f12 = axis.factor %ax1 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %f13 = axis.factor %ax2 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %rep14 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %f15 = axis.factor %rep14 : !distributed.replication_axis<2> <2, 1>
  %f16 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs17 = axis.product (%f12 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %rhs18 = axis.product (%f13 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %lhs19 = axis.product (%f15 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %rhs20 = axis.product (%f16 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map21 = axis.map %lhs17, %lhs19 to %rhs18, %rhs20 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]
  %h22 = distributed.Collective %in : tensor<4xf32> on %mesh7 : !axis.factor_group<2> to tensor<4xf32> on %mesh9 : !axis.factor_group<2> reduces (%red11 : !axis.factor_group<2>) maps %map21 : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }
  %v23 = distributed.Await %h22 : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
  %f24 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh25 = axis.product (%f24 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f26 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh27 = axis.product (%f26 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f28 = axis.factor %ax2 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %f29 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f30 = axis.factor %ax3 : !axis.shape_axis<tensor<2xf32>, 0> <2, 1>
  %f31 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rep32 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %f33 = axis.factor %rep32 : !distributed.replication_axis<2> <2, 1>
  %lhs34 = axis.product (%f28 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %rhs35 = axis.product (%f29 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %f30 : !axis.axis_factor<!axis.shape_axis<tensor<2xf32>, 0>, 2, 1>)
  %lhs36 = axis.product (%f31 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %rhs37 = axis.product (%f33 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %map38 = axis.map %lhs34, %lhs36 to %rhs35, %rhs37 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]
  %h39 = distributed.Collective %v23 : tensor<4xf32> on %mesh25 : !axis.factor_group<2> to tensor<2xf32> on %mesh27 : !axis.factor_group<2> reduces () maps %map38 : !axis.map
  %v40 = distributed.Await %h39 : !distributed.asynch_handle<tensor<2xf32>> -> tensor<2xf32>
  %f41 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh42 = axis.product (%f41 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f43 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh44 = axis.product (%f43 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f45 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f46 = axis.factor %ax3 : !axis.shape_axis<tensor<2xf32>, 0> <2, 1>
  %f47 = axis.factor %ax5 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %rep48 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %f49 = axis.factor %rep48 : !distributed.replication_axis<2> <2, 1>
  %f50 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs51 = axis.product (%f45 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %f46 : !axis.axis_factor<!axis.shape_axis<tensor<2xf32>, 0>, 2, 1>)
  %rhs52 = axis.product (%f47 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %lhs53 = axis.product (%f49 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %rhs54 = axis.product (%f50 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map55 = axis.map %lhs51, %lhs53 to %rhs52, %rhs54 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]
  %h56 = distributed.Collective %v40 : tensor<2xf32> on %mesh42 : !axis.factor_group<2> to tensor<4xf32> on %mesh44 : !axis.factor_group<2> reduces () maps %map55 : !axis.map
  %v57 = distributed.Await %h56 : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
}

// -----

// C1 slices a 16-element tile into (mesh a, 8-element tile) and C2 slices that 8-element tile into (mesh b, 4-element tile). C1's output tile is one factor of
// 8 and C2's input tile is cut 2*4, so the two collectives cut the middle tile differently; the joint refinement cuts C1's input tile as (2, 2, 4) to match. Both
// mesh atoms end up as local slices of their own piece of the input tile.
// IR-LABEL: module @mismatched_tile_cuts
// IR: distributed.Collective
// IR-NOT: distributed.Collective
// PLAN: remark: mesh0.0: extent 2 stride 1 in replicate out tile(in0.0) => local-slice
// PLAN-NEXT: mesh1.0: extent 2 stride 1 in replicate out tile(in0.1) => local-slice
// PLAN-NEXT: in0.0: extent 2 pairs mesh0.0
// PLAN-NEXT: in0.1: extent 2 pairs mesh1.0
// PLAN-NEXT: in0.2: extent 4 pairs out0.0
// PLAN-NEXT: out0.0: extent 4 pairs in0.2
// PLAN-NEXT: payload bytes: 64
module @mismatched_tile_cuts {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>

  %ax1 = axis.getaxis tensor<16xf32> 0
  %ax2 = axis.getaxis tensor<8xf32> 0
  %ax3 = axis.getaxis tensor<4xf32> 0
  %in = tensor.empty() : tensor<16xf32>
  %f4 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f5 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh6 = axis.product (%f4 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f5 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f8 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh9 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f8 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f10 = axis.factor %ax1 : !axis.shape_axis<tensor<16xf32>, 0> <16, 1>
  %f11 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f12 = axis.factor %ax2 : !axis.shape_axis<tensor<8xf32>, 0> <8, 1>
  %f13 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %rep14 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %f15 = axis.factor %rep14 : !distributed.replication_axis<2> <2, 1>
  %f16 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f17 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs18 = axis.product (%f10 : !axis.axis_factor<!axis.shape_axis<tensor<16xf32>, 0>, 16, 1>)
  %rhs19 = axis.product (%f11 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f12 : !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 8, 1>)
  %lhs20 = axis.product (%f13 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %rhs21 = axis.product (%f15 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %lhs22 = axis.product (%f16 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %rhs23 = axis.product (%f17 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map24 = axis.map %lhs18, %lhs20, %lhs22 to %rhs19, %rhs21, %rhs23 : [!axis.factor_group<16>, !axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<16>, !axis.factor_group<2>, !axis.factor_group<2>]
  %h25 = distributed.Collective %in : tensor<16xf32> on %mesh6 : !axis.factor_group<4> to tensor<8xf32> on %mesh9 : !axis.factor_group<4> reduces () maps %map24 : !axis.map
  %v26 = distributed.Await %h25 : !distributed.asynch_handle<tensor<8xf32>> -> tensor<8xf32>
  %f27 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f28 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh29 = axis.product (%f27 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f28 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f30 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f31 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh32 = axis.product (%f30 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f31 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f33 = axis.factor %ax2 : !axis.shape_axis<tensor<8xf32>, 0> <8, 1>
  %f34 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f35 = axis.factor %ax3 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %f36 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rep37 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %f38 = axis.factor %rep37 : !distributed.replication_axis<2> <2, 1>
  %f39 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f40 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %lhs41 = axis.product (%f33 : !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 8, 1>)
  %rhs42 = axis.product (%f34 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %f35 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %lhs43 = axis.product (%f36 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %rhs44 = axis.product (%f38 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %lhs45 = axis.product (%f39 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %rhs46 = axis.product (%f40 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %map47 = axis.map %lhs41, %lhs43, %lhs45 to %rhs42, %rhs44, %rhs46 : [!axis.factor_group<8>, !axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<8>, !axis.factor_group<2>, !axis.factor_group<2>]
  %h48 = distributed.Collective %v26 : tensor<8xf32> on %mesh29 : !axis.factor_group<4> to tensor<4xf32> on %mesh32 : !axis.factor_group<4> reduces () maps %map47 : !axis.map
  %v49 = distributed.Await %h48 : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
}

// -----

// The first collective's result has a second user (the negate), so it must stay materialized: nothing is fused and the remark says why. The remark is on the
// second collective.
// IR-LABEL: module @first_result_has_two_users
// IR: distributed.Collective
// IR: distributed.Collective
// IR-NOT: distributed.Collective
// REMARK: remark: fuse-collectives: not fused with the preceding collective (the first collective's result has other users)
// PLAN: remark: mesh0.0: extent 2 stride 1 in reduced out replicate => all-reduce
// PLAN-NEXT: in0.0: extent 4 pairs out0.0
// PLAN-NEXT: out0.0: extent 4 pairs in0.0
// PLAN-NEXT: reduction: add
// PLAN-NEXT: payload bytes: 16
// PLAN: remark: mesh0.0: extent 2 stride 1 in replicate out tile(in0.0) => local-slice
module @first_result_has_two_users {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>

  %ax1 = axis.getaxis tensor<4xf32> 0
  %ax2 = axis.getaxis tensor<4xf32> 0
  %ax3 = axis.getaxis tensor<2xf32> 0
  %in = tensor.empty() : tensor<4xf32>
  %f4 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh5 = axis.product (%f4 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f6 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh7 = axis.product (%f6 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f8 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %red9 = axis.product (%f8 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f10 = axis.factor %ax1 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %f11 = axis.factor %ax2 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %rep12 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %f13 = axis.factor %rep12 : !distributed.replication_axis<2> <2, 1>
  %f14 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs15 = axis.product (%f10 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %rhs16 = axis.product (%f11 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %lhs17 = axis.product (%f13 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %rhs18 = axis.product (%f14 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map19 = axis.map %lhs15, %lhs17 to %rhs16, %rhs18 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]
  %h20 = distributed.Collective %in : tensor<4xf32> on %mesh5 : !axis.factor_group<2> to tensor<4xf32> on %mesh7 : !axis.factor_group<2> reduces (%red9 : !axis.factor_group<2>) maps %map19 : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }
  %v21 = distributed.Await %h20 : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
  %use = stablehlo.negate %v21 : tensor<4xf32>
  %f22 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh23 = axis.product (%f22 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f24 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh25 = axis.product (%f24 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f26 = axis.factor %ax2 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %f27 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f28 = axis.factor %ax3 : !axis.shape_axis<tensor<2xf32>, 0> <2, 1>
  %f29 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rep30 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %f31 = axis.factor %rep30 : !distributed.replication_axis<2> <2, 1>
  %lhs32 = axis.product (%f26 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %rhs33 = axis.product (%f27 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %f28 : !axis.axis_factor<!axis.shape_axis<tensor<2xf32>, 0>, 2, 1>)
  %lhs34 = axis.product (%f29 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %rhs35 = axis.product (%f31 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %map36 = axis.map %lhs32, %lhs34 to %rhs33, %rhs35 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]
  %h37 = distributed.Collective %v21 : tensor<4xf32> on %mesh23 : !axis.factor_group<2> to tensor<2xf32> on %mesh25 : !axis.factor_group<2> reduces () maps %map36 : !axis.map
  %v38 = distributed.Await %h37 : !distributed.asynch_handle<tensor<2xf32>> -> tensor<2xf32>
}

// -----

// C1 reduces with add and C2 with max, so they cannot share one reduction body: left alone.
// IR-LABEL: module @different_reduction_kinds
// IR: distributed.Collective
// IR: distributed.Collective
// IR-NOT: distributed.Collective
// REMARK: remark: fuse-collectives: not fused with the preceding collective (the reduction bodies are different kinds)
// PLAN: remark: mesh0.0: extent 2 stride 1 in reduced out replicate => all-reduce
// PLAN: remark: mesh0.0: extent 2 stride 1 in mesh(mesh0.0) out mesh(mesh0.0) => no-op
// PLAN-NEXT: mesh1.0: extent 2 stride 1 in reduced out replicate => all-reduce
// PLAN-NEXT: in0.0: extent 4 pairs out0.0
// PLAN-NEXT: out0.0: extent 4 pairs in0.0
// PLAN-NEXT: reduction: max
module @different_reduction_kinds {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>

  %ax1 = axis.getaxis tensor<4xf32> 0
  %ax2 = axis.getaxis tensor<4xf32> 0
  %ax3 = axis.getaxis tensor<4xf32> 0
  %in = tensor.empty() : tensor<4xf32>
  %f4 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f5 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh6 = axis.product (%f4 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f5 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f8 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh9 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f8 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f10 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %red11 = axis.product (%f10 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f12 = axis.factor %ax1 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %f13 = axis.factor %ax2 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %rep14 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %f15 = axis.factor %rep14 : !distributed.replication_axis<2> <2, 1>
  %f16 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f17 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f18 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs19 = axis.product (%f12 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %rhs20 = axis.product (%f13 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %lhs21 = axis.product (%f15 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %rhs22 = axis.product (%f16 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %lhs23 = axis.product (%f17 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %rhs24 = axis.product (%f18 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map25 = axis.map %lhs19, %lhs21, %lhs23 to %rhs20, %rhs22, %rhs24 : [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>]
  %h26 = distributed.Collective %in : tensor<4xf32> on %mesh6 : !axis.factor_group<4> to tensor<4xf32> on %mesh9 : !axis.factor_group<4> reduces (%red11 : !axis.factor_group<2>) maps %map25 : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }
  %v27 = distributed.Await %h26 : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
  %f28 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f29 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh30 = axis.product (%f28 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f29 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f31 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f32 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh33 = axis.product (%f31 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f32 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f34 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %red35 = axis.product (%f34 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f36 = axis.factor %ax2 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %f37 = axis.factor %ax3 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %f38 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f39 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %rep40 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %f41 = axis.factor %rep40 : !distributed.replication_axis<2> <2, 1>
  %f42 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs43 = axis.product (%f36 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %rhs44 = axis.product (%f37 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %lhs45 = axis.product (%f38 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %rhs46 = axis.product (%f39 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %lhs47 = axis.product (%f41 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %rhs48 = axis.product (%f42 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map49 = axis.map %lhs43, %lhs45, %lhs47 to %rhs44, %rhs46, %rhs48 : [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>]
  %h50 = distributed.Collective %v27 : tensor<4xf32> on %mesh30 : !axis.factor_group<4> to tensor<4xf32> on %mesh33 : !axis.factor_group<4> reduces (%red35 : !axis.factor_group<2>) maps %map49 : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %r = stablehlo.maximum %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }
  %v51 = distributed.Await %h50 : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
}

// -----

// Both reduce with a body that is not a recognized kind (subtract is not associative), so fusion is refused even though the kinds are equal.
// IR-LABEL: module @unrecognized_reduction_body
// IR: distributed.Collective
// IR: distributed.Collective
// IR-NOT: distributed.Collective
// REMARK: remark: fuse-collectives: not fused with the preceding collective (the reduction body is not a recognized kind)
// PLAN: remark: mesh0.0: extent 2 stride 1 in reduced out replicate => all-reduce
// PLAN-NEXT: in0.0: extent 4 pairs out0.0
// PLAN-NEXT: out0.0: extent 4 pairs in0.0
// PLAN-NEXT: reduction: unknown
module @unrecognized_reduction_body {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>

  %ax1 = axis.getaxis tensor<4xf32> 0
  %ax2 = axis.getaxis tensor<4xf32> 0
  %ax3 = axis.getaxis tensor<2xf32> 0
  %in = tensor.empty() : tensor<4xf32>
  %f4 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh5 = axis.product (%f4 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f6 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh7 = axis.product (%f6 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f8 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %red9 = axis.product (%f8 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f10 = axis.factor %ax1 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %f11 = axis.factor %ax2 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %rep12 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %f13 = axis.factor %rep12 : !distributed.replication_axis<2> <2, 1>
  %f14 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs15 = axis.product (%f10 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %rhs16 = axis.product (%f11 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %lhs17 = axis.product (%f13 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %rhs18 = axis.product (%f14 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map19 = axis.map %lhs15, %lhs17 to %rhs16, %rhs18 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]
  %h20 = distributed.Collective %in : tensor<4xf32> on %mesh5 : !axis.factor_group<2> to tensor<4xf32> on %mesh7 : !axis.factor_group<2> reduces (%red9 : !axis.factor_group<2>) maps %map19 : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %r = stablehlo.subtract %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }
  %v21 = distributed.Await %h20 : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
  %f22 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh23 = axis.product (%f22 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f24 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh25 = axis.product (%f24 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f26 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %red27 = axis.product (%f26 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f28 = axis.factor %ax2 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %f29 = axis.factor %ax2 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %rep30 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %f31 = axis.factor %rep30 : !distributed.replication_axis<2> <2, 1>
  %f32 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs33 = axis.product (%f28 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %rhs34 = axis.product (%f29 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %lhs35 = axis.product (%f31 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %rhs36 = axis.product (%f32 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map37 = axis.map %lhs33, %lhs35 to %rhs34, %rhs36 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]
  %h38 = distributed.Collective %v21 : tensor<4xf32> on %mesh23 : !axis.factor_group<2> to tensor<4xf32> on %mesh25 : !axis.factor_group<2> reduces (%red27 : !axis.factor_group<2>) maps %map37 : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %r = stablehlo.subtract %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }
  %v39 = distributed.Await %h38 : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
}
