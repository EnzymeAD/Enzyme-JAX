// RUN: enzymexlamlir-opt %s --distributed-search-lowering-pipeline | FileCheck %s
// RUN: enzymexlamlir-opt %s --distributed-search-lowering-pipeline 2>&1 >/dev/null | FileCheck %s --check-prefix=REMARK
// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: enzymexlamlir-opt %s --distributed-search-lowering-pipeline="dump-kernel-modules-to=%t.dir" -o /dev/null 2>/dev/null
// RUN: ls %t.dir | wc -l | FileCheck %s --check-prefix=DUMPCOUNT
// DUMPCOUNT: 1

// End-to-end smoke test for buildDistributedSearchLoweringPipeline
// (--distributed-search-lowering-pipeline), which otherwise has no lit
// coverage: a genuine physical all-reduce (mesh0's upper sub-factor of
// physical_comm_axis<4,2> is reduced then replicated back across it, the
// "reduced -> replicate" row of the atomic-form design) sitting downstream
// of a real DistributedKernel, run through all sixteen passes in the
// pipeline. The kernel is intentionally unsharded (an embarrassingly
// parallel elementwise op, dim_partitioning_axes = [[]]) so
// RefinePartitioningSlots/CanonicalizeShardedFactorOrder/
// InlineDeviceLocalAxes -- whose own preconditions are about how
// DeviceLocalAxis slots are laid out -- have nothing to do here; there is no
// DeviceLocalAxis in this module at all, only genuine physical mesh axes,
// so those three passes and the DistributedKernel-shrinking passes around
// distributed-lower-kernels are exercised as pure pass-throughs and the
// interesting rewriting happens in distributed-make-replications-explicit
// and distributed-atomize-collectives right before the final
// distributed-lower-kernels-to-executable.
//
// The full 16-pass IR is not pinned line by line (SSA numbering and
// structural churn at every stage make that brittle to the point of
// meaninglessness, per this repo's convention for pipeline-level tests);
// instead this checks structural anchors: the kernel and collective both
// survive to the end (this collective is never an identity, so
// DropIdentityCollectives must not remove it), the kernel body's
// sharding attributes are gone (distributed-lower-kernels is their last
// consumer), and the REMARK lines are atomize's own atom-structure report,
// re-run against the rewritten (not original) collective -- a cheap,
// stable-ish signal that atomize actually ran and succeeded on this
// collective, following the same REMARK convention as atomize_collectives.mlir.
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 2>, !distributed.physical_comm_axis<2, 1>]

  func.func @main() { return }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 2>, !distributed.physical_comm_axis<2, 1>

  %lf0_upper = axis.factor %p0 : !distributed.physical_comm_axis<4, 2> <2, 2>
  %lf0_lower = axis.factor %p0 : !distributed.physical_comm_axis<4, 2> <2, 1>
  %lf1 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>

  %ta = axis.getaxis tensor<8xf32> 0
  %tf0 = axis.factor %ta : !axis.shape_axis<tensor<8xf32>, 0> <8, 1>

  %r0 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %rf0 = axis.factor %r0 : !distributed.replication_axis<2> <2, 1>

  %mesh_in = axis.product (%lf0_upper : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 2, 2>, %lf0_lower : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 2, 1>, %lf1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %mesh_out = axis.product (%lf0_upper : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 2, 2>, %lf0_lower : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 2, 1>, %lf1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %reduction = axis.product (%lf0_upper : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 2, 2>)

  %lhs_group_0 = axis.product (%rf0 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %rhs_group_0 = axis.product (%lf0_upper : !axis.axis_factor<!distributed.physical_comm_axis<4,2>, 2, 2>)

  %lhs_group_1 = axis.product (%lf0_lower : !axis.axis_factor<!distributed.physical_comm_axis<4,2>, 2, 1>)
  %rhs_group_1 = axis.product (%lf0_lower : !axis.axis_factor<!distributed.physical_comm_axis<4,2>, 2, 1>)

  %lhs_group_2 = axis.product (%lf1 : !axis.axis_factor<!distributed.physical_comm_axis<2,1>, 2, 1>)
  %rhs_group_2 = axis.product (%lf1 : !axis.axis_factor<!distributed.physical_comm_axis<2,1>, 2, 1>)

  %lhs_group_3 = axis.product (%tf0 : !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 8, 1>)
  %rhs_group_3 = axis.product (%tf0 : !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 8, 1>)

  %mapping = axis.map %lhs_group_0, %lhs_group_1, %lhs_group_2, %lhs_group_3 to %rhs_group_0, %rhs_group_1, %rhs_group_2, %rhs_group_3 : [!axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<8>] [!axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<8>]

  %input = stablehlo.constant dense<1.0> : tensor<8xf32>
  %sum = distributed.DistributedKernel (%input : tensor<8xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<8xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
  ^bb0(%a: tensor<8xf32>):
    %c = stablehlo.multiply %a, %a {distributed.argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>, distributed.output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>} : tensor<8xf32>
    distributed.DistributedYield (%c : tensor<8xf32>)
  }

  %h = distributed.Collective %sum : tensor<8xf32> on %mesh_in : !axis.factor_group<8> to tensor<8xf32> on %mesh_out : !axis.factor_group<8> reduces (%reduction : !axis.factor_group<2>) maps %mapping : !axis.map {
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %s = stablehlo.add %lhs, %rhs : tensor<f32>
    stablehlo.return %s : tensor<f32>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<8xf32>> -> tensor<8xf32>
}

// CHECK: distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 2>, !distributed.physical_comm_axis<2, 1>]
// The kernel survives to the end unmerged (nothing else to merge it with)
// and its body's sharding attributes are gone: distributed-lower-kernels is
// their last consumer, and drop-kernel-body-sharding-attrs/
// drop-sharding-rule-attrs run right after it in this pipeline.
// CHECK: %[[KERNEL:[0-9]+]] = distributed.DistributedKernel
// CHECK: stablehlo.multiply %{{.*}}, %{{.*}} : tensor<8xf32>
// CHECK-NOT: distributed.argument_shardings
// CHECK-NOT: distributed.output_shardings
// The collective is never an identity (mesh_in's reduced atom becomes a
// fresh replicate on the output side, not the same factor back), so
// distributed-drop-identity-collectives must leave it in place; atomize
// then rebuilds its mesh/reduction/map operands into atomic form.
// CHECK: distributed.Collective %[[KERNEL]] : tensor<8xf32> on %{{[0-9]+}} : <8> to tensor<8xf32> on %{{[0-9]+}} : <8> reduces (%{{[0-9]+}} : !axis.factor_group<2>) maps %{{[0-9]+}} : !axis.map
// CHECK: distributed.Await %{{[0-9]+}} : <tensor<8xf32>> -> tensor<8xf32>

// REMARK: remark: mesh0: 2, 2
// REMARK-NEXT: mesh1: 2
// REMARK-NEXT: in0: 8
// REMARK-NEXT: out0: 8
// REMARK-NEXT: reduction0: mesh0.0
// REMARK-NEXT: pair0: replicate0.0 -> mesh0.0
// REMARK-NEXT: pair1: mesh0.1 -> mesh0.1
// REMARK-NEXT: pair2: mesh1.0 -> mesh1.0
// REMARK-NEXT: pair3: in0.0 -> out0.0
