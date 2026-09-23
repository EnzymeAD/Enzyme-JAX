// RUN: enzymexlamlir-opt %s --distributed-search-lowering-pipeline="finalize-schedule=false" | FileCheck %s --check-prefix=NOFINAL
// RUN: enzymexlamlir-opt %s --distributed-search-lowering-pipeline="finalize-schedule=false" 2>&1 >/dev/null | FileCheck %s --check-prefix=REMARK
// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: enzymexlamlir-opt %s --distributed-search-lowering-pipeline="dump-kernel-modules-to=%t.dir finalize-schedule=false" -o /dev/null 2>/dev/null
// RUN: ls %t.dir | wc -l | FileCheck %s --check-prefix=DUMPCOUNT
// DUMPCOUNT: 1
// RUN: enzymexlamlir-opt %s --distributed-search-lowering-pipeline | FileCheck %s --check-prefix=FINAL

// End-to-end smoke test for buildDistributedSearchLoweringPipeline
// (--distributed-search-lowering-pipeline), which otherwise has no lit
// coverage: a genuine physical all-reduce (mesh0's upper sub-factor of
// physical_comm_axis<4,2> is reduced then replicated back across it, the
// "reduced -> replicate" row of the atomic-form design) sitting downstream
// of a real DistributedKernel, run through all sixteen non-finalization
// passes in the pipeline. The kernel is intentionally unsharded (an
// embarrassingly parallel elementwise op, dim_partitioning_axes = [[]]) so
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
// The full pipeline's IR is not pinned line by line (SSA numbering and
// structural churn at every stage make that brittle to the point of
// meaninglessness, per this repo's convention for pipeline-level tests);
// instead this checks structural anchors.
//
// distributed-inline-kernels and distributed-lower-collectives always run
// regardless of finalize-schedule: every candidate reaching schedule-graph-
// building needs the same dissolved-kernel, lowered-collective IR shape that
// a ScheduleGraphAnalysis costs from, scoring clone or eventual winner
// alike. So the kernel body's multiply is spliced directly into the module,
// and the collective (never an identity, since mesh_in's reduced atom
// becomes a fresh replicate on the output side rather than the same factor
// back, so distributed-drop-identity-collectives must leave it in place)
// becomes a real async-wrapped stablehlo.all_reduce; neither NOFINAL nor
// FINAL sees a surviving distributed.DistributedKernel or
// distributed.Collective.
//
// Only distributed-reorder-schedule stays behind finalize-schedule. This
// module's ops already sit in simulated admission order (one straight-line
// dependency chain, nothing to overlap or reorder), so NOFINAL and FINAL
// check identical structural anchors here; the two RUN lines still each
// guard that lowering succeeds whether or not that last physical-reordering
// step runs.
//
// The REMARK lines are atomize's own atom-structure report, re-run against
// the rewritten (not original) collective, checked once against the
// finalize-schedule=false invocation. That report is a cheap, stable-ish
// signal that atomize actually ran and succeeded on this collective,
// following the same REMARK convention as atomize_collectives.mlir.
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

// NOFINAL: distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 2>, !distributed.physical_comm_axis<2, 1>]
// NOFINAL-NOT: distributed.DistributedKernel
// NOFINAL-NOT: distributed.Collective
// NOFINAL: stablehlo.multiply %{{.*}}, %{{.*}} : tensor<8xf32>
// NOFINAL-NOT: distributed.argument_shardings
// NOFINAL-NOT: distributed.output_shardings
// NOFINAL: stablehlo.async_start
// NOFINAL: stablehlo.all_reduce
// NOFINAL: stablehlo.async_done

// FINAL: distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 2>, !distributed.physical_comm_axis<2, 1>]
// FINAL-NOT: distributed.DistributedKernel
// FINAL-NOT: distributed.Collective
// FINAL: stablehlo.multiply %{{.*}}, %{{.*}} : tensor<8xf32>
// FINAL: stablehlo.async_start
// FINAL: stablehlo.all_reduce
// FINAL: stablehlo.async_done

// REMARK: remark: mesh0: 2, 2
// REMARK-NEXT: mesh1: 2
// REMARK-NEXT: in0: 8
// REMARK-NEXT: out0: 8
// REMARK-NEXT: reduction0: mesh0.0
// REMARK-NEXT: pair0: replicate0.0 -> mesh0.0
// REMARK-NEXT: pair1: mesh0.1 -> mesh0.1
// REMARK-NEXT: pair2: mesh1.0 -> mesh1.0
// REMARK-NEXT: pair3: in0.0 -> out0.0
