// RUN: rm -rf %t.dir && mkdir -p %t.dir
//
// --set-distributed-logging stamps distributed.logging_root as it would ahead
// of a real search, but nothing here runs cloneAndApplyDecisions (the only
// place distributed.logging_candidate_uid is stamped). This reproduces the
// winner-finalization run: distributed-search-strategies binds its winning
// decisions onto the original module and hands that module to
// distributed-search-lowering-pipeline without going through
// cloneAndApplyDecisions (see DumpDistributedLoggingCheckpointPass's
// description). Every checkpoint-dump instance in the pipeline must see
// logging_root present and logging_candidate_uid absent, and do nothing: no
// crash and no stray log entry.
// RUN: enzymexlamlir-opt %s --set-distributed-logging="root=%t.dir module-name=winner" --distributed-search-lowering-pipeline -o %t.out.mlir
// RUN: FileCheck %s < %t.out.mlir
// RUN: find %t.dir/winner-*/Candidates -mindepth 1 | wc -l | FileCheck %s --check-prefix=NOCANDIDATES

// NOCANDIDATES: 0

// The rest of the pipeline still runs to completion (same structural anchors
// as search_lowering_pipeline.mlir's FINAL check).
// CHECK: distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 2>, !distributed.physical_comm_axis<2, 1>]
// CHECK-NOT: distributed.DistributedKernel
// CHECK-NOT: distributed.Collective
// CHECK: stablehlo.multiply %{{.*}}, %{{.*}} : tensor<8xf32>
// CHECK: stablehlo.all_reduce

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
