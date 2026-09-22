// RUN: enzymexlamlir-opt --distributed-atomize-collectives --distributed-inline-kernels --distributed-lower-collectives --distributed-print-schedule-graph %s 2>&1 >/dev/null | FileCheck %s

// Exercises ScheduleGraph.cpp's buildBlockNodes on the two IR shapes that
// only appear once a module has gone through distributed-inline-kernels (D4,
// which dissolves a DistributedKernel wrapper by splicing its body directly
// into the surrounding block) and distributed-lower-collectives (D5, which
// replaces an atomic distributed.Collective/Await pair with a real
// stablehlo.async_start/async_done chain tagged with a
// distributed.primitive_cost attribute): a bare stablehlo op sitting
// directly in a block, and a cost-tagged async_start/async_done pair.
//
// This module exercises both shapes at once: distributed-inline-kernels
// dissolves the kernel below into two bare stablehlo.add ops sitting
// directly in the module body, and distributed-lower-collectives replaces
// the all-reduce collective consuming the kernel's result with one
// async_start/async_done pair. The module-scope tensor.empty feeding the
// kernel is deliberately not stablehlo dialect (see ScheduleGraph.cpp's bare
// op case): it is scaffolding to give the kernel an operand, matching the
// convention elsewhere in this directory (e.g. schedule_cost.mlir), and must
// stay uncosted, the same as the axis-algebra metadata ops around it.
//
// The two adds are each 4 f32 elements: flops=4, bytes=48 (two operand reads
// plus one result write, 16 bytes each), time=max(4/2000, 48/20)=2.4 with
// the default device rates, matching schedule_graph.mlir's own worked
// example. The all-reduce moves a 16-byte payload over the default mesh
// bandwidth/latency; its own isolated duration is read directly off the
// distributed.primitive_cost attribute distributed-lower-collectives
// stamped on the async_start, rather than recomputed here.
// CHECK: node 0: compute stablehlo.add: flops=4 bytes=48 time=2.4 preds=[]
// CHECK: node 1: compute stablehlo.add: flops=4 bytes=48 time=2.4 preds=[0]
// CHECK: node 2: communication all-reduce (step 0): time={{.*}} preds=[1]
// CHECK: schedule graph: 3 nodes (2 compute, 1 communication, 0 call)
module @lowered_collective_schedule {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<4xf32> 0
  %to0 = axis.getaxis tensor<4xf32> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %red0 = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f4 = axis.factor %ti0 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %lhs0 = axis.product (%f4 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %f5 = axis.factor %to0 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %rhs0 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %f6 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %lhs1 = axis.product (%f6 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs1 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]

  %a = tensor.empty() : tensor<4xf32>
  %k1 = distributed.DistributedKernel (%a : tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    -> (tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    axes () {
  ^bb0(%x: tensor<4xf32>):
    %d = stablehlo.add %x, %x : tensor<4xf32>
    %e = stablehlo.add %d, %d : tensor<4xf32>
    distributed.DistributedYield (%e : tensor<4xf32>)
  }

  %h = distributed.Collective %k1 : tensor<4xf32> on %mesh_in : !axis.factor_group<2> to tensor<4xf32> on %mesh_out : !axis.factor_group<2> reduces (%red0 : !axis.factor_group<2>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
}
