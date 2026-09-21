// RUN: enzymexlamlir-opt --split-input-file --distributed-print-schedule-graph %s 2>&1 >/dev/null | FileCheck %s

// distributed-print-schedule-graph reports the event DAG (ScheduleGraph.h):
// one remark per node with its index, kind, cost and predecessor node
// indices, in module build order.
//
// Each stablehlo.add/multiply here is 4 f32 elements: flops = 4 (arithmetic
// kind, ComputeCost.h), bytes = 3 * 16 = 48 (two operand reads plus one
// result write, 16 bytes each), time = max(4/2000, 48/20) = 2.4 with the
// default device rates (no mesh performance metadata given).

// A linear SSA dependency chain inside one kernel body produces a chain of
// compute nodes, each depending on the one before it; the first depends on
// nothing since its operand is the kernel's own block argument, not a node
// in the graph.
// CHECK: node 0: compute stablehlo.add: flops=4 bytes=48 time=2.4 preds=[]
// CHECK: node 1: compute stablehlo.add: flops=4 bytes=48 time=2.4 preds=[0]
// CHECK: schedule graph: 2 nodes (2 compute, 0 communication, 0 call)
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main(%a: tensor<4xf32>) -> tensor<4xf32> {
    %k = distributed.DistributedKernel (%a : tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
    ^bb0(%x: tensor<4xf32>):
      %d = stablehlo.add %x, %x : tensor<4xf32>
      %e = stablehlo.add %d, %d : tensor<4xf32>
      distributed.DistributedYield (%e : tensor<4xf32>)
    }
    return %k : tensor<4xf32>
  }
}

// -----

// Two ops that read the same kernel block argument but not each other's
// results are independent: neither node's predecessor list names the other,
// since edges are plain SSA def-use (ScheduleGraph.h) and there is none
// between them.
// CHECK: node 0: compute stablehlo.add: flops=4 bytes=48 time=2.4 preds=[]
// CHECK: node 1: compute stablehlo.multiply: flops=4 bytes=48 time=2.4 preds=[]
// CHECK: schedule graph: 2 nodes (2 compute, 0 communication, 0 call)
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main(%a: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
    %k:2 = distributed.DistributedKernel (%a : tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<4xf32>, tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>, <dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
    ^bb0(%x: tensor<4xf32>):
      %d = stablehlo.add %x, %x : tensor<4xf32>
      %e = stablehlo.multiply %x, %x : tensor<4xf32>
      distributed.DistributedYield (%d : tensor<4xf32>, %e : tensor<4xf32>)
    }
    return %k#0, %k#1 : tensor<4xf32>, tensor<4xf32>
  }
}

// -----

// A kernel's own operand/result list is at a different SSA identity than its
// body's block arguments/yielded values (ScheduleGraph.h aliases across this
// boundary instead of rewriting anything): the second kernel's compute node
// still depends on the first kernel's compute node, even though the edge
// between them is `kernel1 result -> kernel2 operand`, not a direct op-to-op
// SSA use.
// CHECK: node 0: compute stablehlo.add: flops=4 bytes=48 time=2.4 preds=[]
// CHECK: node 1: compute stablehlo.add: flops=4 bytes=48 time=2.4 preds=[0]
// CHECK: schedule graph: 2 nodes (2 compute, 0 communication, 0 call)
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main(%a: tensor<4xf32>) -> tensor<4xf32> {
    %k1 = distributed.DistributedKernel (%a : tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
    ^bb0(%x: tensor<4xf32>):
      %d = stablehlo.add %x, %x : tensor<4xf32>
      distributed.DistributedYield (%d : tensor<4xf32>)
    }
    %k2 = distributed.DistributedKernel (%k1 : tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
    ^bb0(%y: tensor<4xf32>):
      %e = stablehlo.add %y, %y : tensor<4xf32>
      distributed.DistributedYield (%e : tensor<4xf32>)
    }
    return %k2 : tensor<4xf32>
  }
}

// -----

// An atomic distributed.Collective between two kernels decomposes (extent-2
// reduced -> replicate is a one-step all-reduce, CollectiveDecomposer.h) into
// one communication node whose chain bridges the kernel-boundary values: it
// depends on the producing kernel's own compute node (through the
// collective's input_object operand), and the consuming kernel's compute node
// depends on it in turn (through the await result).
//
// Axis-algebra metadata ops (distributed.GetPhysicalMeshAxes, axis.factor,
// axis.product, axis.map) must sit directly in the module body rather than
// inside a function, matching the atomic-collective examples elsewhere in
// this directory (e.g. print_collective_plan.mlir); the kernels, collective
// and await are placed alongside them rather than inside func.func @main.
// CHECK: node 0: compute stablehlo.add: flops=4 bytes=48 time=2.4 preds=[]
// CHECK: node 1: communication all-reduce (step 0): {{.*}} preds=[0]
// CHECK: node 2: compute stablehlo.add: flops=4 bytes=48 time=2.4 preds=[1]
// CHECK: schedule graph: 3 nodes (2 compute, 1 communication, 0 call)
module {
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
    %s = stablehlo.add %x, %x : tensor<4xf32>
    distributed.DistributedYield (%s : tensor<4xf32>)
  }

  %h = distributed.Collective %k1 : tensor<4xf32> on %mesh_in : !axis.factor_group<2> to tensor<4xf32> on %mesh_out : !axis.factor_group<2> reduces (%red0 : !axis.factor_group<2>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>

  %k2 = distributed.DistributedKernel (%v : tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    -> (tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    axes () {
  ^bb0(%y: tensor<4xf32>):
    %t = stablehlo.add %y, %y : tensor<4xf32>
    distributed.DistributedYield (%t : tensor<4xf32>)
  }
}

// -----

// A distributed.DistributedCall is a hard scheduling barrier: the kernel
// before the call (node 0) is a predecessor of the call event (node 1), the
// kernel after the call (node 2) depends on the call event rather than on
// anything before it, and the callee's own two independent ops (from its own
// DistributedKernel, see the multi-function IR shape in
// call_distributed_functions.mlir) get their own nested "callee node" events
// with no predecessors of their own. The callee's internal parallelism is
// preserved rather than serialized through the barrier, since only the
// caller-side ordering is a hard barrier, not the callee's internal one. The
// call's own isolatedDuration (4.8 = 2.4 + 2.4) is only a proxy: a serial sum
// of the callee's own node durations, not a real simulated makespan.
// CHECK: node 0: compute stablehlo.add: flops=4 bytes=48 time=2.4 preds=[]
// CHECK: node 1: call distributed.DistributedCall: callee_nodes=2 time=4.8 (proxy: serial sum of callee node durations, pending D2 timeline simulator) preds=[0]
// CHECK: callee node 0: compute stablehlo.add: flops=4 bytes=48 time=2.4 preds=[]
// CHECK: callee node 1: compute stablehlo.multiply: flops=4 bytes=48 time=2.4 preds=[]
// CHECK: node 2: compute stablehlo.add: flops=4 bytes=48 time=2.4 preds=[1]
// CHECK: schedule graph: 5 nodes (4 compute, 0 communication, 1 call)
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %a = tensor.empty() : tensor<4xf32>
  %k1 = distributed.DistributedKernel (%a : tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    -> (tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    axes () {
  ^bb0(%x: tensor<4xf32>):
    %s = stablehlo.add %x, %x : tensor<4xf32>
    distributed.DistributedYield (%s : tensor<4xf32>)
  }

  %c = distributed.DistributedCall @callee (%k1 : tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    -> (tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    axes ()

  %k2 = distributed.DistributedKernel (%c : tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    -> (tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    axes () {
  ^bb0(%y: tensor<4xf32>):
    %t = stablehlo.add %y, %y : tensor<4xf32>
    distributed.DistributedYield (%t : tensor<4xf32>)
  }

  "distributed.DistributedFunction"() <{
    argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>,
    function_type = (tensor<4xf32>) -> tensor<4xf32>,
    output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>,
    sym_name = "callee"
  }> ({
  ^bb0(%z: tensor<4xf32>):
    %m = distributed.DistributedKernel (%z : tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
    ^bb0(%w: tensor<4xf32>):
      %u1 = stablehlo.add %w, %w : tensor<4xf32>
      %u2 = stablehlo.multiply %w, %w : tensor<4xf32>
      distributed.DistributedYield (%u1 : tensor<4xf32>)
    }
    distributed.DistributedYield (%m : tensor<4xf32>)
  }) : () -> ()
}
