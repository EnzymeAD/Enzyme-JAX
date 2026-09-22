// RUN: enzymexlamlir-opt --split-input-file --distributed-print-schedule-cost %s 2>&1 >/dev/null | FileCheck %s

// distributed-print-schedule-cost runs the event DAG builder (ScheduleGraph.h)
// followed by the discrete-event, strict-priority-dispatch timeline simulator
// (ScheduleSimulator.h): one remark per node with its simulated start/finish
// time, and a module summary remark with the predicted makespan and the
// DAG's own (contention-free) critical path.
//
// Every mesh below sets axis_bandwidths/axis_latencies/launch_latency
// explicitly so the numbers below are exact rather than depending on
// MeshCostParams's abstract defaults. Compute nodes use the default device
// rates (DeviceCostParams::kDefaultFlops = 2000, kDefaultMemBandwidth = 20):
// a stablehlo.add of 4 f32 elements is flops=4, bytes=48 (two operand reads
// plus one result write, 16 bytes each), time = max(4/2000, 48/20) = 2.4,
// matching schedule_graph.mlir's own worked example.

// A linear dependency chain inside one kernel body must simulate as the
// serial sum of each op's isolated duration. It also pins down
// ScheduleGraph.h's predecessorMultiplicity: %e and %f each reference their
// single producer through two operand slots (`add %d, %d`), so their
// readiness counters are initialized to 2 and must be decremented twice, not
// once, to become ready. Building the counters from the deduplicated
// predecessor list instead of per-operand multiplicity would leave node 1
// and node 2 permanently blocked and deadlock the simulation.
// CHECK: node 0: compute start=0 finish=2.4
// CHECK: node 1: compute start=2.4 finish=4.8
// CHECK: node 2: compute start=4.8 finish=7.2
// CHECK: schedule cost: makespan=7.2 over 3 nodes, critical path=[0, 1, 2]
// CHECK: schedule cost summary: communicating=0 computing=7.2 utilization=1 idle=0
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main(%a: tensor<4xf32>) -> tensor<4xf32> {
    %k = distributed.DistributedKernel (%a : tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
    ^bb0(%x: tensor<4xf32>):
      %d = stablehlo.add %x, %x : tensor<4xf32>
      %e = stablehlo.add %d, %d : tensor<4xf32>
      %f = stablehlo.add %e, %e : tensor<4xf32>
      distributed.DistributedYield (%f : tensor<4xf32>)
    }
    return %k : tensor<4xf32>
  }
}

// -----

// Two independent all-reduces on different physical mesh axes overlap: each
// axis has bandwidth 1 and the 4xf32 payload (16 bytes) gives an isolated
// duration of 16 on either axis, but since they contend for different
// resources the makespan is max(16, 16) = 16, not their sum.
// CHECK: node 0: communication start=0 finish=16
// CHECK: node 1: communication start=0 finish=16
// CHECK: schedule cost: makespan=16 over 2 nodes, critical path=[0]
// CHECK: schedule cost summary: communicating=16 computing=0 utilization=0 idle=16
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>] {axis_bandwidths = [1.0e0, 1.0e0], axis_latencies = [0.0e0, 0.0e0], launch_latency = 0.0e0}

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %rep2b = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<4xf32> 0
  %to0 = axis.getaxis tensor<4xf32> 0
  %ti1 = axis.getaxis tensor<4xf32> 0
  %to1 = axis.getaxis tensor<4xf32> 0

  // Collective A on axis 0.
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %mesh_in0 = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %mesh_out0 = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %red0 = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f4 = axis.factor %ti0 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %lhs0 = axis.product (%f4 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %f5 = axis.factor %to0 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %rhs0 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %f6 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %lhs1 = axis.product (%f6 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %rhs1 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %map0 = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]

  %a0 = tensor.empty() : tensor<4xf32>
  %h0 = distributed.Collective %a0 : tensor<4xf32> on %mesh_in0 : !axis.factor_group<2> to tensor<4xf32> on %mesh_out0 : !axis.factor_group<2> reduces (%red0 : !axis.factor_group<2>) maps %map0 : !axis.map {
  ^bb0(%lhs_v0: tensor<f32>, %rhs_v0: tensor<f32>):
    %r0 = stablehlo.add %lhs_v0, %rhs_v0 : tensor<f32>
    stablehlo.return %r0 : tensor<f32>
  }
  %v0 = distributed.Await %h0 : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>

  // Collective B on axis 1, independent of A.
  %g1 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in1 = axis.product (%g1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %g2 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out1 = axis.product (%g2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %g3 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %red1 = axis.product (%g3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %g4 = axis.factor %ti1 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %lhs2 = axis.product (%g4 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %g5 = axis.factor %to1 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %rhs2 = axis.product (%g5 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %g6 = axis.factor %rep2b : !distributed.replication_axis<2> <2, 1>
  %lhs3 = axis.product (%g6 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %g7 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs3 = axis.product (%g7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map1 = axis.map %lhs2, %lhs3 to %rhs2, %rhs3 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]

  %a1 = tensor.empty() : tensor<4xf32>
  %h1 = distributed.Collective %a1 : tensor<4xf32> on %mesh_in1 : !axis.factor_group<2> to tensor<4xf32> on %mesh_out1 : !axis.factor_group<2> reduces (%red1 : !axis.factor_group<2>) maps %map1 : !axis.map {
  ^bb0(%lhs_v1: tensor<f32>, %rhs_v1: tensor<f32>):
    %r1 = stablehlo.add %lhs_v1, %rhs_v1 : tensor<f32>
    stablehlo.return %r1 : tensor<f32>
  }
  %v1 = distributed.Await %h1 : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
}

// -----

// Two independent all-reduces on the SAME axis must contend for its
// bandwidth: node 0, ranked first, dispatches at full rate and finishes at
// 16; node 1 gets scale 0 while node 0 holds the axis, then dispatches at
// full rate once node 0 finishes, so it finishes at 16 + 16 = 32.
// Conservation of work on one shared resource, the same reason two
// independent compute ops on the one shared compute resource (case 4 below)
// would also sum rather than overlap.
// CHECK: node 0: communication start=0 finish=16
// CHECK: node 1: communication start=0 finish=32
// CHECK: schedule cost: makespan=32 over 2 nodes, critical path=[0]
// CHECK: schedule cost summary: communicating=32 computing=0 utilization=0 idle=32
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>] {axis_bandwidths = [1.0e0], axis_latencies = [0.0e0], launch_latency = 0.0e0}

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %rep2b = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<4xf32> 0
  %to0 = axis.getaxis tensor<4xf32> 0
  %ti1 = axis.getaxis tensor<4xf32> 0
  %to1 = axis.getaxis tensor<4xf32> 0

  // Collective A on axis 0.
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in0 = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out0 = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
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
  %map0 = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]

  %a0 = tensor.empty() : tensor<4xf32>
  %h0 = distributed.Collective %a0 : tensor<4xf32> on %mesh_in0 : !axis.factor_group<2> to tensor<4xf32> on %mesh_out0 : !axis.factor_group<2> reduces (%red0 : !axis.factor_group<2>) maps %map0 : !axis.map {
  ^bb0(%lhs_v0: tensor<f32>, %rhs_v0: tensor<f32>):
    %r0 = stablehlo.add %lhs_v0, %rhs_v0 : tensor<f32>
    stablehlo.return %r0 : tensor<f32>
  }
  %v0 = distributed.Await %h0 : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>

  // Collective B, also on axis 0, independent of A.
  %g1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in1 = axis.product (%g1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %g2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out1 = axis.product (%g2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %g3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %red1 = axis.product (%g3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %g4 = axis.factor %ti1 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %lhs2 = axis.product (%g4 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %g5 = axis.factor %to1 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %rhs2 = axis.product (%g5 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %g6 = axis.factor %rep2b : !distributed.replication_axis<2> <2, 1>
  %lhs3 = axis.product (%g6 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %g7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs3 = axis.product (%g7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map1 = axis.map %lhs2, %lhs3 to %rhs2, %rhs3 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]

  %a1 = tensor.empty() : tensor<4xf32>
  %h1 = distributed.Collective %a1 : tensor<4xf32> on %mesh_in1 : !axis.factor_group<2> to tensor<4xf32> on %mesh_out1 : !axis.factor_group<2> reduces (%red1 : !axis.factor_group<2>) maps %map1 : !axis.map {
  ^bb0(%lhs_v1: tensor<f32>, %rhs_v1: tensor<f32>):
    %r1 = stablehlo.add %lhs_v1, %rhs_v1 : tensor<f32>
    stablehlo.return %r1 : tensor<f32>
  }
  %v1 = distributed.Await %h1 : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
}

// -----

// An independent compute kernel and collective use disjoint resources
// (compute/memBW vs. the one mesh axis) and fully overlap: makespan is
// max(2.4, 16) = 16, not their sum.
// CHECK: node 0: compute start=0 finish=2.4
// CHECK: node 1: communication start=0 finish=16
// CHECK: schedule cost: makespan=16 over 2 nodes, critical path=[1]
// CHECK: schedule cost summary: communicating=16 computing=2.4 utilization=0.15 idle=13.6
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>] {axis_bandwidths = [1.0e0], axis_latencies = [0.0e0], launch_latency = 0.0e0}

  func.func @main(%a: tensor<4xf32>) -> tensor<4xf32> {
    %k = distributed.DistributedKernel (%a : tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
    ^bb0(%x: tensor<4xf32>):
      %d = stablehlo.add %x, %x : tensor<4xf32>
      distributed.DistributedYield (%d : tensor<4xf32>)
    }
    return %k : tensor<4xf32>
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<4xf32> 0
  %to0 = axis.getaxis tensor<4xf32> 0

  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in0 = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out0 = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
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
  %map0 = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]

  %a0 = tensor.empty() : tensor<4xf32>
  %h0 = distributed.Collective %a0 : tensor<4xf32> on %mesh_in0 : !axis.factor_group<2> to tensor<4xf32> on %mesh_out0 : !axis.factor_group<2> reduces (%red0 : !axis.factor_group<2>) maps %map0 : !axis.map {
  ^bb0(%lhs_v0: tensor<f32>, %rhs_v0: tensor<f32>):
    %r0 = stablehlo.add %lhs_v0, %rhs_v0 : tensor<f32>
    stablehlo.return %r0 : tensor<f32>
  }
  %v0 = distributed.Await %h0 : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
}

// -----

// Two independent compute ops (same kernel body, both reading the block
// argument, neither depending on the other) must serialize rather than
// fair-share the compute/memBW resources: the compute engine can only
// dispatch one kernel at a time, so the makespan is the SUM of their
// isolated durations (2.4 + 2.4 = 4.8), not max(2.4, 2.4). This exercises
// mutual exclusion on kComputeResource with two concurrently-ready Compute
// events.
// CHECK: node 0: compute start=0 finish=2.4
// CHECK: node 1: compute start=2.4 finish=4.8
// CHECK: schedule cost: makespan=4.8 over 2 nodes, critical path=[0]
// CHECK: schedule cost summary: communicating=0 computing=4.8 utilization=1 idle=0
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

// Same mutual-exclusion requirement, but with heterogeneous demand: a
// compute-bound dot_general (2x2 . 2x2, flops=16, bytes=48) and a
// memory-bound add (flops=4, bytes=48), under a mesh whose device rates
// (device_flops=1, device_mem_bandwidth=5) are chosen so the two ops are
// bottlenecked on different resources. dot_general's isolated duration is
// max(16/1, 48/5) = 16 (compute-bound), the add's is max(4/1, 48/5) = 9.6
// (memory-bound). Serialization must still hold regardless of which
// resource each op is individually bottlenecked on: makespan is their SUM,
// 16 + 9.6 = 25.6, not max(16, 9.6). A memory-bound kernel occupying the
// device still fully blocks a compute-bound kernel from running
// concurrently, even though neither op saturates the resource the other is
// bottlenecked on.
// CHECK: node 0: compute start=0 finish=16
// CHECK: node 1: compute start=16 finish=25.6
// CHECK: schedule cost: makespan=25.6 over 2 nodes, critical path=[0]
// CHECK: schedule cost summary: communicating=0 computing=25.6 utilization=1 idle=0
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>] {axis_bandwidths = [1.0e0], axis_latencies = [0.0e0], launch_latency = 0.0e0, device_flops = 1.0e0, device_mem_bandwidth = 5.0e0}

  func.func @main(%a: tensor<2x2xf32>, %b: tensor<4xf32>) -> (tensor<2x2xf32>, tensor<4xf32>) {
    %k:2 = distributed.DistributedKernel (%a : tensor<2x2xf32>, %b : tensor<4xf32>) <[<dim_partitioning_axes = [[], []] : unreduced_axes = []>, <dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<2x2xf32>, tensor<4xf32>) <[<dim_partitioning_axes = [[], []] : unreduced_axes = []>, <dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
    ^bb0(%x: tensor<2x2xf32>, %y: tensor<4xf32>):
      %d = "stablehlo.dot_general"(%x, %x) {
        dot_dimension_numbers = #stablehlo.dot<
          lhs_contracting_dimensions = [1],
          rhs_contracting_dimensions = [0]
        >,
        precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
      } : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
      %e = stablehlo.add %y, %y : tensor<4xf32>
      distributed.DistributedYield (%d : tensor<2x2xf32>, %e : tensor<4xf32>)
    }
    return %k#0, %k#1 : tensor<2x2xf32>, tensor<4xf32>
  }
}

// -----

// A bottlenecked axis (bandwidth 0.1) carries two contending all-reduces
// (isolated 160 each): node 0, ranked first, takes the whole axis and
// finishes at 160; node 1 sits at scale 0 until then, then finishes at
// 160 + 160 = 320. An independent all-reduce on a much faster axis
// (bandwidth 100, isolated 0.16) runs concurrently and never contends with
// either: its own node finishes at 0.16, far ahead of the makespan, and the
// makespan itself (320) is exactly what the bottleneck pair alone would
// produce. The extra concurrent load on the fast axis does not inflate it.
// CHECK: node 0: communication start=0 finish=160
// CHECK: node 1: communication start=0 finish=320
// CHECK: node 2: communication start=0 finish=0.16
// CHECK: schedule cost: makespan=320 over 3 nodes, critical path=[0]
// CHECK: schedule cost summary: communicating=320 computing=0 utilization=0 idle=320
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>] {axis_bandwidths = [1.0e-1, 1.0e2], axis_latencies = [0.0e0, 0.0e0], launch_latency = 0.0e0}

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %rep2b = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %rep2c = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<4xf32> 0
  %to0 = axis.getaxis tensor<4xf32> 0
  %ti1 = axis.getaxis tensor<4xf32> 0
  %to1 = axis.getaxis tensor<4xf32> 0
  %ti2 = axis.getaxis tensor<4xf32> 0
  %to2 = axis.getaxis tensor<4xf32> 0

  // Collective A on the slow axis 0.
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %mesh_in0 = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %mesh_out0 = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %red0 = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f4 = axis.factor %ti0 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %lhs0 = axis.product (%f4 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %f5 = axis.factor %to0 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %rhs0 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %f6 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %lhs1 = axis.product (%f6 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %rhs1 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %map0 = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]

  %a0 = tensor.empty() : tensor<4xf32>
  %h0 = distributed.Collective %a0 : tensor<4xf32> on %mesh_in0 : !axis.factor_group<2> to tensor<4xf32> on %mesh_out0 : !axis.factor_group<2> reduces (%red0 : !axis.factor_group<2>) maps %map0 : !axis.map {
  ^bb0(%lhs_v0: tensor<f32>, %rhs_v0: tensor<f32>):
    %r0 = stablehlo.add %lhs_v0, %rhs_v0 : tensor<f32>
    stablehlo.return %r0 : tensor<f32>
  }
  %v0 = distributed.Await %h0 : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>

  // Collective B, also on the slow axis 0, independent of A.
  %g1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %mesh_in1 = axis.product (%g1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %g2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %mesh_out1 = axis.product (%g2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %g3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %red1 = axis.product (%g3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %g4 = axis.factor %ti1 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %lhs2 = axis.product (%g4 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %g5 = axis.factor %to1 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %rhs2 = axis.product (%g5 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %g6 = axis.factor %rep2b : !distributed.replication_axis<2> <2, 1>
  %lhs3 = axis.product (%g6 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %g7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %rhs3 = axis.product (%g7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %map1 = axis.map %lhs2, %lhs3 to %rhs2, %rhs3 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]

  %a1 = tensor.empty() : tensor<4xf32>
  %h1 = distributed.Collective %a1 : tensor<4xf32> on %mesh_in1 : !axis.factor_group<2> to tensor<4xf32> on %mesh_out1 : !axis.factor_group<2> reduces (%red1 : !axis.factor_group<2>) maps %map1 : !axis.map {
  ^bb0(%lhs_v1: tensor<f32>, %rhs_v1: tensor<f32>):
    %r1 = stablehlo.add %lhs_v1, %rhs_v1 : tensor<f32>
    stablehlo.return %r1 : tensor<f32>
  }
  %v1 = distributed.Await %h1 : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>

  // Collective C on the fast axis 1, independent of A and B.
  %k1 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in2 = axis.product (%k1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %k2 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out2 = axis.product (%k2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %k3 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %red2 = axis.product (%k3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %k4 = axis.factor %ti2 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %lhs4 = axis.product (%k4 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %k5 = axis.factor %to2 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %rhs4 = axis.product (%k5 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %k6 = axis.factor %rep2c : !distributed.replication_axis<2> <2, 1>
  %lhs5 = axis.product (%k6 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %k7 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs5 = axis.product (%k7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map2 = axis.map %lhs4, %lhs5 to %rhs4, %rhs5 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]

  %a2 = tensor.empty() : tensor<4xf32>
  %h2 = distributed.Collective %a2 : tensor<4xf32> on %mesh_in2 : !axis.factor_group<2> to tensor<4xf32> on %mesh_out2 : !axis.factor_group<2> reduces (%red2 : !axis.factor_group<2>) maps %map2 : !axis.map {
  ^bb0(%lhs_v2: tensor<f32>, %rhs_v2: tensor<f32>):
    %r2 = stablehlo.add %lhs_v2, %rhs_v2 : tensor<f32>
    stablehlo.return %r2 : tensor<f32>
  }
  %v2 = distributed.Await %h2 : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
}

// -----

// A single all-reduce whose axis bandwidth is enormous (1e18) relative to
// its 16-byte payload: transfer time is negligible next to the 0.8 latency
// floor (launch 0.3 plus one round at 0.5), so the makespan is exactly the
// latency, none of it spent waiting on bandwidth. This also exercises the
// simulator's "instant finish" path (ScheduleSimulator.cpp) for a
// contending event with no meaningful resource demand once its latency
// clears: progressive filling has no fixed point for such an event, since it
// never appears in any resource's contender list.
// CHECK: node 0: communication start=0 finish=0.8
// CHECK: schedule cost: makespan=0.8 over 1 nodes, critical path=[0]
// CHECK: schedule cost summary: communicating=0.8 computing=0 utilization=0 idle=0.8
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>] {axis_bandwidths = [1.0e18], axis_latencies = [5.0e-1], launch_latency = 3.0e-1}

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<4xf32> 0
  %to0 = axis.getaxis tensor<4xf32> 0

  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in0 = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out0 = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
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
  %map0 = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]

  %a0 = tensor.empty() : tensor<4xf32>
  %h0 = distributed.Collective %a0 : tensor<4xf32> on %mesh_in0 : !axis.factor_group<2> to tensor<4xf32> on %mesh_out0 : !axis.factor_group<2> reduces (%red0 : !axis.factor_group<2>) maps %map0 : !axis.map {
  ^bb0(%lhs_v0: tensor<f32>, %rhs_v0: tensor<f32>):
    %r0 = stablehlo.add %lhs_v0, %rhs_v0 : tensor<f32>
    stablehlo.return %r0 : tensor<f32>
  }
  %v0 = distributed.Await %h0 : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
}

// -----

// Monotonicity, baseline: the kernel consumes the collective's Await result,
// a real dependency, so it can only start once the collective finishes.
// Makespan is the serial sum, 16 + 2.4 = 18.4. The next split-file case is
// the same graph with that one dependency removed; FileCheck cannot compare
// the two makespans arithmetically, but 16 (next case) < 18.4 (this case)
// demonstrates the required property directly: removing a dependency (or,
// by the same reasoning as case 5, increasing a resource's
// capacity) must never increase the predicted makespan.
// CHECK: node 0: communication start=0 finish=16
// CHECK: node 1: compute start=16 finish=18.4
// CHECK: schedule cost: makespan=18.4 over 2 nodes, critical path=[0, 1]
// CHECK: schedule cost summary: communicating=16 computing=2.4 utilization=0.130435 idle=16
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>] {axis_bandwidths = [1.0e0], axis_latencies = [0.0e0], launch_latency = 0.0e0}

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<4xf32> 0
  %to0 = axis.getaxis tensor<4xf32> 0

  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in0 = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out0 = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
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
  %map0 = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]

  %a0 = tensor.empty() : tensor<4xf32>
  %h0 = distributed.Collective %a0 : tensor<4xf32> on %mesh_in0 : !axis.factor_group<2> to tensor<4xf32> on %mesh_out0 : !axis.factor_group<2> reduces (%red0 : !axis.factor_group<2>) maps %map0 : !axis.map {
  ^bb0(%lhs_v0: tensor<f32>, %rhs_v0: tensor<f32>):
    %r0 = stablehlo.add %lhs_v0, %rhs_v0 : tensor<f32>
    stablehlo.return %r0 : tensor<f32>
  }
  %v0 = distributed.Await %h0 : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>

  %k = distributed.DistributedKernel (%v0 : tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    -> (tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    axes () {
  ^bb0(%x: tensor<4xf32>):
    %d = stablehlo.add %x, %x : tensor<4xf32>
    distributed.DistributedYield (%d : tensor<4xf32>)
  }
}

// -----

// Monotonicity, variant: identical to the previous case except the kernel
// now takes its input from an unrelated value instead of the collective's
// Await result, so the one dependency edge is gone. Makespan drops to
// max(16, 2.4) = 16 < 18.4 (the baseline above), as required: removing a
// dependency never increases the predicted makespan.
// CHECK: node 0: communication start=0 finish=16
// CHECK: node 1: compute start=0 finish=2.4
// CHECK: schedule cost: makespan=16 over 2 nodes, critical path=[0]
// CHECK: schedule cost summary: communicating=16 computing=2.4 utilization=0.15 idle=13.6
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>] {axis_bandwidths = [1.0e0], axis_latencies = [0.0e0], launch_latency = 0.0e0}

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<4xf32> 0
  %to0 = axis.getaxis tensor<4xf32> 0

  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in0 = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out0 = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
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
  %map0 = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]

  %a0 = tensor.empty() : tensor<4xf32>
  %h0 = distributed.Collective %a0 : tensor<4xf32> on %mesh_in0 : !axis.factor_group<2> to tensor<4xf32> on %mesh_out0 : !axis.factor_group<2> reduces (%red0 : !axis.factor_group<2>) maps %map0 : !axis.map {
  ^bb0(%lhs_v0: tensor<f32>, %rhs_v0: tensor<f32>):
    %r0 = stablehlo.add %lhs_v0, %rhs_v0 : tensor<f32>
    stablehlo.return %r0 : tensor<f32>
  }
  %v0 = distributed.Await %h0 : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>

  %b0 = tensor.empty() : tensor<4xf32>
  %k = distributed.DistributedKernel (%b0 : tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    -> (tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    axes () {
  ^bb0(%x: tensor<4xf32>):
    %d = stablehlo.add %x, %x : tensor<4xf32>
    distributed.DistributedYield (%d : tensor<4xf32>)
  }
}
