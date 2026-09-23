// RUN: enzymexlamlir-opt --distributed-atomize-collectives --distributed-print-collective-plan=chain=true %s 2>&1 >/dev/null | FileCheck %s
// RUN: enzymexlamlir-opt --distributed-atomize-collectives --distributed-print-collective-plan="chain=true bandwidths=4" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=OVERRIDE

// The decomposer costs collectives with the physical mesh's metadata, and the
// pass's explicit bandwidths option overrides the mesh's axis_bandwidths. The
// mesh's own attributes are SI seconds/bytes-per-second (Ops.td); the pass's
// own bandwidths= option is already in the internal bytes/ns representation
// (CollectiveCost.h's Units note), so it bypasses that conversion entirely.
//
// An all-gather of a 2-byte payload over 2 devices sends V = 2 bytes. The mesh
// declares bandwidth 2.0e10 bytes/s (20 bytes/ns), round latency 5.0e-7s
// (500ns) and launch latency 1.0e-6s (1000ns), so the latency is
// 1000 + 500 = 1500ns and the transfer time (2/20 = 0.1ns) rounds to 0,
// giving a duration of 1500. With the bandwidth overridden to 4 bytes/ns the
// transfer time is 2/4 = 0.5ns, which rounds up to 1, giving 1501.
// CHECK: chain: 1 steps, total duration 1500
// CHECK: latency: 1500
// OVERRIDE: chain: 1 steps, total duration 1501
module @all_gather {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>] {axis_bandwidths = [2.0e10], axis_latencies = [5.0e-7], launch_latency = 1.0e-6}

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<2xi8> 0
  %to0 = axis.getaxis tensor<4xi8> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f4 = axis.factor %ti0 : !axis.shape_axis<tensor<2xi8>, 0> <2, 1>
  %lhs0 = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %f4 : !axis.axis_factor<!axis.shape_axis<tensor<2xi8>, 0>, 2, 1>)
  %f5 = axis.factor %to0 : !axis.shape_axis<tensor<4xi8>, 0> <4, 1>
  %rhs0 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<4xi8>, 0>, 4, 1>)
  %f6 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %lhs1 = axis.product (%f6 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs1 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<2xi8>
  %h = distributed.Collective %in : tensor<2xi8> on %mesh_in : !axis.factor_group<2> to tensor<4xi8> on %mesh_out : !axis.factor_group<2> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xi8>> -> tensor<4xi8>
}
