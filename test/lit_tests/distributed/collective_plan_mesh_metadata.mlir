// RUN: enzymexlamlir-opt --distributed-atomize-collectives --distributed-print-collective-plan=chain=true %s 2>&1 >/dev/null | FileCheck %s
// RUN: enzymexlamlir-opt --distributed-atomize-collectives --distributed-print-collective-plan="chain=true bandwidths=4" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=OVERRIDE

// The decomposer costs collectives with the physical mesh's metadata, and the
// pass's explicit bandwidths option overrides the mesh's axis_bandwidths.
//
// An all-gather of a 2-byte payload over 2 devices sends V = 2 bytes. The mesh
// declares bandwidth 2, round latency 0.5 and launch latency 1, so the latency
// is 1 + 0.5 * 1 = 1.5 and the duration is 1.5 + 2 / 2 = 2.5. With the
// bandwidth overridden to 4 it is 1.5 + 2 / 4 = 2.
// CHECK: chain: 1 steps, total duration 2.5
// CHECK: latency: 1.5
// OVERRIDE: chain: 1 steps, total duration 2
module @all_gather {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>] {axis_bandwidths = [2.0e0], axis_latencies = [5.0e-1], launch_latency = 1.0e0}

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
