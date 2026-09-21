// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true bandwidths=4,1 disable-relay-variants=true" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=FAST0
// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true bandwidths=1,4 disable-relay-variants=true" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=FAST1

// Bandwidth changes the order of the first halves of a half-split component (see decompose_collective_meshcoupled.mlir); the relay variants are disabled so the half-split is what is tested. Round latency 0.01 and launch latency 0.1 as in decompose_collective.mlir.

// Two gathers before the slice. mesh0 is (tile, mesh1): its digit joins the output tile and it takes mesh1's digit; mesh1 is (mesh0, replicate). The one slice (onto mesh0) waits for both gathers, so both grow the payload for good. S = 2, n = 2. A gather at payload P on axis a costs P / BW[a] plus latency 0.11.
//   mesh0 first: 2 / BW0 + 4 / BW1
//   mesh1 first: 2 / BW1 + 4 / BW0
// The slower axis goes first, while the payload is small.
// bandwidths=4,1 (axis0 fast): mesh0 first = 0.11 + 0.5 + 0.11 + 4 = 4.72; mesh1 first = 0.11 + 2 + 0.11 + 1 = 3.22.
// bandwidths=1,4 (axis1 fast): mesh0 first = 0.11 + 2 + 0.11 + 1 = 3.22; mesh1 first = 0.11 + 0.5 + 0.11 + 4 = 4.72.
// The final slice is free (8 -> 4).
// FAST0: chain: 3 steps, total duration 3.22
// FAST0-NEXT: step 0: all-gather
// FAST0-NEXT: atoms: axis1.0(x2)
// FAST0-NEXT: payload: 2 -> 4
// FAST0-NEXT: latency: 0.11
// FAST0-NEXT: V: [0, 2]
// FAST0-NEXT: rho: [0, 1]
// FAST0-NEXT: duration: 2.11
// FAST0-NEXT: step 1: all-gather
// FAST0-NEXT: atoms: axis0.0(x2)
// FAST0-NEXT: payload: 4 -> 8
// FAST0-NEXT: latency: 0.11
// FAST0-NEXT: V: [4, 0]
// FAST0-NEXT: rho: [1, 0]
// FAST0-NEXT: duration: 1.11
// FAST0-NEXT: step 2: local-slice
// FAST0-NEXT: atoms: axis0.0(x2)
// FAST0-NEXT: payload: 8 -> 4
// FAST0-NEXT: latency: 0
// FAST0-NEXT: V: [0, 0]
// FAST0-NEXT: rho: [0, 0]
// FAST0-NEXT: duration: 0
// FAST0-NEXT: semantics: verified
// FAST0-NEXT: input port: tensor<2xi8> (the collective's input_object operand) -> step 0
// FAST0-NEXT: result port: tensor<4xi8> (the await result, 0 uses) <- step 2
// FAST1: chain: 3 steps, total duration 3.22
// FAST1-NEXT: step 0: all-gather
// FAST1-NEXT: atoms: axis0.0(x2)
// FAST1-NEXT: payload: 2 -> 4
// FAST1-NEXT: latency: 0.11
// FAST1-NEXT: V: [2, 0]
// FAST1-NEXT: rho: [1, 0]
// FAST1-NEXT: duration: 2.11
// FAST1-NEXT: step 1: all-gather
// FAST1-NEXT: atoms: axis1.0(x2)
// FAST1-NEXT: payload: 4 -> 8
// FAST1-NEXT: latency: 0.11
// FAST1-NEXT: V: [0, 4]
// FAST1-NEXT: rho: [0, 1]
// FAST1-NEXT: duration: 1.11
// FAST1-NEXT: step 2: local-slice
// FAST1-NEXT: atoms: axis0.0(x2)
// FAST1-NEXT: payload: 8 -> 4
// FAST1-NEXT: latency: 0
// FAST1-NEXT: V: [0, 0]
// FAST1-NEXT: rho: [0, 0]
// FAST1-NEXT: duration: 0
// FAST1-NEXT: semantics: verified
// FAST1-NEXT: input port: tensor<2xi8> (the collective's input_object operand) -> step 0
// FAST1-NEXT: result port: tensor<4xi8> (the await result, 0 uses) <- step 2
module @gathers_before_slice {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<2xi8> 0
  %to0 = axis.getaxis tensor<4xi8> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f4 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f4 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f5 = axis.factor %ti0 : !axis.shape_axis<tensor<2xi8>, 0> <2, 1>
  %lhs0 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<2xi8>, 0>, 2, 1>)
  %f6 = axis.factor %to0 : !axis.shape_axis<tensor<4xi8>, 0> <2, 1>
  %rhs0 = axis.product (%f6 : !axis.axis_factor<!axis.shape_axis<tensor<4xi8>, 0>, 2, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %lhs1 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f8 = axis.factor %to0 : !axis.shape_axis<tensor<4xi8>, 0> <2, 2>
  %rhs1 = axis.product (%f8 : !axis.axis_factor<!axis.shape_axis<tensor<4xi8>, 0>, 2, 2>)
  %f9 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs2 = axis.product (%f9 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f10 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %rhs2 = axis.product (%f10 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f11 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %lhs3 = axis.product (%f11 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f12 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs3 = axis.product (%f12 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map = axis.map %lhs0, %lhs1, %lhs2, %lhs3 to %rhs0, %rhs1, %rhs2, %rhs3 : [!axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<2xi8>
  %h = distributed.Collective %in : tensor<2xi8> on %mesh_in : !axis.factor_group<4> to tensor<4xi8> on %mesh_out : !axis.factor_group<4> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xi8>> -> tensor<4xi8>
}
