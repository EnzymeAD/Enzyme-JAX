// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true bandwidths=4,1" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=FAST0
// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true bandwidths=1,4" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=FAST1
// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true bandwidths=1000,1000" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=FAST

// Non-uniform per-axis bandwidth changes what the decomposer picks. Round
// latency 0.01 and launch latency 0.1 as in decompose_collective.mlir.

// Order of two gathers. Payload S = 8, axis0 has n = 2 and axis1 has n = 4. A gather on axis a at payload P costs P (n - 1) / BW[a] plus latency (0.11 for n = 2, 0.12 for n = 4), and the payload then grows n-fold.
// Uniform bandwidth: axis0 first is 8 + 48 = 56 (+ latencies), axis1 first is 24 + 32 = 56, an exact tie, so the order is not asserted here.
// bandwidths=4,1 (axis0 fast, axis1 slow): axis0 first = 0.11 + 8/4 + 0.12 + 48/1 = 50.23; axis1 first = 0.12 + 24/1 + 0.11 + 32/4 = 32.23. The slow axis goes first, while the payload is small.
// bandwidths=1,4: axis0 first = 0.11 + 8 + 0.12 + 48/4 = 20.23; axis1 first = 0.12 + 24/4 + 0.11 + 32/1 = 38.23. Now the order flips.
// FAST0: chain: 2 steps, total duration 32.23
// FAST0-NEXT: step 0: all-gather
// FAST0-NEXT: atoms: axis1.0(x4)
// FAST0-NEXT: payload: 8 -> 32
// FAST0-NEXT: latency: 0.12
// FAST0-NEXT: V: [0, 24]
// FAST0-NEXT: rho: [0, 1]
// FAST0-NEXT: duration: 24.12
// FAST0-NEXT: step 1: all-gather
// FAST0-NEXT: atoms: axis0.0(x2)
// FAST0-NEXT: payload: 32 -> 64
// FAST0-NEXT: latency: 0.11
// FAST0-NEXT: V: [32, 0]
// FAST0-NEXT: rho: [1, 0]
// FAST0-NEXT: duration: 8.11
// FAST0-NEXT: semantics: verified
// FAST0-NEXT: input port: tensor<8x1x1xi8> (the collective's input_object operand) -> step 0
// FAST0-NEXT: result port: tensor<8x2x4xi8> (the await result, 0 uses) <- step 1
// FAST1: chain: 2 steps, total duration 20.23
// FAST1-NEXT: step 0: all-gather
// FAST1-NEXT: atoms: axis0.0(x2)
// FAST1-NEXT: payload: 8 -> 16
// FAST1-NEXT: latency: 0.11
// FAST1-NEXT: V: [8, 0]
// FAST1-NEXT: rho: [1, 0]
// FAST1-NEXT: duration: 8.11
// FAST1-NEXT: step 1: all-gather
// FAST1-NEXT: atoms: axis1.0(x4)
// FAST1-NEXT: payload: 16 -> 64
// FAST1-NEXT: latency: 0.12
// FAST1-NEXT: V: [0, 48]
// FAST1-NEXT: rho: [0, 1]
// FAST1-NEXT: duration: 12.12
// FAST1-NEXT: semantics: verified
// FAST1-NEXT: input port: tensor<8x1x1xi8> (the collective's input_object operand) -> step 0
// FAST1-NEXT: result port: tensor<8x2x4xi8> (the await result, 0 uses) <- step 1
module @gather_two_axes {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<4, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<4, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %rep4 = distributed.ReplicationAxis 4 : !distributed.replication_axis<4>
  %ti0 = axis.getaxis tensor<8x1x1xi8> 0
  %ti1 = axis.getaxis tensor<8x1x1xi8> 1
  %ti2 = axis.getaxis tensor<8x1x1xi8> 2
  %to0 = axis.getaxis tensor<8x2x4xi8> 0
  %to1 = axis.getaxis tensor<8x2x4xi8> 1
  %to2 = axis.getaxis tensor<8x2x4xi8> 2
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %f4 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %mesh_out = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>, %f4 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f5 = axis.factor %ti0 : !axis.shape_axis<tensor<8x1x1xi8>, 0> <8, 1>
  %lhs0 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<8x1x1xi8>, 0>, 8, 1>)
  %f6 = axis.factor %to0 : !axis.shape_axis<tensor<8x2x4xi8>, 0> <8, 1>
  %rhs0 = axis.product (%f6 : !axis.axis_factor<!axis.shape_axis<tensor<8x2x4xi8>, 0>, 8, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %lhs1 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>)
  %f8 = axis.factor %to1 : !axis.shape_axis<tensor<8x2x4xi8>, 1> <2, 1>
  %rhs1 = axis.product (%f8 : !axis.axis_factor<!axis.shape_axis<tensor<8x2x4xi8>, 1>, 2, 1>)
  %f9 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %lhs2 = axis.product (%f9 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f10 = axis.factor %to2 : !axis.shape_axis<tensor<8x2x4xi8>, 2> <4, 1>
  %rhs2 = axis.product (%f10 : !axis.axis_factor<!axis.shape_axis<tensor<8x2x4xi8>, 2>, 4, 1>)
  %f11 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %lhs3 = axis.product (%f11 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f12 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %rhs3 = axis.product (%f12 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>)
  %f13 = axis.factor %rep4 : !distributed.replication_axis<4> <4, 1>
  %lhs4 = axis.product (%f13 : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
  %f14 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %rhs4 = axis.product (%f14 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %map = axis.map %lhs0, %lhs1, %lhs2, %lhs3, %lhs4 to %rhs0, %rhs1, %rhs2, %rhs3, %rhs4 : [!axis.factor_group<8>, !axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<4>] [!axis.factor_group<8>, !axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<4>]

  %in = tensor.empty() : tensor<8x1x1xi8>
  %h = distributed.Collective %in : tensor<8x1x1xi8> on %mesh_in : !axis.factor_group<8> to tensor<8x2x4xi8> on %mesh_out : !axis.factor_group<8> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<8x2x4xi8>> -> tensor<8x2x4xi8>
}

// -----

// Bruck versus pairwise all-to-all (N9) depends on bandwidth. Two all-to-alls at S = 16 with bandwidth 1000 on both axes, so latency dominates.
//   axis0 (n = 2): pairwise V = 16 * 1 / 2 = 8, latency 0.11 -> 0.11 + 8/1000 = 0.118; Bruck (k = 1) is identical, ties go to pairwise.
//   axis1 (n = 4): pairwise V = 12, latency 0.13 -> 0.142; Bruck V = 16 * 2 / 2 = 16, latency 0.1 + 2 * 0.01 = 0.12 -> 0.136. Bruck wins (at bandwidth 1 pairwise wins: 12.13 vs 16.12).
//   total 0.118 + 0.136 = 0.254
// FAST: chain: 2 steps, total duration 0.254
// FAST-NEXT: step 0: all-to-all
// FAST-NEXT: atoms: axis0.0(x2)
// FAST-NEXT: payload: 16 -> 16
// FAST-NEXT: latency: 0.11
// FAST-NEXT: V: [8, 0]
// FAST-NEXT: rho: [1, 0]
// FAST-NEXT: duration: 0.118
// FAST-NEXT: step 1: all-to-all
// FAST-NEXT: atoms: axis1.0(x4)
// FAST-NEXT: payload: 16 -> 16
// FAST-NEXT: latency: 0.12
// FAST-NEXT: V: [0, 16]
// FAST-NEXT: rho: [0, 1]
// FAST-NEXT: duration: 0.136
// FAST-NEXT: semantics: verified
// FAST-NEXT: input port: tensor<16x1xi8> (the collective's input_object operand) -> step 0
// FAST-NEXT: result port: tensor<2x8xi8> (the await result, 0 uses) <- step 1
module @all_to_all_two_axes {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<4, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<4, 1>
  %ti0 = axis.getaxis tensor<16x1xi8> 0
  %ti1 = axis.getaxis tensor<16x1xi8> 1
  %to0 = axis.getaxis tensor<2x8xi8> 0
  %to1 = axis.getaxis tensor<2x8xi8> 1
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %f4 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %mesh_out = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>, %f4 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f5 = axis.factor %ti0 : !axis.shape_axis<tensor<16x1xi8>, 0> <16, 1>
  %lhs0 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<16x1xi8>, 0>, 16, 1>)
  %f6 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %f7 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %f8 = axis.factor %to0 : !axis.shape_axis<tensor<2x8xi8>, 0> <2, 1>
  %rhs0 = axis.product (%f6 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>, %f7 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>, %f8 : !axis.axis_factor<!axis.shape_axis<tensor<2x8xi8>, 0>, 2, 1>)
  %f9 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %f10 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %lhs1 = axis.product (%f9 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>, %f10 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f11 = axis.factor %to1 : !axis.shape_axis<tensor<2x8xi8>, 1> <8, 1>
  %rhs1 = axis.product (%f11 : !axis.axis_factor<!axis.shape_axis<tensor<2x8xi8>, 1>, 8, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<16>, !axis.factor_group<8>] [!axis.factor_group<16>, !axis.factor_group<8>]

  %in = tensor.empty() : tensor<16x1xi8>
  %h = distributed.Collective %in : tensor<16x1xi8> on %mesh_in : !axis.factor_group<8> to tensor<2x8xi8> on %mesh_out : !axis.factor_group<8> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<2x8xi8>> -> tensor<2x8xi8>
}