// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true bandwidths=1,4" %s 2>&1 >/dev/null | FileCheck %s --check-prefixes=EXACT,ALL
// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true bandwidths=1,4 k-order=100" %s 2>&1 >/dev/null | FileCheck %s --check-prefixes=EXACT,ALL
// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true bandwidths=1,4 k-order=1" %s 2>&1 >/dev/null | FileCheck %s --check-prefixes=GREEDY,ALL
// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true bandwidths=1,4 max-states=1" %s 2>&1 >/dev/null | FileCheck %s --check-prefixes=BUDGET

// The heuristic step orderer (k-order, k-variant, max-states). With none of the
// options the search is the exact one; k-order=100 applies the size tiers and
// the soft score but keeps every candidate of the lowest tier, and must land on
// the same cost as the exact search. Round latency 0.01 and launch latency 0.1
// as in decompose_collective.mlir.

// Greedy loses here. Two gathers before one slice (S = 2, n = 2; see
// decompose_collective_meshcoupled_bandwidth.mlir). mesh0 is (tile, mesh1),
// mesh1 is (mesh0, replicate), and axis0 has bandwidth 1, axis1 bandwidth 4.
// Both first steps are expands. The soft score is -1 / BW, so the slow axis0
// gather ranks first:
//   greedy (k-order=1): gather axis0 = 0.11 + 2 / 1 = 2.11 (payload 2 -> 4),
//     gather axis1 = 0.11 + 4 / 4 = 1.11 (4 -> 8), free slice: 3.22
// The exact search instead gathers the fast axis1 first and then turns mesh0's
// gather + slice into one all-to-all, which the axis1 gather unlocks (D14):
//   gather axis1 = 0.11 + 2 / 4 = 0.61 (2 -> 4), all-to-all axis0 at payload 4
//     (pairwise V = 4 / 2 = 2) = 0.11 + 2 / 1 = 2.11: 2.72
// The exchange argument cannot see this, since the unlock is a dependency edge
// (D12) rather than a cost; k-order=2 already keeps the second gather.
// EXACT: chain: 2 steps, total duration 2.72
// EXACT-NEXT: step 0: all-gather
// EXACT-NEXT: atoms: axis1.0(x2)
// EXACT-NEXT: payload: 2 -> 4
// EXACT-NEXT: latency: 0.11
// EXACT-NEXT: V: [0, 2]
// EXACT-NEXT: rho: [0, 1]
// EXACT-NEXT: duration: 0.61
// EXACT-NEXT: step 1: all-to-all
// EXACT-NEXT: atoms: axis0.0(x2)
// EXACT-NEXT: payload: 4 -> 4
// EXACT-NEXT: latency: 0.11
// EXACT-NEXT: V: [2, 0]
// EXACT-NEXT: rho: [1, 0]
// EXACT-NEXT: duration: 2.11
// EXACT-NEXT: semantics: verified
// GREEDY: chain: 3 steps, total duration 3.22
// GREEDY-NEXT: step 0: all-gather
// GREEDY-NEXT: atoms: axis0.0(x2)
// GREEDY-NEXT: payload: 2 -> 4
// GREEDY-NEXT: latency: 0.11
// GREEDY-NEXT: V: [2, 0]
// GREEDY-NEXT: rho: [1, 0]
// GREEDY-NEXT: duration: 2.11
// GREEDY-NEXT: step 1: all-gather
// GREEDY-NEXT: atoms: axis1.0(x2)
// GREEDY-NEXT: payload: 4 -> 8
// GREEDY-NEXT: latency: 0.11
// GREEDY-NEXT: V: [0, 4]
// GREEDY-NEXT: rho: [0, 1]
// GREEDY-NEXT: duration: 1.11
// GREEDY-NEXT: step 2: local-slice
// GREEDY-NEXT: atoms: axis0.0(x2)
// GREEDY-NEXT: payload: 8 -> 4
// GREEDY: semantics: verified
// BUDGET: print-collective-plan: no chain (search budget exceeded (more than 1 decomposer states))
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

// -----

// A shrink tier ordered by the soft score, at every k. Two reduce-scatters (S =
// 16), each n = 2 onto its own tile atom, axis0 bandwidth 1 and axis1
// bandwidth 4. The score is 1 / BW, so the fast axis1 goes first while the
// payload is large: axis1 = 0.11 + 8 / 4 = 2.11 (16 -> 8), axis0 = 0.11 + 4 /
// 1 = 4.11 (8 -> 4), total 6.22. Even greedy (k-order=1) finds it.
// ALL: chain: 2 steps, total duration 6.22
// ALL-NEXT: step 0: reduce-scatter
// ALL-NEXT: atoms: axis1.0(x2)
// ALL-NEXT: payload: 16 -> 8
// ALL-NEXT: latency: 0.11
// ALL-NEXT: V: [0, 8]
// ALL-NEXT: rho: [0, 1]
// ALL-NEXT: duration: 2.11
// ALL-NEXT: step 1: reduce-scatter
// ALL-NEXT: atoms: axis0.0(x2)
// ALL-NEXT: payload: 8 -> 4
// ALL-NEXT: latency: 0.11
// ALL-NEXT: V: [4, 0]
// ALL-NEXT: rho: [1, 0]
// ALL-NEXT: duration: 4.11
// ALL-NEXT: semantics: verified
module @reduce_scatter_two_axes {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %ti0 = axis.getaxis tensor<16xi8> 0
  %to0 = axis.getaxis tensor<4xi8> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f4 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f4 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f5 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f6 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %red0 = axis.product (%f5 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f6 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f7 = axis.factor %ti0 : !axis.shape_axis<tensor<16xi8>, 0> <16, 1>
  %lhs0 = axis.product (%f7 : !axis.axis_factor<!axis.shape_axis<tensor<16xi8>, 0>, 16, 1>)
  %f8 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f9 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f10 = axis.factor %to0 : !axis.shape_axis<tensor<4xi8>, 0> <4, 1>
  %rhs0 = axis.product (%f8 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f9 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %f10 : !axis.axis_factor<!axis.shape_axis<tensor<4xi8>, 0>, 4, 1>)
  %map = axis.map %lhs0 to %rhs0 : [!axis.factor_group<16>] [!axis.factor_group<16>]

  %in = tensor.empty() : tensor<16xi8>
  %h = distributed.Collective %in : tensor<16xi8> on %mesh_in : !axis.factor_group<4> to tensor<4xi8> on %mesh_out : !axis.factor_group<4> reduces (%red0 : !axis.factor_group<4>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<i8>, %rhs_v: tensor<i8>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<i8>
    stablehlo.return %r : tensor<i8>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xi8>> -> tensor<4xi8>
}
