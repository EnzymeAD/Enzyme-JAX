// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan=chain=true %s 2>&1 >/dev/null | FileCheck %s
// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true disable-relay-variants=true" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOREL
// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true bandwidths=1,4" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=FAST
// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true bandwidths=1,4 disable-relay-variants=true" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=FASTNOREL

// The relay variants (D14 to D16) mesh-coupled components are offered besides
// the half-split (see decompose_collective_meshcoupled.mlir, which pins the
// half-split with disable-relay-variants=true). CHECK/NOREL use the default
// parameters (bandwidth 1, round latency 0.01, launch latency 0.1); FAST/
// FASTNOREL give axis1 four times axis0's bandwidth, which is what D16
// (conjugation) needs to have any effect.

// D14: fused exchange. mesh0 and mesh1 each both release their own digit (in
// is Tile or Mesh) and place another (out is Tile or Mesh), the shape a
// TileToTile or ReduceScatter unit fuses into one all-to-all instead of a
// gather followed by a free slice. S = 2, n = 2.
//   half-split (NOREL): all-gather mesh0 (V = 2, 2.11) + slice + all-gather
//     mesh1 (V = 2, 2.11) + slice = 4.22
//   fused exchange (CHECK): all-to-all mesh0 (pairwise V = 2 * 1 / 2 = 1,
//     latency 0.11, duration 1.11) + all-to-all mesh1 (the same, 1.11) = 2.22
// CHECK: chain: 2 steps, total duration 2.22
// CHECK-NEXT: step 0: all-to-all
// CHECK-NEXT: atoms: axis0.0(x2)
// CHECK-NEXT: payload: 2 -> 2
// CHECK-NEXT: latency: 0.11
// CHECK-NEXT: V: [1, 0]
// CHECK-NEXT: rho: [1, 0]
// CHECK-NEXT: duration: 1.11
// CHECK-NEXT: step 1: all-to-all
// CHECK-NEXT: atoms: axis1.0(x2)
// CHECK-NEXT: payload: 2 -> 2
// CHECK-NEXT: latency: 0.11
// CHECK-NEXT: V: [0, 1]
// CHECK-NEXT: rho: [0, 1]
// CHECK-NEXT: duration: 1.11
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<2xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<2xi8> (the await result, 0 uses) <- step 1
// NOREL: chain: 4 steps, total duration 4.22
// NOREL-NEXT: step 0: all-gather
// NOREL-NEXT: atoms: axis0.0(x2)
// NOREL-NEXT: payload: 2 -> 4
// NOREL-NEXT: latency: 0.11
// NOREL-NEXT: V: [2, 0]
// NOREL-NEXT: rho: [1, 0]
// NOREL-NEXT: duration: 2.11
// NOREL-NEXT: step 1: local-slice
// NOREL-NEXT: atoms: axis0.0(x2)
// NOREL-NEXT: payload: 4 -> 2
// NOREL-NEXT: latency: 0
// NOREL-NEXT: V: [0, 0]
// NOREL-NEXT: rho: [0, 0]
// NOREL-NEXT: duration: 0
// NOREL-NEXT: step 2: all-gather
// NOREL-NEXT: atoms: axis1.0(x2)
// NOREL-NEXT: payload: 2 -> 4
// NOREL-NEXT: latency: 0.11
// NOREL-NEXT: V: [0, 2]
// NOREL-NEXT: rho: [0, 1]
// NOREL-NEXT: duration: 2.11
// NOREL-NEXT: step 3: local-slice
// NOREL-NEXT: atoms: axis1.0(x2)
// NOREL-NEXT: payload: 4 -> 2
// NOREL-NEXT: latency: 0
// NOREL-NEXT: V: [0, 0]
// NOREL-NEXT: rho: [0, 0]
// NOREL-NEXT: duration: 0
// NOREL-NEXT: semantics: verified
// NOREL-NEXT: input port: tensor<2xi8> (the collective's input_object operand) -> step 0
// NOREL-NEXT: result port: tensor<2xi8> (the await result, 0 uses) <- step 3
module @d14_fused_exchange {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %ti0 = axis.getaxis tensor<2xi8> 0
  %to0 = axis.getaxis tensor<2xi8> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f4 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f4 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f5 = axis.factor %ti0 : !axis.shape_axis<tensor<2xi8>, 0> <2, 1>
  %lhs0 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<2xi8>, 0>, 2, 1>)
  %f6 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %rhs0 = axis.product (%f6 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %lhs1 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f8 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs1 = axis.product (%f8 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f9 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs2 = axis.product (%f9 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f10 = axis.factor %to0 : !axis.shape_axis<tensor<2xi8>, 0> <2, 1>
  %rhs2 = axis.product (%f10 : !axis.axis_factor<!axis.shape_axis<tensor<2xi8>, 0>, 2, 1>)
  %map = axis.map %lhs0, %lhs1, %lhs2 to %rhs0, %rhs1, %rhs2 : [!axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<2xi8>
  %h = distributed.Collective %in : tensor<2xi8> on %mesh_in : !axis.factor_group<4> to tensor<2xi8> on %mesh_out : !axis.factor_group<4> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<2xi8>> -> tensor<2xi8>
}

// -----

// D15: path closure. mesh0 is reduced and its output comes from mesh1, whose
// own output is replicated: an open path (mesh1 -> mesh0), not a permutation
// cycle. n = 2, S = 4.
//   half-split (NOREL): all-reduce mesh0 (V = 4, 4.11) + all-gather mesh1
//     (V = 4, 4.11) + free slice = 8.22
//   path closure (CHECK, default bandwidth): all-reduce mesh0 empties it
//     (4.11), then one permute of both atoms, V = S * (1 - 1/2) = 2 on each
//     axis, latency 0.1 + 0.01 + 0.01 = 0.12, duration 2.12. Total 6.23.
//   at FAST (axis1 four times axis0's bandwidth): the half-split's expensive
//     step (the gather) lands on the fast axis1 (1.11 instead of 4.11: total
//     5.22), while the permute is still bottlenecked by slow axis0 (V = 2 on
//     each axis, transferTime = max(2 / 1, 2 / 4) = 2, so still 2.12) and the
//     all-reduce is unaffected by axis1's bandwidth (still 4.11), so closure
//     stays 6.23. Closure is tried (it is still a variant) but no longer
//     cheaper, so FAST and FASTNOREL agree: the half-split wins here, at 5.22.
// CHECK: chain: 2 steps, total duration 6.23
// CHECK-NEXT: step 0: all-reduce
// CHECK-NEXT: atoms: axis0.0(x2)
// CHECK-NEXT: payload: 4 -> 4
// CHECK-NEXT: latency: 0.11
// CHECK-NEXT: V: [4, 0]
// CHECK-NEXT: rho: [1, 0]
// CHECK-NEXT: duration: 4.11
// CHECK-NEXT: step 1: permute
// CHECK-NEXT: atoms: axis0.0(x2) axis1.0(x2)
// CHECK-NEXT: payload: 4 -> 4
// CHECK-NEXT: latency: 0.12
// CHECK-NEXT: V: [2, 2]
// CHECK-NEXT: rho: [1, 1]
// CHECK-NEXT: duration: 2.12
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<4xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<4xi8> (the await result, 0 uses) <- step 1
// NOREL: chain: 3 steps, total duration 8.22
// NOREL-NEXT: step 0: all-reduce
// NOREL-NEXT: atoms: axis0.0(x2)
// NOREL-NEXT: payload: 4 -> 4
// NOREL-NEXT: latency: 0.11
// NOREL-NEXT: V: [4, 0]
// NOREL-NEXT: rho: [1, 0]
// NOREL-NEXT: duration: 4.11
// NOREL-NEXT: step 1: all-gather
// NOREL-NEXT: atoms: axis1.0(x2)
// NOREL-NEXT: payload: 4 -> 8
// NOREL-NEXT: latency: 0.11
// NOREL-NEXT: V: [0, 4]
// NOREL-NEXT: rho: [0, 1]
// NOREL-NEXT: duration: 4.11
// NOREL-NEXT: step 2: local-slice
// NOREL-NEXT: atoms: axis0.0(x2)
// NOREL-NEXT: payload: 8 -> 4
// NOREL-NEXT: latency: 0
// NOREL-NEXT: V: [0, 0]
// NOREL-NEXT: rho: [0, 0]
// NOREL-NEXT: duration: 0
// NOREL-NEXT: semantics: verified
// NOREL-NEXT: input port: tensor<4xi8> (the collective's input_object operand) -> step 0
// NOREL-NEXT: result port: tensor<4xi8> (the await result, 0 uses) <- step 2
// FAST: chain: 3 steps, total duration 5.22
// FAST-NEXT: step 0: all-reduce
// FAST-NEXT: atoms: axis0.0(x2)
// FAST-NEXT: payload: 4 -> 4
// FAST-NEXT: latency: 0.11
// FAST-NEXT: V: [4, 0]
// FAST-NEXT: rho: [1, 0]
// FAST-NEXT: duration: 4.11
// FAST-NEXT: step 1: all-gather
// FAST-NEXT: atoms: axis1.0(x2)
// FAST-NEXT: payload: 4 -> 8
// FAST-NEXT: latency: 0.11
// FAST-NEXT: V: [0, 4]
// FAST-NEXT: rho: [0, 1]
// FAST-NEXT: duration: 1.11
// FAST-NEXT: step 2: local-slice
// FAST-NEXT: atoms: axis0.0(x2)
// FAST-NEXT: payload: 8 -> 4
// FAST-NEXT: latency: 0
// FAST-NEXT: V: [0, 0]
// FAST-NEXT: rho: [0, 0]
// FAST-NEXT: duration: 0
// FAST-NEXT: semantics: verified
// FAST-NEXT: input port: tensor<4xi8> (the collective's input_object operand) -> step 0
// FAST-NEXT: result port: tensor<4xi8> (the await result, 0 uses) <- step 2
// FASTNOREL: chain: 3 steps, total duration 5.22
// FASTNOREL-NEXT: step 0: all-reduce
// FASTNOREL-NEXT: atoms: axis0.0(x2)
// FASTNOREL-NEXT: payload: 4 -> 4
// FASTNOREL-NEXT: latency: 0.11
// FASTNOREL-NEXT: V: [4, 0]
// FASTNOREL-NEXT: rho: [1, 0]
// FASTNOREL-NEXT: duration: 4.11
// FASTNOREL-NEXT: step 1: all-gather
// FASTNOREL-NEXT: atoms: axis1.0(x2)
// FASTNOREL-NEXT: payload: 4 -> 8
// FASTNOREL-NEXT: latency: 0.11
// FASTNOREL-NEXT: V: [0, 4]
// FASTNOREL-NEXT: rho: [0, 1]
// FASTNOREL-NEXT: duration: 1.11
// FASTNOREL-NEXT: step 2: local-slice
// FASTNOREL-NEXT: atoms: axis0.0(x2)
// FASTNOREL-NEXT: payload: 8 -> 4
// FASTNOREL-NEXT: latency: 0
// FASTNOREL-NEXT: V: [0, 0]
// FASTNOREL-NEXT: rho: [0, 0]
// FASTNOREL-NEXT: duration: 0
// FASTNOREL-NEXT: semantics: verified
// FASTNOREL-NEXT: input port: tensor<4xi8> (the collective's input_object operand) -> step 0
// FASTNOREL-NEXT: result port: tensor<4xi8> (the await result, 0 uses) <- step 2
module @d15_reduce_then_permute_n2 {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<4xi8> 0
  %to0 = axis.getaxis tensor<4xi8> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f4 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f4 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f5 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %red0 = axis.product (%f5 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f6 = axis.factor %ti0 : !axis.shape_axis<tensor<4xi8>, 0> <4, 1>
  %lhs0 = axis.product (%f6 : !axis.axis_factor<!axis.shape_axis<tensor<4xi8>, 0>, 4, 1>)
  %f7 = axis.factor %to0 : !axis.shape_axis<tensor<4xi8>, 0> <4, 1>
  %rhs0 = axis.product (%f7 : !axis.axis_factor<!axis.shape_axis<tensor<4xi8>, 0>, 4, 1>)
  %f8 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs1 = axis.product (%f8 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f9 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %rhs1 = axis.product (%f9 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f10 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %lhs2 = axis.product (%f10 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f11 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs2 = axis.product (%f11 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map = axis.map %lhs0, %lhs1, %lhs2 to %rhs0, %rhs1, %rhs2 : [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<4xi8>
  %h = distributed.Collective %in : tensor<4xi8> on %mesh_in : !axis.factor_group<4> to tensor<4xi8> on %mesh_out : !axis.factor_group<4> reduces (%red0 : !axis.factor_group<2>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<i8>, %rhs_v: tensor<i8>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<i8>
    stablehlo.return %r : tensor<i8>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xi8>> -> tensor<4xi8>
}

// -----

// D15 at n = 4: the same reduce-then-permute shape, scaled up. S = 4.
//   half-split (NOREL): all-reduce mesh0 (halving+doubling, 2k = 4 rounds,
//     V = 2 * 4 * 3 / 4 = 6, latency 0.1 + 0.04 = 0.14, duration 6.14) +
//     all-gather mesh1 (k = 2, V = 4 * 3 = 12, latency 0.12, duration 12.12) +
//     free slice = 18.26
//   path closure (CHECK): all-reduce mesh0 (6.14, as above) + one permute,
//     V = S * (1 - 1/4) = 3 on each axis, latency 0.12, duration 3.12 = 9.26
// CHECK: chain: 2 steps, total duration 9.26
// CHECK-NEXT: step 0: all-reduce
// CHECK-NEXT: atoms: axis0.0(x4)
// CHECK-NEXT: payload: 4 -> 4
// CHECK-NEXT: latency: 0.14
// CHECK-NEXT: V: [6, 0]
// CHECK-NEXT: rho: [1, 0]
// CHECK-NEXT: duration: 6.14
// CHECK-NEXT: step 1: permute
// CHECK-NEXT: atoms: axis0.0(x4) axis1.0(x4)
// CHECK-NEXT: payload: 4 -> 4
// CHECK-NEXT: latency: 0.12
// CHECK-NEXT: V: [3, 3]
// CHECK-NEXT: rho: [1, 1]
// CHECK-NEXT: duration: 3.12
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<4xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<4xi8> (the await result, 0 uses) <- step 1
// NOREL: chain: 3 steps, total duration 18.26
// NOREL-NEXT: step 0: all-reduce
// NOREL-NEXT: atoms: axis0.0(x4)
// NOREL-NEXT: payload: 4 -> 4
// NOREL-NEXT: latency: 0.14
// NOREL-NEXT: V: [6, 0]
// NOREL-NEXT: rho: [1, 0]
// NOREL-NEXT: duration: 6.14
// NOREL-NEXT: step 1: all-gather
// NOREL-NEXT: atoms: axis1.0(x4)
// NOREL-NEXT: payload: 4 -> 16
// NOREL-NEXT: latency: 0.12
// NOREL-NEXT: V: [0, 12]
// NOREL-NEXT: rho: [0, 1]
// NOREL-NEXT: duration: 12.12
// NOREL-NEXT: step 2: local-slice
// NOREL-NEXT: atoms: axis0.0(x4)
// NOREL-NEXT: payload: 16 -> 4
// NOREL-NEXT: latency: 0
// NOREL-NEXT: V: [0, 0]
// NOREL-NEXT: rho: [0, 0]
// NOREL-NEXT: duration: 0
// NOREL-NEXT: semantics: verified
// NOREL-NEXT: input port: tensor<4xi8> (the collective's input_object operand) -> step 0
// NOREL-NEXT: result port: tensor<4xi8> (the await result, 0 uses) <- step 2
module @d15_reduce_then_permute_n4 {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 4>, !distributed.physical_comm_axis<4, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 4>, !distributed.physical_comm_axis<4, 1>
  %rep4 = distributed.ReplicationAxis 4 : !distributed.replication_axis<4>
  %ti0 = axis.getaxis tensor<4xi8> 0
  %to0 = axis.getaxis tensor<4xi8> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %f4 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %mesh_out = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>, %f4 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f5 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %red0 = axis.product (%f5 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>)
  %f6 = axis.factor %ti0 : !axis.shape_axis<tensor<4xi8>, 0> <4, 1>
  %lhs0 = axis.product (%f6 : !axis.axis_factor<!axis.shape_axis<tensor<4xi8>, 0>, 4, 1>)
  %f7 = axis.factor %to0 : !axis.shape_axis<tensor<4xi8>, 0> <4, 1>
  %rhs0 = axis.product (%f7 : !axis.axis_factor<!axis.shape_axis<tensor<4xi8>, 0>, 4, 1>)
  %f8 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %lhs1 = axis.product (%f8 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f9 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %rhs1 = axis.product (%f9 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>)
  %f10 = axis.factor %rep4 : !distributed.replication_axis<4> <4, 1>
  %lhs2 = axis.product (%f10 : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
  %f11 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %rhs2 = axis.product (%f11 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %map = axis.map %lhs0, %lhs1, %lhs2 to %rhs0, %rhs1, %rhs2 : [!axis.factor_group<4>, !axis.factor_group<4>, !axis.factor_group<4>] [!axis.factor_group<4>, !axis.factor_group<4>, !axis.factor_group<4>]

  %in = tensor.empty() : tensor<4xi8>
  %h = distributed.Collective %in : tensor<4xi8> on %mesh_in : !axis.factor_group<16> to tensor<4xi8> on %mesh_out : !axis.factor_group<16> reduces (%red0 : !axis.factor_group<4>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<i8>, %rhs_v: tensor<i8>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<i8>
    stablehlo.return %r : tensor<i8>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xi8>> -> tensor<4xi8>
}

// -----

// D15 with both ends free: mesh0 (in = replicate, out = mesh1's digit) is the
// end of the path, mesh1 (in = mesh0's digit, out = replicate) is the start.
// Neither end needs draining (mesh0's input carries nothing to gather away)
// nor a trailing slice (mesh1's output is not a tile digit), so closure is
// just one permute; no tile atoms are touched by the mesh at all, so the
// payload is carried on a plain identity pair. n = 2, S = 2.
//   half-split (NOREL): mesh0 has no first half (in = replicate); its free
//     slice waits for mesh1's all-gather (V = 2, 2.11). mesh1 has no second
//     half (out = replicate). Total 2.11.
//   path closure (CHECK): one permute, V = S * (1 - 1/2) = 1 on each axis,
//     latency 0.12, duration 1.12.
// CHECK: chain: 1 steps, total duration 1.12
// CHECK-NEXT: step 0: permute
// CHECK-NEXT: atoms: axis0.0(x2) axis1.0(x2)
// CHECK-NEXT: payload: 2 -> 2
// CHECK-NEXT: latency: 0.12
// CHECK-NEXT: V: [1, 1]
// CHECK-NEXT: rho: [1, 1]
// CHECK-NEXT: duration: 1.12
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<2xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<2xi8> (the await result, 0 uses) <- step 0
// NOREL: chain: 2 steps, total duration 2.11
// NOREL-NEXT: step 0: all-gather
// NOREL-NEXT: atoms: axis1.0(x2)
// NOREL-NEXT: payload: 2 -> 4
// NOREL-NEXT: latency: 0.11
// NOREL-NEXT: V: [0, 2]
// NOREL-NEXT: rho: [0, 1]
// NOREL-NEXT: duration: 2.11
// NOREL-NEXT: step 1: local-slice
// NOREL-NEXT: atoms: axis0.0(x2)
// NOREL-NEXT: payload: 4 -> 2
// NOREL-NEXT: latency: 0
// NOREL-NEXT: V: [0, 0]
// NOREL-NEXT: rho: [0, 0]
// NOREL-NEXT: duration: 0
// NOREL-NEXT: semantics: verified
// NOREL-NEXT: input port: tensor<2xi8> (the collective's input_object operand) -> step 0
// NOREL-NEXT: result port: tensor<2xi8> (the await result, 0 uses) <- step 1
module @d15_replicate_mesh_path {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %rep2a = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %rep2b = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<2xi8> 0
  %to0 = axis.getaxis tensor<2xi8> 0

  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f4 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f4 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)

  // Tile dim0 passes straight through, untouched by the mesh.
  %f5 = axis.factor %ti0 : !axis.shape_axis<tensor<2xi8>, 0> <2, 1>
  %lhs0 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<2xi8>, 0>, 2, 1>)
  %f6 = axis.factor %to0 : !axis.shape_axis<tensor<2xi8>, 0> <2, 1>
  %rhs0 = axis.product (%f6 : !axis.axis_factor<!axis.shape_axis<tensor<2xi8>, 0>, 2, 1>)

  // mesh0 (in = replicate): the end of the path, drained by nothing.
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %lhs1 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f8 = axis.factor %rep2a : !distributed.replication_axis<2> <2, 1>
  %rhs1 = axis.product (%f8 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)

  // mesh1 -> mesh0: mesh1's digit becomes mesh0's final digit.
  %f9 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs2 = axis.product (%f9 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f10 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %rhs2 = axis.product (%f10 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)

  // mesh1 (out = replicate): the start of the path, fed by nothing.
  %f11 = axis.factor %rep2b : !distributed.replication_axis<2> <2, 1>
  %lhs3 = axis.product (%f11 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f12 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs3 = axis.product (%f12 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)

  %map = axis.map %lhs0, %lhs1, %lhs2, %lhs3 to %rhs0, %rhs1, %rhs2, %rhs3 : [!axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<2xi8>
  %h = distributed.Collective %in : tensor<2xi8> on %mesh_in : !axis.factor_group<4> to tensor<2xi8> on %mesh_out : !axis.factor_group<4> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<2xi8>> -> tensor<2xi8>
}

// -----

// D16: conjugation. mesh0 is a plain all-gather (Tile, Replicate) on axis0;
// mesh1 has no role at all (Replicate, Replicate), so it is an idle atom of
// equal extent D16 can borrow. n = 2, S = 2.
//   CHECK is bandwidth 1 on both axes (the default): no axis has strictly
//     more bandwidth than another, so conjugationTarget finds nothing and the
//     gather runs on mesh0 unchanged, the same 2.11 as disabling the relay
//     variants entirely. This is the "uniform bandwidth never conjugates" case.
//   FAST gives axis1 four times axis0's bandwidth: a swap permute moves
//     mesh0's digit onto mesh1 (V = S * (1 - 1/2) = 1 on each axis, latency
//     0.12, transferTime = max(1 / 1, 1 / 4) = 1, duration 1.12), then the
//     gather runs on the fast axis1 (V = 2, latency 0.11, transferTime =
//     2 / 4 = 0.5, duration 0.61). Total 1.73, cheaper than FASTNOREL's 2.11.
// CHECK: chain: 1 steps, total duration 2.11
// CHECK-NEXT: step 0: all-gather
// CHECK-NEXT: atoms: axis0.0(x2)
// CHECK-NEXT: payload: 2 -> 4
// CHECK-NEXT: latency: 0.11
// CHECK-NEXT: V: [2, 0]
// CHECK-NEXT: rho: [1, 0]
// CHECK-NEXT: duration: 2.11
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<2xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<4xi8> (the await result, 0 uses) <- step 0
// FAST: chain: 2 steps, total duration 1.73
// FAST-NEXT: step 0: permute
// FAST-NEXT: atoms: axis0.0(x2) axis1.0(x2)
// FAST-NEXT: payload: 2 -> 2
// FAST-NEXT: latency: 0.12
// FAST-NEXT: V: [1, 1]
// FAST-NEXT: rho: [1, 0.25]
// FAST-NEXT: duration: 1.12
// FAST-NEXT: step 1: all-gather
// FAST-NEXT: atoms: axis1.0(x2)
// FAST-NEXT: payload: 2 -> 4
// FAST-NEXT: latency: 0.11
// FAST-NEXT: V: [0, 2]
// FAST-NEXT: rho: [0, 1]
// FAST-NEXT: duration: 0.61
// FAST-NEXT: semantics: verified
// FAST-NEXT: input port: tensor<2xi8> (the collective's input_object operand) -> step 0
// FAST-NEXT: result port: tensor<4xi8> (the await result, 0 uses) <- step 1
// FASTNOREL: chain: 1 steps, total duration 2.11
// FASTNOREL-NEXT: step 0: all-gather
// FASTNOREL-NEXT: atoms: axis0.0(x2)
// FASTNOREL-NEXT: payload: 2 -> 4
// FASTNOREL-NEXT: latency: 0.11
// FASTNOREL-NEXT: V: [2, 0]
// FASTNOREL-NEXT: rho: [1, 0]
// FASTNOREL-NEXT: duration: 2.11
// FASTNOREL-NEXT: semantics: verified
// FASTNOREL-NEXT: input port: tensor<2xi8> (the collective's input_object operand) -> step 0
// FASTNOREL-NEXT: result port: tensor<4xi8> (the await result, 0 uses) <- step 0
module @d16_allgather_conjugate {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>, !distributed.physical_comm_axis<2, 2>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>, !distributed.physical_comm_axis<2, 2>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %rep2b = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %rep2c = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<2xi8> 0
  %to0 = axis.getaxis tensor<4xi8> 0

  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f1b = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %f1b : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)

  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f2b = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %mesh_out = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %f2b : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)

  // mesh0 (in = tile, out = replicate): the all-gather.
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f4 = axis.factor %ti0 : !axis.shape_axis<tensor<2xi8>, 0> <2, 1>
  %lhs0 = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %f4 : !axis.axis_factor<!axis.shape_axis<tensor<2xi8>, 0>, 2, 1>)
  %f5 = axis.factor %to0 : !axis.shape_axis<tensor<4xi8>, 0> <4, 1>
  %rhs0 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<4xi8>, 0>, 4, 1>)
  %f6 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %lhs1 = axis.product (%f6 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs1 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)

  // mesh1 (in = replicate, out = replicate): idle, a D16 conjugation target.
  %f8 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %lhs2 = axis.product (%f8 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f9 = axis.factor %rep2b : !distributed.replication_axis<2> <2, 1>
  %rhs2 = axis.product (%f9 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f10 = axis.factor %rep2c : !distributed.replication_axis<2> <2, 1>
  %lhs3 = axis.product (%f10 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f11 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %rhs3 = axis.product (%f11 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)

  %map = axis.map %lhs0, %lhs1, %lhs2, %lhs3 to %rhs0, %rhs1, %rhs2, %rhs3 : [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<2xi8>
  %h = distributed.Collective %in : tensor<2xi8> on %mesh_in : !axis.factor_group<4> to tensor<4xi8> on %mesh_out : !axis.factor_group<4> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xi8>> -> tensor<4xi8>
}

// -----

// D16 correctly skipped: two independent all-gathers, one per tensor dim, on
// axis0 (slow) and axis1 (fast). Unlike the module above, axis1's own atom
// already has a role (it is itself an all-gather, not idle), so it fails
// conjugationTarget's "holds no digit and has no role" test and cannot be
// axis0's target: FAST and FASTNOREL agree.
// FAST: chain: 2 steps, total duration 6.22
// FAST-NEXT: step 0: all-gather
// FAST-NEXT: atoms: axis0.0(x2)
// FAST-NEXT: payload: 4 -> 8
// FAST-NEXT: latency: 0.11
// FAST-NEXT: V: [4, 0]
// FAST-NEXT: rho: [1, 0]
// FAST-NEXT: duration: 4.11
// FAST-NEXT: step 1: all-gather
// FAST-NEXT: atoms: axis1.0(x2)
// FAST-NEXT: payload: 8 -> 16
// FAST-NEXT: latency: 0.11
// FAST-NEXT: V: [0, 8]
// FAST-NEXT: rho: [0, 1]
// FAST-NEXT: duration: 2.11
// FAST-NEXT: semantics: verified
// FAST-NEXT: input port: tensor<2x2xi8> (the collective's input_object operand) -> step 0
// FAST-NEXT: result port: tensor<4x4xi8> (the await result, 0 uses) <- step 1
// FASTNOREL: chain: 2 steps, total duration 6.22
// FASTNOREL-NEXT: step 0: all-gather
// FASTNOREL-NEXT: atoms: axis0.0(x2)
// FASTNOREL-NEXT: payload: 4 -> 8
// FASTNOREL-NEXT: latency: 0.11
// FASTNOREL-NEXT: V: [4, 0]
// FASTNOREL-NEXT: rho: [1, 0]
// FASTNOREL-NEXT: duration: 4.11
// FASTNOREL-NEXT: step 1: all-gather
// FASTNOREL-NEXT: atoms: axis1.0(x2)
// FASTNOREL-NEXT: payload: 8 -> 16
// FASTNOREL-NEXT: latency: 0.11
// FASTNOREL-NEXT: V: [0, 8]
// FASTNOREL-NEXT: rho: [0, 1]
// FASTNOREL-NEXT: duration: 2.11
// FASTNOREL-NEXT: semantics: verified
// FASTNOREL-NEXT: input port: tensor<2x2xi8> (the collective's input_object operand) -> step 0
// FASTNOREL-NEXT: result port: tensor<4x4xi8> (the await result, 0 uses) <- step 1
module @d16_no_conjugation_target {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>, !distributed.physical_comm_axis<2, 2>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>, !distributed.physical_comm_axis<2, 2>
  %rep2a = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %rep2b = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<2x2xi8> 0
  %ti1 = axis.getaxis tensor<2x2xi8> 1
  %to0 = axis.getaxis tensor<4x4xi8> 0
  %to1 = axis.getaxis tensor<4x4xi8> 1

  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f1b = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %f1b : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)

  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f2b = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %mesh_out = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %f2b : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)

  // dim0, gathered by axis0 (slow, bandwidth 1).
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f4 = axis.factor %ti0 : !axis.shape_axis<tensor<2x2xi8>, 0> <2, 1>
  %lhs0 = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %f4 : !axis.axis_factor<!axis.shape_axis<tensor<2x2xi8>, 0>, 2, 1>)
  %f5 = axis.factor %to0 : !axis.shape_axis<tensor<4x4xi8>, 0> <4, 1>
  %rhs0 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<4x4xi8>, 0>, 4, 1>)
  %f6 = axis.factor %rep2a : !distributed.replication_axis<2> <2, 1>
  %lhs1 = axis.product (%f6 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs1 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)

  // dim1, gathered by axis1 (fast, bandwidth 4): a separate all-gather,
  // unrelated to axis0's, so axis1 already has a role and cannot be axis0's
  // D16 target.
  %f8 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f9 = axis.factor %ti1 : !axis.shape_axis<tensor<2x2xi8>, 1> <2, 1>
  %lhs2 = axis.product (%f8 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f9 : !axis.axis_factor<!axis.shape_axis<tensor<2x2xi8>, 1>, 2, 1>)
  %f10 = axis.factor %to1 : !axis.shape_axis<tensor<4x4xi8>, 1> <4, 1>
  %rhs2 = axis.product (%f10 : !axis.axis_factor<!axis.shape_axis<tensor<4x4xi8>, 1>, 4, 1>)
  %f11 = axis.factor %rep2b : !distributed.replication_axis<2> <2, 1>
  %lhs3 = axis.product (%f11 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f12 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %rhs3 = axis.product (%f12 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)

  %map = axis.map %lhs0, %lhs1, %lhs2, %lhs3 to %rhs0, %rhs1, %rhs2, %rhs3 : [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<2x2xi8>
  %h = distributed.Collective %in : tensor<2x2xi8> on %mesh_in : !axis.factor_group<4> to tensor<4x4xi8> on %mesh_out : !axis.factor_group<4> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4x4xi8>> -> tensor<4x4xi8>
}
