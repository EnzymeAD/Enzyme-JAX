// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan=chain=true %s 2>&1 >/dev/null | FileCheck %s
// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true disable-relay-variants=true" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOREL
// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true bandwidths=1,4" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=FAST
// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true bandwidths=1,4 disable-relay-variants=true" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=FASTNOREL

// The relay variants (D14 to D16) mesh-coupled components are offered besides
// the half-split (see decompose_collective_meshcoupled.mlir, which pins the
// half-split with disable-relay-variants=true). CHECK/NOREL use the default
// parameters (bandwidth 50 bytes/ns, round latency 1000ns, launch latency
// 5000ns); FAST/FASTNOREL give axis1 four times axis0's bandwidth (still
// bytes/ns; both are far slower than the default, an explicit scenario
// unrelated to it). D16 (conjugation) is exercised below, but a permute's
// cost is floored by the slowest axis it crosses (a moving device sends its
// whole payload there too), so moving a digit off axis0 through a swap
// permute is never cheaper than gathering it on axis0 directly; see
// d16_allgather_conjugate below. duration = latency + round(V / BW)
// (CollectiveCost.h's Units note); at these tiny payloads that rounded term
// is usually 0 or a small remainder.

// D14: fused exchange. mesh0 and mesh1 each both release their own digit (in
// is Tile or Mesh) and place another (out is Tile or Mesh), the shape a
// TileToTile or ReduceScatter unit fuses into one all-to-all instead of a
// gather followed by a free slice. S = 2, n = 2; every transfer time here
// rounds to 0, so both shapes cost the same and CHECK/NOREL only differ in
// which chain shape the DP happens to try first.
//   half-split (NOREL): all-gather mesh0 (6000) + slice + all-gather
//     mesh1 (6000) + slice = 12000
//   fused exchange (CHECK): all-to-all mesh0 (6000) + all-to-all mesh1
//     (6000) = 12000
// CHECK: chain: 2 steps, total duration 12000
// CHECK-NEXT: step 0: all-to-all
// CHECK-NEXT: atoms: axis0.0(x2)
// CHECK-NEXT: payload: 2 -> 2
// CHECK-NEXT: latency: 6000
// CHECK-NEXT: V: [1, 0]
// CHECK-NEXT: rho: [1, 0]
// CHECK-NEXT: duration: 6000
// CHECK-NEXT: step 1: all-to-all
// CHECK-NEXT: atoms: axis1.0(x2)
// CHECK-NEXT: payload: 2 -> 2
// CHECK-NEXT: latency: 6000
// CHECK-NEXT: V: [0, 1]
// CHECK-NEXT: rho: [0, 1]
// CHECK-NEXT: duration: 6000
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<2xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<2xi8> (the await result, 0 uses) <- step 1
// NOREL: chain: 4 steps, total duration 12000
// NOREL-NEXT: step 0: all-gather
// NOREL-NEXT: atoms: axis0.0(x2)
// NOREL-NEXT: payload: 2 -> 4
// NOREL-NEXT: latency: 6000
// NOREL-NEXT: V: [2, 0]
// NOREL-NEXT: rho: [1, 0]
// NOREL-NEXT: duration: 6000
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
// NOREL-NEXT: latency: 6000
// NOREL-NEXT: V: [0, 2]
// NOREL-NEXT: rho: [0, 1]
// NOREL-NEXT: duration: 6000
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
// cycle. n = 2, S = 4; every transfer time here rounds to 0 or a small
// remainder.
//   half-split (NOREL, default bandwidth): all-reduce mesh0 (6000) +
//     all-gather mesh1 (6000) + free slice = 12000
//   path closure (a moving device sends its whole payload, so the permute is
//     no cheaper than gathering the same bytes): all-reduce mesh0 empties it
//     (6000), then one permute of both atoms crossing two round latencies
//     (7000). Total 13000, more than the half split, so CHECK (default
//     bandwidth) takes the half split.
//   at FAST (axis0 kept at bandwidth 1, axis1 four times that): the
//     all-reduce still runs on slow axis0, so its own transfer time (V = 4)
//     no longer rounds to 0 (duration 6004); the gather moves to the fast
//     axis1 (duration 6001), for a half-split total of 12005. The permute
//     also crosses slow axis0, so its own transfer time stops rounding to 0
//     too (duration 7004): closure costs 13008, so the half split still
//     wins.
// CHECK: chain: 3 steps, total duration 12000
// CHECK-NEXT: step 0: all-reduce
// CHECK-NEXT: atoms: axis0.0(x2)
// CHECK-NEXT: payload: 4 -> 4
// CHECK-NEXT: latency: 6000
// CHECK-NEXT: V: [4, 0]
// CHECK-NEXT: rho: [1, 0]
// CHECK-NEXT: duration: 6000
// CHECK-NEXT: step 1: all-gather
// CHECK-NEXT: atoms: axis1.0(x2)
// CHECK-NEXT: payload: 4 -> 8
// CHECK-NEXT: latency: 6000
// CHECK-NEXT: V: [0, 4]
// CHECK-NEXT: rho: [0, 1]
// CHECK-NEXT: duration: 6000
// CHECK-NEXT: step 2: local-slice
// CHECK-NEXT: atoms: axis0.0(x2)
// CHECK-NEXT: payload: 8 -> 4
// CHECK-NEXT: latency: 0
// CHECK-NEXT: V: [0, 0]
// CHECK-NEXT: rho: [0, 0]
// CHECK-NEXT: duration: 0
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<4xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<4xi8> (the await result, 0 uses) <- step 2
// NOREL: chain: 3 steps, total duration 12000
// NOREL-NEXT: step 0: all-reduce
// NOREL-NEXT: atoms: axis0.0(x2)
// NOREL-NEXT: payload: 4 -> 4
// NOREL-NEXT: latency: 6000
// NOREL-NEXT: V: [4, 0]
// NOREL-NEXT: rho: [1, 0]
// NOREL-NEXT: duration: 6000
// NOREL-NEXT: step 1: all-gather
// NOREL-NEXT: atoms: axis1.0(x2)
// NOREL-NEXT: payload: 4 -> 8
// NOREL-NEXT: latency: 6000
// NOREL-NEXT: V: [0, 4]
// NOREL-NEXT: rho: [0, 1]
// NOREL-NEXT: duration: 6000
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
// FAST: chain: 3 steps, total duration 12005
// FAST-NEXT: step 0: all-reduce
// FAST-NEXT: atoms: axis0.0(x2)
// FAST-NEXT: payload: 4 -> 4
// FAST-NEXT: latency: 6000
// FAST-NEXT: V: [4, 0]
// FAST-NEXT: rho: [1, 0]
// FAST-NEXT: duration: 6004
// FAST-NEXT: step 1: all-gather
// FAST-NEXT: atoms: axis1.0(x2)
// FAST-NEXT: payload: 4 -> 8
// FAST-NEXT: latency: 6000
// FAST-NEXT: V: [0, 4]
// FAST-NEXT: rho: [0, 1]
// FAST-NEXT: duration: 6001
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
// FASTNOREL: chain: 3 steps, total duration 12005
// FASTNOREL-NEXT: step 0: all-reduce
// FASTNOREL-NEXT: atoms: axis0.0(x2)
// FASTNOREL-NEXT: payload: 4 -> 4
// FASTNOREL-NEXT: latency: 6000
// FASTNOREL-NEXT: V: [4, 0]
// FASTNOREL-NEXT: rho: [1, 0]
// FASTNOREL-NEXT: duration: 6004
// FASTNOREL-NEXT: step 1: all-gather
// FASTNOREL-NEXT: atoms: axis1.0(x2)
// FASTNOREL-NEXT: payload: 4 -> 8
// FASTNOREL-NEXT: latency: 6000
// FASTNOREL-NEXT: V: [0, 4]
// FASTNOREL-NEXT: rho: [0, 1]
// FASTNOREL-NEXT: duration: 6001
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

// D15 at n = 4: the same reduce-then-permute shape, scaled up. S = 4; every
// transfer time here rounds to 0.
//   half-split: all-reduce mesh0 (recursive doubling needs k = 2 rounds
//     against halving+doubling's 2k = 4, so it wins on rounds alone: latency
//     5000 + 2000 = 7000) + all-gather mesh1 (k = 2, latency 7000) + free
//     slice = 14000
//   path closure: all-reduce mesh0 (7000, as above) + one permute crossing
//     two round latencies (latency 7000) = 14000, tying the half split; the
//     DP's tie-break here keeps the half split, so CHECK and NOREL agree.
// CHECK: chain: 3 steps, total duration 14000
// CHECK-NEXT: step 0: all-reduce
// CHECK-NEXT: atoms: axis0.0(x4)
// CHECK-NEXT: payload: 4 -> 4
// CHECK-NEXT: latency: 7000
// CHECK-NEXT: V: [8, 0]
// CHECK-NEXT: rho: [1, 0]
// CHECK-NEXT: duration: 7000
// CHECK-NEXT: step 1: all-gather
// CHECK-NEXT: atoms: axis1.0(x4)
// CHECK-NEXT: payload: 4 -> 16
// CHECK-NEXT: latency: 7000
// CHECK-NEXT: V: [0, 12]
// CHECK-NEXT: rho: [0, 1]
// CHECK-NEXT: duration: 7000
// CHECK-NEXT: step 2: local-slice
// CHECK-NEXT: atoms: axis0.0(x4)
// CHECK-NEXT: payload: 16 -> 4
// CHECK-NEXT: latency: 0
// CHECK-NEXT: V: [0, 0]
// CHECK-NEXT: rho: [0, 0]
// CHECK-NEXT: duration: 0
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<4xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<4xi8> (the await result, 0 uses) <- step 2
// NOREL: chain: 3 steps, total duration 14000
// NOREL-NEXT: step 0: all-reduce
// NOREL-NEXT: atoms: axis0.0(x4)
// NOREL-NEXT: payload: 4 -> 4
// NOREL-NEXT: latency: 7000
// NOREL-NEXT: V: [8, 0]
// NOREL-NEXT: rho: [1, 0]
// NOREL-NEXT: duration: 7000
// NOREL-NEXT: step 1: all-gather
// NOREL-NEXT: atoms: axis1.0(x4)
// NOREL-NEXT: payload: 4 -> 16
// NOREL-NEXT: latency: 7000
// NOREL-NEXT: V: [0, 12]
// NOREL-NEXT: rho: [0, 1]
// NOREL-NEXT: duration: 7000
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
//     slice waits for mesh1's all-gather (6000). mesh1 has no second half
//     (out = replicate). Total 6000.
//   path closure: one permute crossing two round latencies, duration 7000,
//     more than the half split, so CHECK (default bandwidth) also takes the
//     half split.
// CHECK: chain: 2 steps, total duration 6000
// CHECK-NEXT: step 0: all-gather
// CHECK-NEXT: atoms: axis1.0(x2)
// CHECK-NEXT: payload: 2 -> 4
// CHECK-NEXT: latency: 6000
// CHECK-NEXT: V: [0, 2]
// CHECK-NEXT: rho: [0, 1]
// CHECK-NEXT: duration: 6000
// CHECK-NEXT: step 1: local-slice
// CHECK-NEXT: atoms: axis0.0(x2)
// CHECK-NEXT: payload: 4 -> 2
// CHECK-NEXT: latency: 0
// CHECK-NEXT: V: [0, 0]
// CHECK-NEXT: rho: [0, 0]
// CHECK-NEXT: duration: 0
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<2xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<2xi8> (the await result, 0 uses) <- step 1
// NOREL: chain: 2 steps, total duration 6000
// NOREL-NEXT: step 0: all-gather
// NOREL-NEXT: atoms: axis1.0(x2)
// NOREL-NEXT: payload: 2 -> 4
// NOREL-NEXT: latency: 6000
// NOREL-NEXT: V: [0, 2]
// NOREL-NEXT: rho: [0, 1]
// NOREL-NEXT: duration: 6000
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
//   CHECK uses the default bandwidth, uniform 50 bytes/ns on both axes: no
//     axis has strictly more bandwidth than another, so conjugationTarget
//     finds nothing and the gather runs on mesh0 unchanged, the same 6000 as
//     disabling the relay variants entirely. This is the "uniform bandwidth
//     never conjugates" case.
//   FAST gives axis1 four times axis0's bandwidth: a swap permute could move
//     mesh0's digit onto mesh1, but a moving device sends its whole payload
//     (V = 2 on each axis, transferTime = max(2 / 1, 2 / 4) = 2, rounds to 0,
//     duration 7000) before the gather could even run on the fast axis1
//     (6001), so conjugation (13001) no longer beats running the gather
//     where it is (6002): FAST agrees with FASTNOREL here.
// CHECK: chain: 1 steps, total duration 6000
// CHECK-NEXT: step 0: all-gather
// CHECK-NEXT: atoms: axis0.0(x2)
// CHECK-NEXT: payload: 2 -> 4
// CHECK-NEXT: latency: 6000
// CHECK-NEXT: V: [2, 0]
// CHECK-NEXT: rho: [1, 0]
// CHECK-NEXT: duration: 6000
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<2xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<4xi8> (the await result, 0 uses) <- step 0
// FAST: chain: 1 steps, total duration 6002
// FAST-NEXT: step 0: all-gather
// FAST-NEXT: atoms: axis0.0(x2)
// FAST-NEXT: payload: 2 -> 4
// FAST-NEXT: latency: 6000
// FAST-NEXT: V: [2, 0]
// FAST-NEXT: rho: [1, 0]
// FAST-NEXT: duration: 6002
// FAST-NEXT: semantics: verified
// FAST-NEXT: input port: tensor<2xi8> (the collective's input_object operand) -> step 0
// FAST-NEXT: result port: tensor<4xi8> (the await result, 0 uses) <- step 0
// FASTNOREL: chain: 1 steps, total duration 6002
// FASTNOREL-NEXT: step 0: all-gather
// FASTNOREL-NEXT: atoms: axis0.0(x2)
// FASTNOREL-NEXT: payload: 2 -> 4
// FASTNOREL-NEXT: latency: 6000
// FASTNOREL-NEXT: V: [2, 0]
// FASTNOREL-NEXT: rho: [1, 0]
// FASTNOREL-NEXT: duration: 6002
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
// FAST: chain: 2 steps, total duration 12006
// FAST-NEXT: step 0: all-gather
// FAST-NEXT: atoms: axis0.0(x2)
// FAST-NEXT: payload: 4 -> 8
// FAST-NEXT: latency: 6000
// FAST-NEXT: V: [4, 0]
// FAST-NEXT: rho: [1, 0]
// FAST-NEXT: duration: 6004
// FAST-NEXT: step 1: all-gather
// FAST-NEXT: atoms: axis1.0(x2)
// FAST-NEXT: payload: 8 -> 16
// FAST-NEXT: latency: 6000
// FAST-NEXT: V: [0, 8]
// FAST-NEXT: rho: [0, 1]
// FAST-NEXT: duration: 6002
// FAST-NEXT: semantics: verified
// FAST-NEXT: input port: tensor<2x2xi8> (the collective's input_object operand) -> step 0
// FAST-NEXT: result port: tensor<4x4xi8> (the await result, 0 uses) <- step 1
// FASTNOREL: chain: 2 steps, total duration 12006
// FASTNOREL-NEXT: step 0: all-gather
// FASTNOREL-NEXT: atoms: axis0.0(x2)
// FASTNOREL-NEXT: payload: 4 -> 8
// FASTNOREL-NEXT: latency: 6000
// FASTNOREL-NEXT: V: [4, 0]
// FASTNOREL-NEXT: rho: [1, 0]
// FASTNOREL-NEXT: duration: 6004
// FASTNOREL-NEXT: step 1: all-gather
// FASTNOREL-NEXT: atoms: axis1.0(x2)
// FASTNOREL-NEXT: payload: 8 -> 16
// FASTNOREL-NEXT: latency: 6000
// FASTNOREL-NEXT: V: [0, 8]
// FASTNOREL-NEXT: rho: [0, 1]
// FASTNOREL-NEXT: duration: 6002
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
