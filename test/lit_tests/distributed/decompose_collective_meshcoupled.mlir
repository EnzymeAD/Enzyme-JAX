// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true disable-relay-variants=true" %s 2>&1 >/dev/null | FileCheck %s

// Mesh-coupled rows with the relay variants disabled (fused exchanges, path
// closures and conjugations; see decompose_collective_relay.mlir), which pins
// the half-split baseline that every collective supports. Atoms whose in or out
// role is another mesh atom form components. The half-split turns every atom
// (in = P, out = Q) of such a component into a first half by P (all-reduce for
// a reduced atom, all-gather for a tile or mesh digit, nothing for a replicate)
// and a free local slice by Q (the input tile digit, or the digit that came
// from the partner atom). A slice waits for its own atom's first half and for
// the partner's first half. A component of pure (Mesh, Mesh) atoms stays one
// permute step (see decompose_collective.mlir).
//
// Hand computation uses the default parameters (bandwidth 50 bytes/ns, round
// latency 1000ns, launch latency 5000ns) and the formulas at the top of
// decompose_collective.mlir. With group size n, k = ceil(log2 n) and payload S:
//   all-gather:  V = S (n - 1), latency 5000 + 1000 k, payload S -> n S
//   all-reduce:  the cheaper of halving + doubling (V = 2 S (n - 1) / n, latency
//                5000 + 2000 k) and recursive doubling (V = S k, latency 5000 + 1000 k)
//   permute:     V[a] = S on every axis with a nonzero change fraction (a
//                moving device sends its whole payload), latency
//                5000 + 1000 x (axes crossed)
//   slice:       free, payload divides by the extent
// duration = latency + round(V / BW) (CollectiveCost.h's Units note): at these
// tiny payloads that rounded term is usually 0 or 1, so most decisions below
// turn on round count rather than volume. Element type i8, so a payload in
// bytes equals its element count.

// Reduce then permute, larger n. mesh0 is reduced and its output comes from mesh1 (in mesh, out replicate), so the atoms form the open path mesh1 -> mesh0 rather than a permutation cycle and both are half-split: all-reduce mesh0, all-gather mesh1, then a free slice onto mesh0 of the digit mesh1 held. n = 4 (k = 2), S = 8; every candidate's transfer time rounds to 0, so the all-reduce turns on round count alone.
//   all-reduce mesh0: recursive doubling needs k = 2 rounds (latency 5000 + 2000 = 7000); halving + doubling needs 2k = 4 rounds (latency 9000) despite its smaller volume coefficient, so recursive doubling wins, V = 8 * 2 = 16, duration 7000
//   all-gather mesh1: V = 8 * 3 = 24, latency 7000, duration 7000, payload 8 -> 32
//   slice onto mesh0: 32 -> 8, free
// Total 7000 + 7000 = 14000.
// CHECK: chain: 3 steps, total duration 14000
// CHECK-NEXT: step 0: all-reduce
// CHECK-NEXT: atoms: axis0.0(x4)
// CHECK-NEXT: payload: 8 -> 8
// CHECK-NEXT: latency: 7000
// CHECK-NEXT: V: [16, 0]
// CHECK-NEXT: rho: [1, 0]
// CHECK-NEXT: duration: 7000
// CHECK-NEXT: step 1: all-gather
// CHECK-NEXT: atoms: axis1.0(x4)
// CHECK-NEXT: payload: 8 -> 32
// CHECK-NEXT: latency: 7000
// CHECK-NEXT: V: [0, 24]
// CHECK-NEXT: rho: [0, 1]
// CHECK-NEXT: duration: 7000
// CHECK-NEXT: step 2: local-slice
// CHECK-NEXT: atoms: axis0.0(x4)
// CHECK-NEXT: payload: 32 -> 8
// CHECK-NEXT: latency: 0
// CHECK-NEXT: V: [0, 0]
// CHECK-NEXT: rho: [0, 0]
// CHECK-NEXT: duration: 0
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<8xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<8xi8> (the await result, 0 uses) <- step 2
module @reduce_then_permute_n4 {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 4>, !distributed.physical_comm_axis<4, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 4>, !distributed.physical_comm_axis<4, 1>
  %rep2 = distributed.ReplicationAxis 4 : !distributed.replication_axis<4>
  %ti0 = axis.getaxis tensor<8xi8> 0
  %to0 = axis.getaxis tensor<8xi8> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %f4 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %mesh_out = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>, %f4 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f5 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %red0 = axis.product (%f5 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>)
  %f6 = axis.factor %ti0 : !axis.shape_axis<tensor<8xi8>, 0> <8, 1>
  %lhs0 = axis.product (%f6 : !axis.axis_factor<!axis.shape_axis<tensor<8xi8>, 0>, 8, 1>)
  %f7 = axis.factor %to0 : !axis.shape_axis<tensor<8xi8>, 0> <8, 1>
  %rhs0 = axis.product (%f7 : !axis.axis_factor<!axis.shape_axis<tensor<8xi8>, 0>, 8, 1>)
  %f8 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %lhs1 = axis.product (%f8 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f9 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %rhs1 = axis.product (%f9 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>)
  %f10 = axis.factor %rep2 : !distributed.replication_axis<4> <4, 1>
  %lhs2 = axis.product (%f10 : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
  %f11 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %rhs2 = axis.product (%f11 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %map = axis.map %lhs0, %lhs1, %lhs2 to %rhs0, %rhs1, %rhs2 : [!axis.factor_group<8>, !axis.factor_group<4>, !axis.factor_group<4>] [!axis.factor_group<8>, !axis.factor_group<4>, !axis.factor_group<4>]

  %in = tensor.empty() : tensor<8xi8>
  %h = distributed.Collective %in : tensor<8xi8> on %mesh_in : !axis.factor_group<16> to tensor<8xi8> on %mesh_out : !axis.factor_group<16> reduces (%red0 : !axis.factor_group<4>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<i8>, %rhs_v: tensor<i8>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<i8>
    stablehlo.return %r : tensor<i8>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<8xi8>> -> tensor<8xi8>
}

// -----

// (Replicate, Mesh) next to (Mesh, Tile). mesh0 has (in replicate, out mesh1) and mesh1 has (in mesh0, out tile): the input tile of 8 holds a digit of extent 2 that is sliced onto mesh1, and mesh0's output is mesh1's input digit. The first halves are only the all-gather of mesh1; mesh0 has none. S = 8, n = 2; transfer time rounds to 0.
//   all-gather mesh1: V = 8, latency 5000 + 1000 = 6000, duration 6000, payload 8 -> 16
//   slice onto mesh0 (mesh1's digit): 16 -> 8, free
//   slice onto mesh1 (the input tile digit): 8 -> 4, free; it waits for mesh1's own gather
// Total 6000.
// The result has two uses, so the result port reports both.
// CHECK: chain: 3 steps, total duration 6000
// CHECK-NEXT: step 0: all-gather
// CHECK-NEXT: atoms: axis1.0(x2)
// CHECK-NEXT: payload: 8 -> 16
// CHECK-NEXT: latency: 6000
// CHECK-NEXT: V: [0, 8]
// CHECK-NEXT: rho: [0, 1]
// CHECK-NEXT: duration: 6000
// CHECK-NEXT: step 1: local-slice
// CHECK-NEXT: atoms: axis0.0(x2)
// CHECK-NEXT: payload: 16 -> 8
// CHECK-NEXT: latency: 0
// CHECK-NEXT: V: [0, 0]
// CHECK-NEXT: rho: [0, 0]
// CHECK-NEXT: duration: 0
// CHECK-NEXT: step 2: local-slice
// CHECK-NEXT: atoms: axis1.0(x2)
// CHECK-NEXT: payload: 8 -> 4
// CHECK-NEXT: latency: 0
// CHECK-NEXT: V: [0, 0]
// CHECK-NEXT: rho: [0, 0]
// CHECK-NEXT: duration: 0
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<8xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<4xi8> (the await result, 2 uses) <- step 2
module @replicate_mesh_row {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<8xi8> 0
  %to0 = axis.getaxis tensor<4xi8> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f4 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f4 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f5 = axis.factor %ti0 : !axis.shape_axis<tensor<8xi8>, 0> <4, 2>
  %lhs0 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<8xi8>, 0>, 4, 2>)
  %f6 = axis.factor %to0 : !axis.shape_axis<tensor<4xi8>, 0> <4, 1>
  %rhs0 = axis.product (%f6 : !axis.axis_factor<!axis.shape_axis<tensor<4xi8>, 0>, 4, 1>)
  %f7 = axis.factor %ti0 : !axis.shape_axis<tensor<8xi8>, 0> <2, 1>
  %lhs1 = axis.product (%f7 : !axis.axis_factor<!axis.shape_axis<tensor<8xi8>, 0>, 2, 1>)
  %f8 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs1 = axis.product (%f8 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f9 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs2 = axis.product (%f9 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f10 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %rhs2 = axis.product (%f10 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f11 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %lhs3 = axis.product (%f11 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f12 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %rhs3 = axis.product (%f12 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %map = axis.map %lhs0, %lhs1, %lhs2, %lhs3 to %rhs0, %rhs1, %rhs2, %rhs3 : [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<8xi8>
  %h = distributed.Collective %in : tensor<8xi8> on %mesh_in : !axis.factor_group<4> to tensor<4xi8> on %mesh_out : !axis.factor_group<4> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xi8>> -> tensor<4xi8>
  %u = stablehlo.add %v, %v : tensor<4xi8>
}

// -----

// (Tile, Mesh) with n = 4. mesh0 sends its digit to the output tile and takes mesh1's digit; mesh1 is (Mesh, Replicate). Both gathers precede the only slice, which needs mesh0's own gather and mesh1's digit. S = 4, k = 2.
//   all-gather mesh0: V = 4 * 3 = 12, transfer time rounds to 0, latency 5000 + 2000 = 7000, duration 7000, payload 4 -> 16
//   all-gather mesh1: V = 16 * 3 = 48, transfer time 48/50 = 0.96ns rounds to 1, latency 7000, duration 7001, payload 16 -> 64
//   slice onto mesh0: 64 -> 16, free
// Running mesh0 first is one nanosecond cheaper than the reverse (mesh1 first would round its own transfer time up while mesh0's still rounds to 0 either way). decompose_collective_meshcoupled_bandwidth.mlir shows the order changing with the bandwidths.
// CHECK: chain: 3 steps, total duration 14001
// CHECK-NEXT: step 0: all-gather
// CHECK-NEXT: atoms: axis0.0(x4)
// CHECK-NEXT: payload: 4 -> 16
// CHECK-NEXT: latency: 7000
// CHECK-NEXT: V: [12, 0]
// CHECK-NEXT: rho: [1, 0]
// CHECK-NEXT: duration: 7000
// CHECK-NEXT: step 1: all-gather
// CHECK-NEXT: atoms: axis1.0(x4)
// CHECK-NEXT: payload: 16 -> 64
// CHECK-NEXT: latency: 7000
// CHECK-NEXT: V: [0, 48]
// CHECK-NEXT: rho: [0, 1]
// CHECK-NEXT: duration: 7001
// CHECK-NEXT: step 2: local-slice
// CHECK-NEXT: atoms: axis0.0(x4)
// CHECK-NEXT: payload: 64 -> 16
// CHECK-NEXT: latency: 0
// CHECK-NEXT: V: [0, 0]
// CHECK-NEXT: rho: [0, 0]
// CHECK-NEXT: duration: 0
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<4xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<16xi8> (the await result, 0 uses) <- step 2
module @tile_mesh_row_n4 {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 4>, !distributed.physical_comm_axis<4, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 4>, !distributed.physical_comm_axis<4, 1>
  %rep2 = distributed.ReplicationAxis 4 : !distributed.replication_axis<4>
  %ti0 = axis.getaxis tensor<4xi8> 0
  %to0 = axis.getaxis tensor<16xi8> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %f4 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %mesh_out = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>, %f4 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f5 = axis.factor %ti0 : !axis.shape_axis<tensor<4xi8>, 0> <4, 1>
  %lhs0 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<4xi8>, 0>, 4, 1>)
  %f6 = axis.factor %to0 : !axis.shape_axis<tensor<16xi8>, 0> <4, 1>
  %rhs0 = axis.product (%f6 : !axis.axis_factor<!axis.shape_axis<tensor<16xi8>, 0>, 4, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %lhs1 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>)
  %f8 = axis.factor %to0 : !axis.shape_axis<tensor<16xi8>, 0> <4, 4>
  %rhs1 = axis.product (%f8 : !axis.axis_factor<!axis.shape_axis<tensor<16xi8>, 0>, 4, 4>)
  %f9 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %lhs2 = axis.product (%f9 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f10 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %rhs2 = axis.product (%f10 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>)
  %f11 = axis.factor %rep2 : !distributed.replication_axis<4> <4, 1>
  %lhs3 = axis.product (%f11 : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
  %f12 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %rhs3 = axis.product (%f12 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %map = axis.map %lhs0, %lhs1, %lhs2, %lhs3 to %rhs0, %rhs1, %rhs2, %rhs3 : [!axis.factor_group<4>, !axis.factor_group<4>, !axis.factor_group<4>, !axis.factor_group<4>] [!axis.factor_group<4>, !axis.factor_group<4>, !axis.factor_group<4>, !axis.factor_group<4>]

  %in = tensor.empty() : tensor<4xi8>
  %h = distributed.Collective %in : tensor<4xi8> on %mesh_in : !axis.factor_group<16> to tensor<16xi8> on %mesh_out : !axis.factor_group<16> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<16xi8>> -> tensor<16xi8>
}

// -----

// A pure permutation cycle and a half-split component in one collective. mesh0 and mesh1 swap digits (one permute step); mesh2 is reduced and takes mesh3's digit, mesh3 is (mesh, replicate). The components are independent DP dimensions that share the payload S = 4, all extents 2; every step's transfer time rounds to 0 or 1.
//   permute mesh0,mesh1: V = 4 on each axis, latency 5000 + 1000 + 1000 = 7000, duration 7000 (duration 7000 at payload 8 too, since 8/50 still rounds to 0)
//   all-reduce mesh2: recursive doubling, latency 6000, duration 6000
//   all-gather mesh3: V = 4, latency 6000, duration 6000, payload 4 -> 8
//   slice onto mesh2: 8 -> 4, free
// The permute never has to run while the gather's 8 bytes are live at this payload scale (transfer time rounds to 0 regardless), so its position is a tie, resolved to the last. Total 6000 + 6000 + 7000 = 19000.
// CHECK: chain: 4 steps, total duration 19000
// CHECK-NEXT: step 0: all-reduce
// CHECK-NEXT: atoms: axis2.0(x2)
// CHECK-NEXT: payload: 4 -> 4
// CHECK-NEXT: latency: 6000
// CHECK-NEXT: V: [0, 0, 4, 0]
// CHECK-NEXT: rho: [0, 0, 1, 0]
// CHECK-NEXT: duration: 6000
// CHECK-NEXT: step 1: all-gather
// CHECK-NEXT: atoms: axis3.0(x2)
// CHECK-NEXT: payload: 4 -> 8
// CHECK-NEXT: latency: 6000
// CHECK-NEXT: V: [0, 0, 0, 4]
// CHECK-NEXT: rho: [0, 0, 0, 1]
// CHECK-NEXT: duration: 6000
// CHECK-NEXT: step 2: local-slice
// CHECK-NEXT: atoms: axis2.0(x2)
// CHECK-NEXT: payload: 8 -> 4
// CHECK-NEXT: latency: 0
// CHECK-NEXT: V: [0, 0, 0, 0]
// CHECK-NEXT: rho: [0, 0, 0, 0]
// CHECK-NEXT: duration: 0
// CHECK-NEXT: step 3: permute
// CHECK-NEXT: atoms: axis0.0(x2) axis1.0(x2)
// CHECK-NEXT: payload: 4 -> 4
// CHECK-NEXT: latency: 7000
// CHECK-NEXT: V: [4, 4, 0, 0]
// CHECK-NEXT: rho: [1, 1, 0, 0]
// CHECK-NEXT: duration: 7000
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<4xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<4xi8> (the await result, 0 uses) <- step 3
module @permute_and_half_split {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 8>, !distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0, %p1, %p2, %p3 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 8>, !distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<4xi8> 0
  %to0 = axis.getaxis tensor<4xi8> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 8> <2, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %f3 = axis.factor %p2 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f4 = axis.factor %p3 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 8>, 2, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>, %f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f4 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f5 = axis.factor %p0 : !distributed.physical_comm_axis<2, 8> <2, 1>
  %f6 = axis.factor %p1 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %f7 = axis.factor %p2 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f8 = axis.factor %p3 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f5 : !axis.axis_factor<!distributed.physical_comm_axis<2, 8>, 2, 1>, %f6 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>, %f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f8 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f9 = axis.factor %p2 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %red0 = axis.product (%f9 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f10 = axis.factor %ti0 : !axis.shape_axis<tensor<4xi8>, 0> <4, 1>
  %lhs0 = axis.product (%f10 : !axis.axis_factor<!axis.shape_axis<tensor<4xi8>, 0>, 4, 1>)
  %f11 = axis.factor %to0 : !axis.shape_axis<tensor<4xi8>, 0> <4, 1>
  %rhs0 = axis.product (%f11 : !axis.axis_factor<!axis.shape_axis<tensor<4xi8>, 0>, 4, 1>)
  %f12 = axis.factor %p0 : !distributed.physical_comm_axis<2, 8> <2, 1>
  %lhs1 = axis.product (%f12 : !axis.axis_factor<!distributed.physical_comm_axis<2, 8>, 2, 1>)
  %f13 = axis.factor %p1 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %rhs1 = axis.product (%f13 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>)
  %f14 = axis.factor %p1 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %lhs2 = axis.product (%f14 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>)
  %f15 = axis.factor %p0 : !distributed.physical_comm_axis<2, 8> <2, 1>
  %rhs2 = axis.product (%f15 : !axis.axis_factor<!distributed.physical_comm_axis<2, 8>, 2, 1>)
  %f16 = axis.factor %p3 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs3 = axis.product (%f16 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f17 = axis.factor %p2 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %rhs3 = axis.product (%f17 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f18 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %lhs4 = axis.product (%f18 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f19 = axis.factor %p3 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs4 = axis.product (%f19 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map = axis.map %lhs0, %lhs1, %lhs2, %lhs3, %lhs4 to %rhs0, %rhs1, %rhs2, %rhs3, %rhs4 : [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<4xi8>
  %h = distributed.Collective %in : tensor<4xi8> on %mesh_in : !axis.factor_group<16> to tensor<4xi8> on %mesh_out : !axis.factor_group<16> reduces (%red0 : !axis.factor_group<2>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<i8>, %rhs_v: tensor<i8>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<i8>
    stablehlo.return %r : tensor<i8>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xi8>> -> tensor<4xi8>
}

// -----

// A half-split component whose reduction body (subtract) is not a recognized associative operation is not decomposed, since the all-reduce that starts the half-split combines partial results in any order.
// CHECK: print-collective-plan: no chain (unsupported reduction body on mesh0.0 (not a single recognized associative operation))

module @half_split_unknown_kind {
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
    %r = stablehlo.subtract %lhs_v, %rhs_v : tensor<i8>
    stablehlo.return %r : tensor<i8>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xi8>> -> tensor<4xi8>
}
