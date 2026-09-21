// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan=chain=true %s 2>&1 >/dev/null | FileCheck %s

// Mesh-coupled rows: atoms whose in or out role is another mesh atom. The
// decomposer half-splits every atom (in = P, out = Q) of such a component into
// a first half by P (all-reduce for a reduced atom, all-gather for a tile or
// mesh digit, nothing for a replicate) and a free local slice by Q (the input
// tile digit, or the digit that came from the partner atom). A slice waits for
// its own atom's first half and for the partner's first half. A component of
// pure (Mesh, Mesh) atoms stays one permute step (see decompose_collective.mlir).
//
// Hand computation uses the default parameters (bandwidth 1, round latency
// 0.01, launch latency 0.1) and the formulas at the top of
// decompose_collective.mlir. With group size n, k = ceil(log2 n) and payload S:
//   all-gather:  V = S (n - 1), latency 0.1 + 0.01 k, payload S -> n S
//   all-reduce:  the cheaper of halving + doubling (V = 2 S (n - 1) / n, latency
//                0.1 + 0.02 k) and recursive doubling (V = S k, latency 0.1 + 0.01 k)
//   permute:     V[a] = S (1 - 1/e) on each axis, latency 0.1 + 0.01 x (axes)
//   slice:       free, payload divides by the extent
// Element type i8, so a payload in bytes equals its element count.

// Reduce then permute, larger n. mesh0 is reduced and its output comes from mesh1 (in mesh, out replicate), so the atoms form the open path mesh1 -> mesh0 rather than a permutation cycle and both are half-split: all-reduce mesh0, all-gather mesh1, then a free slice onto mesh0 of the digit mesh1 held. n = 4 (k = 2), S = 8.
//   all-reduce mesh0: halving + doubling V = 2 * 8 * 3 / 4 = 12, latency 0.1 + 0.04 = 0.14, duration 12.14 (recursive doubling would be V = 16, 16.12)
//   all-gather mesh1: V = 8 * 3 = 24, latency 0.12, duration 24.12, payload 8 -> 32
//   slice onto mesh0: 32 -> 8, free
// All-reduce runs before the gather because it keeps the payload while the gather grows it: the other order would all-reduce 32 bytes (V = 48).
// Total 12.14 + 24.12 = 36.26. The gather moves 24 bytes per port where the digit only needs to move to mesh0 (at most 8 bytes per port), so a dedicated move after the all-reduce would be cheaper.
// CHECK: chain: 3 steps, total duration 36.26
// CHECK-NEXT: step 0: all-reduce
// CHECK-NEXT: atoms: axis0.0(x4)
// CHECK-NEXT: payload: 8 -> 8
// CHECK-NEXT: latency: 0.14
// CHECK-NEXT: V: [12, 0]
// CHECK-NEXT: rho: [1, 0]
// CHECK-NEXT: duration: 12.14
// CHECK-NEXT: step 1: all-gather
// CHECK-NEXT: atoms: axis1.0(x4)
// CHECK-NEXT: payload: 8 -> 32
// CHECK-NEXT: latency: 0.12
// CHECK-NEXT: V: [0, 24]
// CHECK-NEXT: rho: [0, 1]
// CHECK-NEXT: duration: 24.12
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

// (Replicate, Mesh) next to (Mesh, Tile). mesh0 has (in replicate, out mesh1) and mesh1 has (in mesh0, out tile): the input tile of 8 holds a digit of extent 2 that is sliced onto mesh1, and mesh0's output is mesh1's input digit. The first halves are only the all-gather of mesh1; mesh0 has none. S = 8, n = 2.
//   all-gather mesh1: V = 8, latency 0.11, duration 8.11, payload 8 -> 16
//   slice onto mesh0 (mesh1's digit): 16 -> 8, free
//   slice onto mesh1 (the input tile digit): 8 -> 4, free; it waits for mesh1's own gather
// Total 8.11. A permute that moves only what each device lacks would send 4 bytes (V = 4) instead of gathering 8.
// The result has two uses, so the result port reports both.
// CHECK: chain: 3 steps, total duration 8.11
// CHECK-NEXT: step 0: all-gather
// CHECK-NEXT: atoms: axis1.0(x2)
// CHECK-NEXT: payload: 8 -> 16
// CHECK-NEXT: latency: 0.11
// CHECK-NEXT: V: [0, 8]
// CHECK-NEXT: rho: [0, 1]
// CHECK-NEXT: duration: 8.11
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
//   all-gather mesh0: V = 4 * 3 = 12, latency 0.12, duration 12.12, payload 4 -> 16
//   all-gather mesh1: V = 16 * 3 = 48, latency 0.12, duration 48.12, payload 16 -> 64
//   slice onto mesh0: 64 -> 16, free
// The two orders tie at 60.24 (12 + 48 either way at uniform bandwidth), and the earlier candidate, mesh0, is kept. decompose_collective_meshcoupled_bandwidth.mlir shows the order changing with the bandwidths.
// CHECK: chain: 3 steps, total duration 60.24
// CHECK-NEXT: step 0: all-gather
// CHECK-NEXT: atoms: axis0.0(x4)
// CHECK-NEXT: payload: 4 -> 16
// CHECK-NEXT: latency: 0.12
// CHECK-NEXT: V: [12, 0]
// CHECK-NEXT: rho: [1, 0]
// CHECK-NEXT: duration: 12.12
// CHECK-NEXT: step 1: all-gather
// CHECK-NEXT: atoms: axis1.0(x4)
// CHECK-NEXT: payload: 16 -> 64
// CHECK-NEXT: latency: 0.12
// CHECK-NEXT: V: [0, 48]
// CHECK-NEXT: rho: [0, 1]
// CHECK-NEXT: duration: 48.12
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

// A pure permutation cycle and a half-split component in one collective. mesh0 and mesh1 swap digits (one permute step); mesh2 is reduced and takes mesh3's digit, mesh3 is (mesh, replicate). The components are independent DP dimensions that share the payload S = 4, all extents 2.
//   permute mesh0,mesh1: V = 4 * (1 - 1/2) = 2 on each axis, latency 0.1 + 0.02 = 0.12, duration 2.12 (4.12 at payload 8)
//   all-reduce mesh2: recursive doubling V = 4, latency 0.11, duration 4.11
//   all-gather mesh3: V = 4, latency 0.11, duration 4.11, payload 4 -> 8
//   slice onto mesh2: 8 -> 4, free
// The permute must run at payload 4 (after the slice), never while the gather's 8 bytes are live. Its position at payload 4 is a tie, resolved to the last. Total 4.11 + 4.11 + 2.12 = 10.34.
// CHECK: chain: 4 steps, total duration 10.34
// CHECK-NEXT: step 0: all-reduce
// CHECK-NEXT: atoms: axis2.0(x2)
// CHECK-NEXT: payload: 4 -> 4
// CHECK-NEXT: latency: 0.11
// CHECK-NEXT: V: [0, 0, 4, 0]
// CHECK-NEXT: rho: [0, 0, 1, 0]
// CHECK-NEXT: duration: 4.11
// CHECK-NEXT: step 1: all-gather
// CHECK-NEXT: atoms: axis3.0(x2)
// CHECK-NEXT: payload: 4 -> 8
// CHECK-NEXT: latency: 0.11
// CHECK-NEXT: V: [0, 0, 0, 4]
// CHECK-NEXT: rho: [0, 0, 0, 1]
// CHECK-NEXT: duration: 4.11
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
// CHECK-NEXT: latency: 0.12
// CHECK-NEXT: V: [2, 2, 0, 0]
// CHECK-NEXT: rho: [1, 1, 0, 0]
// CHECK-NEXT: duration: 2.12
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
