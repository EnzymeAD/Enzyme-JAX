// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan=chain=true %s 2>&1 >/dev/null | FileCheck %s
// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true disable-reduce-scatter=true" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NORS

// Decomposition of collectives with reductions (see decompose_collective.mlir
// for the reduction-free rows and the conventions of the printed chain).
//
// Hand computation uses the default parameters: bandwidth 1 on every axis,
// round latency 0.01, launch latency 0.1. For a group of n on one axis with
// per-device payload S and k = ceil(log2 n) (CollectiveCost.h):
//   all-reduce:     halving + doubling V = 2 S (n - 1) / n, latency 0.1 + 0.02 k;
//                   recursive doubling V = S k, latency 0.1 + 0.01 k; cheaper wins
//   reduce-scatter: V = S (n - 1) / n, latency 0.1 + 0.01 k, payload S -> S / n
// duration = latency + V / BW. Every reduced atom is its own step. The
// NORS run removes reduce-scatter from the candidates, which exercises the
// all-reduce + local slice alternative that a cost-minimizing search never
// selects. Element type i8, so a payload in bytes equals its element count.

// All-reduce (n = 8, S = 8): the reduced atom's output is replicated.
//   halving + doubling: 2k = 6 rounds, V = 2 * 8 * 7 / 8 = 14, latency 0.1 + 0.06 = 0.16, duration 14.16
//   recursive doubling: k = 3 rounds, V = 8 * 3 = 24, latency 0.13, duration 24.13 (not chosen)
// CHECK: chain: 1 steps, total duration 14.16
// CHECK-NEXT: step 0: all-reduce
// CHECK-NEXT: atoms: axis0.0(x8)
// CHECK-NEXT: payload: 8 -> 8
// CHECK-NEXT: latency: 0.16
// CHECK-NEXT: V: [14]
// CHECK-NEXT: rho: [1]
// CHECK-NEXT: duration: 14.16
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<8xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<8xi8> (the await result, 0 uses) <- step 0
// NORS: chain: 1 steps, total duration 14.16
// NORS-NEXT: step 0: all-reduce
// NORS-NEXT: atoms: axis0.0(x8)
// NORS-NEXT: payload: 8 -> 8
// NORS-NEXT: latency: 0.16
// NORS-NEXT: V: [14]
// NORS-NEXT: rho: [1]
// NORS-NEXT: duration: 14.16
// NORS-NEXT: semantics: verified
// NORS-NEXT: input port: tensor<8xi8> (the collective's input_object operand) -> step 0
// NORS-NEXT: result port: tensor<8xi8> (the await result, 0 uses) <- step 0
module @all_reduce_n8 {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<8, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<8, 1>
  %rep8 = distributed.ReplicationAxis 8 : !distributed.replication_axis<8>
  %ti0 = axis.getaxis tensor<8xi8> 0
  %to0 = axis.getaxis tensor<8xi8> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %mesh_out = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %red0 = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %f4 = axis.factor %ti0 : !axis.shape_axis<tensor<8xi8>, 0> <8, 1>
  %lhs0 = axis.product (%f4 : !axis.axis_factor<!axis.shape_axis<tensor<8xi8>, 0>, 8, 1>)
  %f5 = axis.factor %to0 : !axis.shape_axis<tensor<8xi8>, 0> <8, 1>
  %rhs0 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<8xi8>, 0>, 8, 1>)
  %f6 = axis.factor %rep8 : !distributed.replication_axis<8> <8, 1>
  %lhs1 = axis.product (%f6 : !axis.axis_factor<!distributed.replication_axis<8>, 8, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %rhs1 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<8>, !axis.factor_group<8>] [!axis.factor_group<8>, !axis.factor_group<8>]

  %in = tensor.empty() : tensor<8xi8>
  %h = distributed.Collective %in : tensor<8xi8> on %mesh_in : !axis.factor_group<8> to tensor<8xi8> on %mesh_out : !axis.factor_group<8> reduces (%red0 : !axis.factor_group<8>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<i8>, %rhs_v: tensor<i8>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<i8>
    stablehlo.return %r : tensor<i8>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<8xi8>> -> tensor<8xi8>
}

// -----

// Reduce-scatter (n = 8, S = 8): the sum is scattered onto a tile atom of extent 8, so each device ends with 1 byte.
//   V = 8 * 7 / 8 = 7, latency 0.1 + 3 * 0.01 = 0.13, duration 7.13, payload 8 -> 1.
// With reduce-scatter disabled: the all-reduce (14.16, as above) followed by a free local slice 8 -> 1 gives 14.16, about twice the reduce-scatter.
// CHECK: chain: 1 steps, total duration 7.13
// CHECK-NEXT: step 0: reduce-scatter
// CHECK-NEXT: atoms: axis0.0(x8)
// CHECK-NEXT: payload: 8 -> 1
// CHECK-NEXT: latency: 0.13
// CHECK-NEXT: V: [7]
// CHECK-NEXT: rho: [1]
// CHECK-NEXT: duration: 7.13
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<8xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<1xi8> (the await result, 0 uses) <- step 0
// NORS: chain: 2 steps, total duration 14.16
// NORS-NEXT: step 0: all-reduce
// NORS-NEXT: atoms: axis0.0(x8)
// NORS-NEXT: payload: 8 -> 8
// NORS-NEXT: latency: 0.16
// NORS-NEXT: V: [14]
// NORS-NEXT: rho: [1]
// NORS-NEXT: duration: 14.16
// NORS-NEXT: step 1: local-slice
// NORS-NEXT: atoms: axis0.0(x8)
// NORS-NEXT: payload: 8 -> 1
// NORS-NEXT: latency: 0
// NORS-NEXT: V: [0]
// NORS-NEXT: rho: [0]
// NORS-NEXT: duration: 0
// NORS-NEXT: semantics: verified
// NORS-NEXT: input port: tensor<8xi8> (the collective's input_object operand) -> step 0
// NORS-NEXT: result port: tensor<1xi8> (the await result, 0 uses) <- step 1
module @reduce_scatter_n8 {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<8, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<8, 1>
  %ti0 = axis.getaxis tensor<8xi8> 0
  %to0 = axis.getaxis tensor<1xi8> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %mesh_out = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %red0 = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %f4 = axis.factor %ti0 : !axis.shape_axis<tensor<8xi8>, 0> <8, 1>
  %lhs0 = axis.product (%f4 : !axis.axis_factor<!axis.shape_axis<tensor<8xi8>, 0>, 8, 1>)
  %f5 = axis.factor %p0 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %rhs0 = axis.product (%f5 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %map = axis.map %lhs0 to %rhs0 : [!axis.factor_group<8>] [!axis.factor_group<8>]

  %in = tensor.empty() : tensor<8xi8>
  %h = distributed.Collective %in : tensor<8xi8> on %mesh_in : !axis.factor_group<8> to tensor<1xi8> on %mesh_out : !axis.factor_group<8> reduces (%red0 : !axis.factor_group<8>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<i8>, %rhs_v: tensor<i8>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<i8>
    stablehlo.return %r : tensor<i8>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<1xi8>> -> tensor<1xi8>
}

// -----

// Order of a reduce-scatter and an all-reduce on different axes (S = 16). axis0 (n = 2) all-reduces and axis1 (n = 4) reduce-scatters onto a tile atom of extent 4, so the tile shrinks 16 -> 4. The all-reduce is the first unit in atom order, so the DP must choose the reduce-scatter first on cost alone:
//   reduce-scatter first: V = 16 * 3 / 4 = 12, latency 0.12 -> 12.12; then the all-reduce runs on 4 bytes: halving + doubling V = 2 * 4 / 2 = 4, latency 0.12, recursive doubling V = 4, latency 0.11 -> 4.11; total 16.23
//   all-reduce first: on 16 bytes V = 16, latency 0.11 -> 16.11; then reduce-scatter 12.12; total 28.23
// With reduce-scatter disabled: axis1 all-reduces at S = 16 (halving + doubling V = 2 * 16 * 3 / 4 = 24, latency 0.14 -> 24.14; recursive doubling V = 32), a free slice 16 -> 4, then the axis0 all-reduce on 4 bytes 4.11; total 28.25. The other order is worse (16.11 + 24.14 = 40.25).
// CHECK: chain: 2 steps, total duration 16.23
// CHECK-NEXT: step 0: reduce-scatter
// CHECK-NEXT: atoms: axis1.0(x4)
// CHECK-NEXT: payload: 16 -> 4
// CHECK-NEXT: latency: 0.12
// CHECK-NEXT: V: [0, 12]
// CHECK-NEXT: rho: [0, 1]
// CHECK-NEXT: duration: 12.12
// CHECK-NEXT: step 1: all-reduce
// CHECK-NEXT: atoms: axis0.0(x2)
// CHECK-NEXT: payload: 4 -> 4
// CHECK-NEXT: latency: 0.11
// CHECK-NEXT: V: [4, 0]
// CHECK-NEXT: rho: [1, 0]
// CHECK-NEXT: duration: 4.11
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<16xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<4xi8> (the await result, 0 uses) <- step 1
// NORS: chain: 3 steps, total duration 28.25
// NORS-NEXT: step 0: all-reduce
// NORS-NEXT: atoms: axis1.0(x4)
// NORS-NEXT: payload: 16 -> 16
// NORS-NEXT: latency: 0.14
// NORS-NEXT: V: [0, 24]
// NORS-NEXT: rho: [0, 1]
// NORS-NEXT: duration: 24.14
// NORS-NEXT: step 1: local-slice
// NORS-NEXT: atoms: axis1.0(x4)
// NORS-NEXT: payload: 16 -> 4
// NORS-NEXT: latency: 0
// NORS-NEXT: V: [0, 0]
// NORS-NEXT: rho: [0, 0]
// NORS-NEXT: duration: 0
// NORS-NEXT: step 2: all-reduce
// NORS-NEXT: atoms: axis0.0(x2)
// NORS-NEXT: payload: 4 -> 4
// NORS-NEXT: latency: 0.11
// NORS-NEXT: V: [4, 0]
// NORS-NEXT: rho: [1, 0]
// NORS-NEXT: duration: 4.11
// NORS-NEXT: semantics: verified
// NORS-NEXT: input port: tensor<16xi8> (the collective's input_object operand) -> step 0
// NORS-NEXT: result port: tensor<4xi8> (the await result, 0 uses) <- step 2
module @reduce_two_axes_ordered {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<4, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<4, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<16xi8> 0
  %to0 = axis.getaxis tensor<4xi8> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %f4 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %mesh_out = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>, %f4 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f5 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %f6 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %red0 = axis.product (%f5 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>, %f6 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f7 = axis.factor %ti0 : !axis.shape_axis<tensor<16xi8>, 0> <16, 1>
  %lhs0 = axis.product (%f7 : !axis.axis_factor<!axis.shape_axis<tensor<16xi8>, 0>, 16, 1>)
  %f8 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %f9 = axis.factor %to0 : !axis.shape_axis<tensor<4xi8>, 0> <4, 1>
  %rhs0 = axis.product (%f8 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>, %f9 : !axis.axis_factor<!axis.shape_axis<tensor<4xi8>, 0>, 4, 1>)
  %f10 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %lhs1 = axis.product (%f10 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f11 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %rhs1 = axis.product (%f11 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<16>, !axis.factor_group<2>] [!axis.factor_group<16>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<16xi8>
  %h = distributed.Collective %in : tensor<16xi8> on %mesh_in : !axis.factor_group<8> to tensor<4xi8> on %mesh_out : !axis.factor_group<8> reduces (%red0 : !axis.factor_group<8>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<i8>, %rhs_v: tensor<i8>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<i8>
    stablehlo.return %r : tensor<i8>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xi8>> -> tensor<4xi8>
}

// -----

// Ports of a reduction chain: the input feeds step 0 and the await result, which here has two uses, is produced by the last step.
// CHECK: chain: 1 steps, total duration 14.16
// CHECK-NEXT: step 0: all-reduce
// CHECK-NEXT: atoms: axis0.0(x8)
// CHECK-NEXT: payload: 8 -> 8
// CHECK-NEXT: latency: 0.16
// CHECK-NEXT: V: [14]
// CHECK-NEXT: rho: [1]
// CHECK-NEXT: duration: 14.16
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<8xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<8xi8> (the await result, 2 uses) <- step 0
module @all_reduce_ports {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<8, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<8, 1>
  %rep8 = distributed.ReplicationAxis 8 : !distributed.replication_axis<8>
  %ti0 = axis.getaxis tensor<8xi8> 0
  %to0 = axis.getaxis tensor<8xi8> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %mesh_out = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %red0 = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %f4 = axis.factor %ti0 : !axis.shape_axis<tensor<8xi8>, 0> <8, 1>
  %lhs0 = axis.product (%f4 : !axis.axis_factor<!axis.shape_axis<tensor<8xi8>, 0>, 8, 1>)
  %f5 = axis.factor %to0 : !axis.shape_axis<tensor<8xi8>, 0> <8, 1>
  %rhs0 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<8xi8>, 0>, 8, 1>)
  %f6 = axis.factor %rep8 : !distributed.replication_axis<8> <8, 1>
  %lhs1 = axis.product (%f6 : !axis.axis_factor<!distributed.replication_axis<8>, 8, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %rhs1 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<8>, !axis.factor_group<8>] [!axis.factor_group<8>, !axis.factor_group<8>]

  %in = tensor.empty() : tensor<8xi8>
  %h = distributed.Collective %in : tensor<8xi8> on %mesh_in : !axis.factor_group<8> to tensor<8xi8> on %mesh_out : !axis.factor_group<8> reduces (%red0 : !axis.factor_group<8>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<i8>, %rhs_v: tensor<i8>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<i8>
    stablehlo.return %r : tensor<i8>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<8xi8>> -> tensor<8xi8>
  %u = stablehlo.add %v, %v : tensor<8xi8>
}

// -----

// A reduction whose body is not a recognized associative operation (subtract) is not decomposed, since the log-round algorithms combine partial results in any order.
// CHECK: print-collective-plan: no chain (unsupported reduction body on mesh0.0 (not a single recognized associative operation))
module @all_reduce_unknown_kind {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<4xi8> 0
  %to0 = axis.getaxis tensor<4xi8> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %red0 = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f4 = axis.factor %ti0 : !axis.shape_axis<tensor<4xi8>, 0> <4, 1>
  %lhs0 = axis.product (%f4 : !axis.axis_factor<!axis.shape_axis<tensor<4xi8>, 0>, 4, 1>)
  %f5 = axis.factor %to0 : !axis.shape_axis<tensor<4xi8>, 0> <4, 1>
  %rhs0 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<4xi8>, 0>, 4, 1>)
  %f6 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %lhs1 = axis.product (%f6 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs1 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<4xi8>
  %h = distributed.Collective %in : tensor<4xi8> on %mesh_in : !axis.factor_group<2> to tensor<4xi8> on %mesh_out : !axis.factor_group<2> reduces (%red0 : !axis.factor_group<2>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<i8>, %rhs_v: tensor<i8>):
    %r = stablehlo.subtract %lhs_v, %rhs_v : tensor<i8>
    stablehlo.return %r : tensor<i8>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xi8>> -> tensor<4xi8>
}

// -----

// Reduce then permute, reduced atom first. mesh0 is reduced and its output comes from mesh1, whose own output is replicated: the atoms the reduced digit is moved through form an open path (mesh1 -> mesh0), not a permutation cycle, so both are half-split: all-reduce mesh0, all-gather mesh1, then a free slice onto mesh0 of the digit mesh1 held. n = 2, S = 4.
//   all-reduce mesh0: recursive doubling V = 4, latency 0.11, duration 4.11
//   all-gather mesh1: V = 4, latency 0.11, duration 4.11, payload 4 -> 8
//   slice onto mesh0: 8 -> 4, free
// Total 8.22. More cases (n = 4, other rows, mixed with a permute) are in decompose_collective_meshcoupled.mlir.
// CHECK: chain: 3 steps, total duration 8.22
// CHECK-NEXT: step 0: all-reduce
// CHECK-NEXT: atoms: axis0.0(x2)
// CHECK-NEXT: payload: 4 -> 4
// CHECK-NEXT: latency: 0.11
// CHECK-NEXT: V: [4, 0]
// CHECK-NEXT: rho: [1, 0]
// CHECK-NEXT: duration: 4.11
// CHECK-NEXT: step 1: all-gather
// CHECK-NEXT: atoms: axis1.0(x2)
// CHECK-NEXT: payload: 4 -> 8
// CHECK-NEXT: latency: 0.11
// CHECK-NEXT: V: [0, 4]
// CHECK-NEXT: rho: [0, 1]
// CHECK-NEXT: duration: 4.11
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
module @reduce_then_permute_reduced_first {
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

// The same collective with the axes swapped (mesh1 reduced, mesh0 feeding it): the chain is the same with the axes exchanged, 8.22.
// CHECK: chain: 3 steps, total duration 8.22
// CHECK-NEXT: step 0: all-reduce
// CHECK-NEXT: atoms: axis1.0(x2)
// CHECK-NEXT: payload: 4 -> 4
// CHECK-NEXT: latency: 0.11
// CHECK-NEXT: V: [0, 4]
// CHECK-NEXT: rho: [0, 1]
// CHECK-NEXT: duration: 4.11
// CHECK-NEXT: step 1: all-gather
// CHECK-NEXT: atoms: axis0.0(x2)
// CHECK-NEXT: payload: 4 -> 8
// CHECK-NEXT: latency: 0.11
// CHECK-NEXT: V: [4, 0]
// CHECK-NEXT: rho: [1, 0]
// CHECK-NEXT: duration: 4.11
// CHECK-NEXT: step 2: local-slice
// CHECK-NEXT: atoms: axis1.0(x2)
// CHECK-NEXT: payload: 8 -> 4
// CHECK-NEXT: latency: 0
// CHECK-NEXT: V: [0, 0]
// CHECK-NEXT: rho: [0, 0]
// CHECK-NEXT: duration: 0
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<4xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<4xi8> (the await result, 0 uses) <- step 2
module @reduce_then_permute_head_first {
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
  %f5 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %red0 = axis.product (%f5 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f6 = axis.factor %ti0 : !axis.shape_axis<tensor<4xi8>, 0> <4, 1>
  %lhs0 = axis.product (%f6 : !axis.axis_factor<!axis.shape_axis<tensor<4xi8>, 0>, 4, 1>)
  %f7 = axis.factor %to0 : !axis.shape_axis<tensor<4xi8>, 0> <4, 1>
  %rhs0 = axis.product (%f7 : !axis.axis_factor<!axis.shape_axis<tensor<4xi8>, 0>, 4, 1>)
  %f8 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %lhs1 = axis.product (%f8 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f9 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs1 = axis.product (%f9 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f10 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %lhs2 = axis.product (%f10 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f11 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %rhs2 = axis.product (%f11 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %map = axis.map %lhs0, %lhs1, %lhs2 to %rhs0, %rhs1, %rhs2 : [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<4xi8>
  %h = distributed.Collective %in : tensor<4xi8> on %mesh_in : !axis.factor_group<4> to tensor<4xi8> on %mesh_out : !axis.factor_group<4> reduces (%red0 : !axis.factor_group<2>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<i8>, %rhs_v: tensor<i8>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<i8>
    stablehlo.return %r : tensor<i8>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xi8>> -> tensor<4xi8>
}
