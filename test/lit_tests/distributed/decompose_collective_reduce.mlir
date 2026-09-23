// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan=chain=true %s 2>&1 >/dev/null | FileCheck %s
// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true disable-reduce-scatter=true" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NORS

// Decomposition of collectives with reductions (see decompose_collective.mlir
// for the reduction-free rows and the conventions of the printed chain).
//
// Hand computation uses the default parameters: bandwidth 50 bytes/ns on
// every axis, round latency 1000ns, launch latency 5000ns. For a group of n
// on one axis with per-device payload S and k = ceil(log2 n)
// (CollectiveCost.h):
//   all-reduce:     halving + doubling V = 2 S (n - 1) / n, latency 5000 + 2000 k;
//                   recursive doubling V = S k, latency 5000 + 1000 k; cheaper wins
//   reduce-scatter: V = S (n - 1) / n, latency 5000 + 1000 k, payload S -> S / n
// duration = latency + round(V / BW) (CollectiveCost.h's Units note): at
// these tiny payloads that rounded term is usually 0 or 1, so most decisions
// below turn on round count rather than volume. Every reduced atom is its own
// step. The NORS run removes reduce-scatter from the candidates, which
// exercises the all-reduce + local slice alternative that a cost-minimizing
// search never selects. Element type i8, so a payload in bytes equals its
// element count.

// All-reduce (n = 8, S = 8): the reduced atom's output is replicated. Both
// algorithms' transfer times round to 0 at this payload, so the fewer-round
// algorithm wins outright.
//   halving + doubling: 2k = 6 rounds, V = 2 * 8 * 7 / 8 = 14, transfer time rounds to 0, latency 5000 + 6000 = 11000, duration 11000 (not chosen)
//   recursive doubling: k = 3 rounds, V = 8 * 3 = 24, transfer time rounds to 0, latency 5000 + 3000 = 8000, duration 8000 (chosen)
// CHECK: chain: 1 steps, total duration 8000
// CHECK-NEXT: step 0: all-reduce
// CHECK-NEXT: atoms: axis0.0(x8)
// CHECK-NEXT: payload: 8 -> 8
// CHECK-NEXT: latency: 8000
// CHECK-NEXT: V: [24]
// CHECK-NEXT: rho: [1]
// CHECK-NEXT: duration: 8000
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<8xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<8xi8> (the await result, 0 uses) <- step 0
// NORS: chain: 1 steps, total duration 8000
// NORS-NEXT: step 0: all-reduce
// NORS-NEXT: atoms: axis0.0(x8)
// NORS-NEXT: payload: 8 -> 8
// NORS-NEXT: latency: 8000
// NORS-NEXT: V: [24]
// NORS-NEXT: rho: [1]
// NORS-NEXT: duration: 8000
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
//   V = 8 * 7 / 8 = 7, transfer time rounds to 0, latency 5000 + 3000 = 8000, duration 8000, payload 8 -> 1.
// With reduce-scatter disabled: the all-reduce (8000, recursive doubling as above) followed by a free local slice 8 -> 1 also gives 8000.
// CHECK: chain: 1 steps, total duration 8000
// CHECK-NEXT: step 0: reduce-scatter
// CHECK-NEXT: atoms: axis0.0(x8)
// CHECK-NEXT: payload: 8 -> 1
// CHECK-NEXT: latency: 8000
// CHECK-NEXT: V: [7]
// CHECK-NEXT: rho: [1]
// CHECK-NEXT: duration: 8000
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<8xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<1xi8> (the await result, 0 uses) <- step 0
// NORS: chain: 2 steps, total duration 8000
// NORS-NEXT: step 0: all-reduce
// NORS-NEXT: atoms: axis0.0(x8)
// NORS-NEXT: payload: 8 -> 8
// NORS-NEXT: latency: 8000
// NORS-NEXT: V: [24]
// NORS-NEXT: rho: [1]
// NORS-NEXT: duration: 8000
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

// Order of a reduce-scatter and an all-reduce on different axes (S = 16). axis0 (n = 2) all-reduces and axis1 (n = 4) reduce-scatters onto a tile atom of extent 4, so the tile shrinks 16 -> 4. Both orders' transfer times round to 0 or a small remainder here, so the DP picks the axis0-first order:
//   axis0 all-reduce first: on 16 bytes, recursive doubling V = 16, transfer time rounds to 0, latency 5000 + 1000 = 6000; then axis1 reduce-scatter on 16 bytes, V = 12, transfer time rounds to 0, latency 5000 + 2000 = 7000; total 13000
// With reduce-scatter disabled: axis1 all-reduces at S = 16 (recursive doubling V = 32, transfer time 32/50 = 0.64ns rounds to 1, latency 5000 + 2000 = 7000, duration 7001), a free slice 16 -> 4, then the axis0 all-reduce on 4 bytes (6000); total 13001.
// CHECK: chain: 2 steps, total duration 13000
// CHECK-NEXT: step 0: all-reduce
// CHECK-NEXT: atoms: axis0.0(x2)
// CHECK-NEXT: payload: 16 -> 16
// CHECK-NEXT: latency: 6000
// CHECK-NEXT: V: [16, 0]
// CHECK-NEXT: rho: [1, 0]
// CHECK-NEXT: duration: 6000
// CHECK-NEXT: step 1: reduce-scatter
// CHECK-NEXT: atoms: axis1.0(x4)
// CHECK-NEXT: payload: 16 -> 4
// CHECK-NEXT: latency: 7000
// CHECK-NEXT: V: [0, 12]
// CHECK-NEXT: rho: [0, 1]
// CHECK-NEXT: duration: 7000
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<16xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<4xi8> (the await result, 0 uses) <- step 1
// NORS: chain: 3 steps, total duration 13001
// NORS-NEXT: step 0: all-reduce
// NORS-NEXT: atoms: axis0.0(x2)
// NORS-NEXT: payload: 16 -> 16
// NORS-NEXT: latency: 6000
// NORS-NEXT: V: [16, 0]
// NORS-NEXT: rho: [1, 0]
// NORS-NEXT: duration: 6000
// NORS-NEXT: step 1: all-reduce
// NORS-NEXT: atoms: axis1.0(x4)
// NORS-NEXT: payload: 16 -> 16
// NORS-NEXT: latency: 7000
// NORS-NEXT: V: [0, 32]
// NORS-NEXT: rho: [0, 1]
// NORS-NEXT: duration: 7001
// NORS-NEXT: step 2: local-slice
// NORS-NEXT: atoms: axis1.0(x4)
// NORS-NEXT: payload: 16 -> 4
// NORS-NEXT: latency: 0
// NORS-NEXT: V: [0, 0]
// NORS-NEXT: rho: [0, 0]
// NORS-NEXT: duration: 0
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
// CHECK: chain: 1 steps, total duration 8000
// CHECK-NEXT: step 0: all-reduce
// CHECK-NEXT: atoms: axis0.0(x8)
// CHECK-NEXT: payload: 8 -> 8
// CHECK-NEXT: latency: 8000
// CHECK-NEXT: V: [24]
// CHECK-NEXT: rho: [1]
// CHECK-NEXT: duration: 8000
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

// Reduce then permute, reduced atom first. mesh0 is reduced and its output comes from mesh1, whose own output is replicated: the atoms the reduced digit is moved through form an open path (mesh1 -> mesh0), not a permutation cycle. It closes into a cycle through mesh0, which the all-reduce empties: all-reduce mesh0, then one permute that swaps the two atoms' contents. n = 2, S = 4; every transfer time here rounds to 0.
//   all-reduce mesh0: recursive doubling, latency 6000, duration 6000
//   permute mesh0, mesh1: latency 5000 + 1000 + 1000 = 7000, duration 7000
// That chain totals 13000, more than the half-split (all-reduce mesh0 6000, all-gather mesh1 6000, then a free slice back to payload 4, total 12000): the half-split wins outright at this payload. decompose_collective_meshcoupled.mlir keeps it as the baseline and has more cases (n = 4, other rows, mixed with a permute).
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

// The same collective with the axes swapped (mesh1 reduced, mesh0 feeding it): the same half-split, expressed here as an all-gather then reduce-scatter chain instead of the other module's all-reduce/all-gather/slice one; both cost 12000 at this payload.
// CHECK: chain: 2 steps, total duration 12000
// CHECK-NEXT: step 0: all-gather
// CHECK-NEXT: atoms: axis0.0(x2)
// CHECK-NEXT: payload: 4 -> 8
// CHECK-NEXT: latency: 6000
// CHECK-NEXT: V: [4, 0]
// CHECK-NEXT: rho: [1, 0]
// CHECK-NEXT: duration: 6000
// CHECK-NEXT: step 1: reduce-scatter
// CHECK-NEXT: atoms: axis1.0(x2)
// CHECK-NEXT: payload: 8 -> 4
// CHECK-NEXT: latency: 6000
// CHECK-NEXT: V: [0, 4]
// CHECK-NEXT: rho: [0, 1]
// CHECK-NEXT: duration: 6000
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<4xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<4xi8> (the await result, 0 uses) <- step 1
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
