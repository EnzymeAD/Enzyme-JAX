// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan=chain=true %s 2>&1 >/dev/null | FileCheck %s
// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true disable-all-to-all=true" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOA2A

// distributed-print-collective-plan=chain=true reports the primitive chain
// the collective decomposer chose for each collective: one block per step
// (kind, atoms, payload bytes before -> after, latency, per-axis volume V and
// demand rho, isolated duration), then `semantics: verified` when the
// independent symbolic checker confirms the chain realizes the collective,
// then the two ports that attach the chain to the surrounding program.
//
// Hand computation uses the default parameters: bandwidth 50 bytes/ns on
// every axis, round latency 1000ns, launch latency 5000ns. Formulas
// (CollectiveCost.h), for a group of n on one axis with per-device payload S
// and k = ceil(log2 n):
//   all-gather:  V = S (n - 1), latency = 5000 + 1000 k, payload out n S
//   all-to-all:  pairwise V = S (n - 1) / n, latency = 5000 + 1000 (n - 1);
//                Bruck V = S k / 2, latency = 5000 + 1000 k; cheaper wins
//   permute:     V[a] = S on every axis with a nonzero change fraction (a
//                moving device sends its whole payload), latency =
//                5000 + 1000 x (axes crossed)
//   local slice: free, payload S / e
// duration = latency + round(max_a V[a] / BW[a]) (CollectiveCost.h's Units
// note: the bandwidth-derived term rounds to the nearest nanosecond; the
// latency term is already an exact integer). At the tiny payloads these
// modules use, that rounded term is usually 0, so most decisions below turn
// on which algorithm needs fewer rounds rather than on volume; the
// volume/bandwidth tradeoff itself is exercised at a realistic payload scale
// in print_primitive_footprints.mlir instead. Inputs are not written in
// atomic form; every RUN line atomizes them first. Element type i8, so a
// payload in bytes equals its element count.

// Local slice only: the input is replicated along the atom and each device keeps its own half (4 -> 2). The chain is one free step, so the duration is 0.
// CHECK: chain: 1 steps, total duration 0
// CHECK-NEXT: step 0: local-slice
// CHECK-NEXT: atoms: axis0.0(x2)
// CHECK-NEXT: payload: 4 -> 2
// CHECK-NEXT: latency: 0
// CHECK-NEXT: V: [0]
// CHECK-NEXT: rho: [0]
// CHECK-NEXT: duration: 0
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<4xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<2xi8> (the await result, 0 uses) <- step 0
module @local_slice_only {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<4xi8> 0
  %to0 = axis.getaxis tensor<2xi8> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs0 = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f4 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %rhs0 = axis.product (%f4 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f5 = axis.factor %ti0 : !axis.shape_axis<tensor<4xi8>, 0> <4, 1>
  %lhs1 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<4xi8>, 0>, 4, 1>)
  %f6 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f7 = axis.factor %to0 : !axis.shape_axis<tensor<2xi8>, 0> <2, 1>
  %rhs1 = axis.product (%f6 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %f7 : !axis.axis_factor<!axis.shape_axis<tensor<2xi8>, 0>, 2, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<2>, !axis.factor_group<4>] [!axis.factor_group<2>, !axis.factor_group<4>]

  %in = tensor.empty() : tensor<4xi8>
  %h = distributed.Collective %in : tensor<4xi8> on %mesh_in : !axis.factor_group<2> to tensor<2xi8> on %mesh_out : !axis.factor_group<2> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<2xi8>> -> tensor<2xi8>
}

// -----

// All-gather: S = 2, n = 2, k = 1. V = 2 * 1 = 2, transfer time 2/50 = 0.04ns rounds to 0, latency = 5000 + 1000 = 6000, duration 6000, payload 2 -> 4.
// CHECK: chain: 1 steps, total duration 6000
// CHECK-NEXT: step 0: all-gather
// CHECK-NEXT: atoms: axis0.0(x2)
// CHECK-NEXT: payload: 2 -> 4
// CHECK-NEXT: latency: 6000
// CHECK-NEXT: V: [2]
// CHECK-NEXT: rho: [1]
// CHECK-NEXT: duration: 6000
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<2xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<4xi8> (the await result, 0 uses) <- step 0
module @all_gather {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

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

// -----

// Single all-to-all versus gather + slice (n = 8, S = 8, one axis). At this
// payload, every candidate's transfer time rounds to 0 or 1 nanosecond, so
// the winner turns on round count rather than volume.
//   all-to-all, pairwise: V = 8 * 7 / 8 = 7, transfer time rounds to 0, latency = 5000 + 7000 = 12000, duration 12000
//   all-to-all, Bruck:    V = 8 * 3 / 2 = 12, transfer time rounds to 0, latency 8000, duration 8000 (chosen: fewer rounds)
//   gather + slice:       gather V = 8 * 7 = 56, transfer time 56/50 = 1.12ns rounds to 1, latency 8000, duration 8001
//                         (payload 8 -> 64), then a free slice 64 -> 8
// The default run picks the Bruck all-to-all. The NOA2A run removes
// all-to-all from the candidates, which exercises the gather + slice
// alternative and its semantic check; the cost-minimizing search never picks it.
// CHECK: chain: 1 steps, total duration 8000
// CHECK-NEXT: step 0: all-to-all
// CHECK-NEXT: atoms: axis0.0(x8)
// CHECK-NEXT: payload: 8 -> 8
// CHECK-NEXT: latency: 8000
// CHECK-NEXT: V: [12]
// CHECK-NEXT: rho: [1]
// CHECK-NEXT: duration: 8000
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<8x1xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<1x8xi8> (the await result, 0 uses) <- step 0
// NOA2A: chain: 2 steps, total duration 8001
// NOA2A-NEXT: step 0: all-gather
// NOA2A-NEXT: atoms: axis0.0(x8)
// NOA2A-NEXT: payload: 8 -> 64
// NOA2A-NEXT: latency: 8000
// NOA2A-NEXT: V: [56]
// NOA2A-NEXT: rho: [1]
// NOA2A-NEXT: duration: 8001
// NOA2A-NEXT: step 1: local-slice
// NOA2A-NEXT: atoms: axis0.0(x8)
// NOA2A-NEXT: payload: 64 -> 8
// NOA2A-NEXT: latency: 0
// NOA2A-NEXT: V: [0]
// NOA2A-NEXT: rho: [0]
// NOA2A-NEXT: duration: 0
// NOA2A-NEXT: semantics: verified
// NOA2A-NEXT: input port: tensor<8x1xi8> (the collective's input_object operand) -> step 0
// NOA2A-NEXT: result port: tensor<1x8xi8> (the await result, 0 uses) <- step 1
module @all_to_all_n8 {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<8, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<8, 1>
  %ti0 = axis.getaxis tensor<8x1xi8> 0
  %ti1 = axis.getaxis tensor<8x1xi8> 1
  %to0 = axis.getaxis tensor<1x8xi8> 0
  %to1 = axis.getaxis tensor<1x8xi8> 1
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %mesh_out = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %f3 = axis.factor %ti0 : !axis.shape_axis<tensor<8x1xi8>, 0> <8, 1>
  %lhs0 = axis.product (%f3 : !axis.axis_factor<!axis.shape_axis<tensor<8x1xi8>, 0>, 8, 1>)
  %f4 = axis.factor %p0 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %rhs0 = axis.product (%f4 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %f5 = axis.factor %p0 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %lhs1 = axis.product (%f5 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %f6 = axis.factor %to1 : !axis.shape_axis<tensor<1x8xi8>, 1> <8, 1>
  %rhs1 = axis.product (%f6 : !axis.axis_factor<!axis.shape_axis<tensor<1x8xi8>, 1>, 8, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<8>, !axis.factor_group<8>] [!axis.factor_group<8>, !axis.factor_group<8>]

  %in = tensor.empty() : tensor<8x1xi8>
  %h = distributed.Collective %in : tensor<8x1xi8> on %mesh_in : !axis.factor_group<8> to tensor<1x8xi8> on %mesh_out : !axis.factor_group<8> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<1x8xi8>> -> tensor<1x8xi8>
}

// -----

// Two all-to-alls on different axes (16 = 2 x 4 devices): each keeps the payload at S = 16, so the two steps are independent and their order is the atom order. Transfer time rounds to 0 at this payload on every candidate, so each axis's winner turns on round count alone.
//   axis0 (n = 2): pairwise and Bruck both take 1 round and reach V = 8; ties go to pairwise (bandwidth-optimal), latency 5000 + 1000 = 6000
//   axis1 (n = 4): pairwise needs 3 rounds (latency 8000); Bruck needs k = 2 rounds (latency 7000) and wins on rounds alone, V = 16
//   total 6000 + 7000 = 13000
// With all-to-all disabled: gather(axis0) V = 16, transfer time rounds to 0, latency 6000; gather(axis1) at payload 16: V = 48, transfer time 48/50 = 0.96ns rounds to 1, latency 7000, duration 7001; total 13001.
// CHECK: chain: 2 steps, total duration 13000
// CHECK-NEXT: step 0: all-to-all
// CHECK-NEXT: atoms: axis0.0(x2)
// CHECK-NEXT: payload: 16 -> 16
// CHECK-NEXT: latency: 6000
// CHECK-NEXT: V: [8, 0]
// CHECK-NEXT: rho: [1, 0]
// CHECK-NEXT: duration: 6000
// CHECK-NEXT: step 1: all-to-all
// CHECK-NEXT: atoms: axis1.0(x4)
// CHECK-NEXT: payload: 16 -> 16
// CHECK-NEXT: latency: 7000
// CHECK-NEXT: V: [0, 16]
// CHECK-NEXT: rho: [0, 1]
// CHECK-NEXT: duration: 7000
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<16x1xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<2x8xi8> (the await result, 0 uses) <- step 1
// NOA2A: chain: 4 steps, total duration 13001
// NOA2A-NEXT: step 0: all-gather
// NOA2A-NEXT: atoms: axis0.0(x2)
// NOA2A-NEXT: payload: 16 -> 32
// NOA2A-NEXT: latency: 6000
// NOA2A-NEXT: V: [16, 0]
// NOA2A-NEXT: rho: [1, 0]
// NOA2A-NEXT: duration: 6000
// NOA2A-NEXT: step 1: local-slice
// NOA2A-NEXT: atoms: axis0.0(x2)
// NOA2A-NEXT: payload: 32 -> 16
// NOA2A-NEXT: latency: 0
// NOA2A-NEXT: V: [0, 0]
// NOA2A-NEXT: rho: [0, 0]
// NOA2A-NEXT: duration: 0
// NOA2A-NEXT: step 2: all-gather
// NOA2A-NEXT: atoms: axis1.0(x4)
// NOA2A-NEXT: payload: 16 -> 64
// NOA2A-NEXT: latency: 7000
// NOA2A-NEXT: V: [0, 48]
// NOA2A-NEXT: rho: [0, 1]
// NOA2A-NEXT: duration: 7001
// NOA2A-NEXT: step 3: local-slice
// NOA2A-NEXT: atoms: axis1.0(x4)
// NOA2A-NEXT: payload: 64 -> 16
// NOA2A-NEXT: latency: 0
// NOA2A-NEXT: V: [0, 0]
// NOA2A-NEXT: rho: [0, 0]
// NOA2A-NEXT: duration: 0
// NOA2A-NEXT: semantics: verified
// NOA2A-NEXT: input port: tensor<16x1xi8> (the collective's input_object operand) -> step 0
// NOA2A-NEXT: result port: tensor<2x8xi8> (the await result, 0 uses) <- step 3
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

// -----

// Slice before gather on different axes. Input tile 2x16 (S = 32); axis1 (n = 4) slices dim 1 (free, 32 -> 8), axis0 (n = 2) gathers dim 0. The slice is free and shrinks the payload, so it runs first and the gather moves 8 bytes instead of 32:
//   gather after the slice: V = 8 * 1 = 8, transfer time 8/50 = 0.16ns rounds to 0, latency 6000, duration 6000.
//   (gather first would move V = 32, transfer time 32/50 = 0.64ns rounds to 1, duration 6001: one nanosecond more.)
// CHECK: chain: 2 steps, total duration 6000
// CHECK-NEXT: step 0: local-slice
// CHECK-NEXT: atoms: axis1.0(x4)
// CHECK-NEXT: payload: 32 -> 8
// CHECK-NEXT: latency: 0
// CHECK-NEXT: V: [0, 0]
// CHECK-NEXT: rho: [0, 0]
// CHECK-NEXT: duration: 0
// CHECK-NEXT: step 1: all-gather
// CHECK-NEXT: atoms: axis0.0(x2)
// CHECK-NEXT: payload: 8 -> 16
// CHECK-NEXT: latency: 6000
// CHECK-NEXT: V: [8, 0]
// CHECK-NEXT: rho: [1, 0]
// CHECK-NEXT: duration: 6000
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<2x16xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<4x4xi8> (the await result, 0 uses) <- step 1
module @slice_then_gather {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<4, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<4, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %rep4 = distributed.ReplicationAxis 4 : !distributed.replication_axis<4>
  %ti0 = axis.getaxis tensor<2x16xi8> 0
  %ti1 = axis.getaxis tensor<2x16xi8> 1
  %to0 = axis.getaxis tensor<4x4xi8> 0
  %to1 = axis.getaxis tensor<4x4xi8> 1
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %f4 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %mesh_out = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>, %f4 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f5 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %f6 = axis.factor %ti0 : !axis.shape_axis<tensor<2x16xi8>, 0> <2, 1>
  %lhs0 = axis.product (%f5 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>, %f6 : !axis.axis_factor<!axis.shape_axis<tensor<2x16xi8>, 0>, 2, 1>)
  %f7 = axis.factor %to0 : !axis.shape_axis<tensor<4x4xi8>, 0> <4, 1>
  %rhs0 = axis.product (%f7 : !axis.axis_factor<!axis.shape_axis<tensor<4x4xi8>, 0>, 4, 1>)
  %f8 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %lhs1 = axis.product (%f8 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f9 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %rhs1 = axis.product (%f9 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>)
  %f10 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %lhs2 = axis.product (%f10 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f11 = axis.factor %rep4 : !distributed.replication_axis<4> <4, 1>
  %rhs2 = axis.product (%f11 : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
  %f12 = axis.factor %ti1 : !axis.shape_axis<tensor<2x16xi8>, 1> <16, 1>
  %lhs3 = axis.product (%f12 : !axis.axis_factor<!axis.shape_axis<tensor<2x16xi8>, 1>, 16, 1>)
  %f13 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %f14 = axis.factor %to1 : !axis.shape_axis<tensor<4x4xi8>, 1> <4, 1>
  %rhs3 = axis.product (%f13 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>, %f14 : !axis.axis_factor<!axis.shape_axis<tensor<4x4xi8>, 1>, 4, 1>)
  %map = axis.map %lhs0, %lhs1, %lhs2, %lhs3 to %rhs0, %rhs1, %rhs2, %rhs3 : [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<16>] [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<16>]

  %in = tensor.empty() : tensor<2x16xi8>
  %h = distributed.Collective %in : tensor<2x16xi8> on %mesh_in : !axis.factor_group<8> to tensor<4x4xi8> on %mesh_out : !axis.factor_group<8> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4x4xi8>> -> tensor<4x4xi8>
}

// -----

// Permute of two equal-extent atoms on different axes swapping digits: S = 4, both axes have a nonzero change fraction, and a moving device sends its whole payload rather than the average across all devices, so V = [4, 4]; transfer time 4/50 = 0.08ns rounds to 0. Latency is launch plus both axes' round latencies: 5000 + 1000 + 1000 = 7000, duration 7000.
// CHECK: chain: 1 steps, total duration 7000
// CHECK-NEXT: step 0: permute
// CHECK-NEXT: atoms: axis0.0(x2) axis1.0(x2)
// CHECK-NEXT: payload: 4 -> 4
// CHECK-NEXT: latency: 7000
// CHECK-NEXT: V: [4, 4]
// CHECK-NEXT: rho: [1, 1]
// CHECK-NEXT: duration: 7000
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<4xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<4xi8> (the await result, 0 uses) <- step 0
module @permute_two_axes {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %ti0 = axis.getaxis tensor<4xi8> 0
  %to0 = axis.getaxis tensor<4xi8> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f4 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f4 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f5 = axis.factor %ti0 : !axis.shape_axis<tensor<4xi8>, 0> <4, 1>
  %lhs0 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<4xi8>, 0>, 4, 1>)
  %f6 = axis.factor %to0 : !axis.shape_axis<tensor<4xi8>, 0> <4, 1>
  %rhs0 = axis.product (%f6 : !axis.axis_factor<!axis.shape_axis<tensor<4xi8>, 0>, 4, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %lhs1 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f8 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs1 = axis.product (%f8 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f9 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs2 = axis.product (%f9 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f10 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %rhs2 = axis.product (%f10 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %map = axis.map %lhs0, %lhs1, %lhs2 to %rhs0, %rhs1, %rhs2 : [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<4xi8>
  %h = distributed.Collective %in : tensor<4xi8> on %mesh_in : !axis.factor_group<4> to tensor<4xi8> on %mesh_out : !axis.factor_group<4> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xi8>> -> tensor<4xi8>
}

// -----

// Permute and a gather on a third axis (S = 4). At this payload every
// transfer time rounds to 0, so permute-first and gather-first cost exactly
// the same total (13000): permute (7000) then gather at payload 8 (6000), or
// gather (6000) then permute at payload 8 (7000). The DP's tie-break keeps
// the earlier-enumerated candidate, which here is gather-first.
// CHECK: chain: 2 steps, total duration 13000
// CHECK-NEXT: step 0: all-gather
// CHECK-NEXT: atoms: axis2.0(x2)
// CHECK-NEXT: payload: 4 -> 8
// CHECK-NEXT: latency: 6000
// CHECK-NEXT: V: [0, 0, 4]
// CHECK-NEXT: rho: [0, 0, 1]
// CHECK-NEXT: duration: 6000
// CHECK-NEXT: step 1: permute
// CHECK-NEXT: atoms: axis0.0(x2) axis1.0(x2)
// CHECK-NEXT: payload: 8 -> 8
// CHECK-NEXT: latency: 7000
// CHECK-NEXT: V: [8, 8, 0]
// CHECK-NEXT: rho: [1, 1, 0]
// CHECK-NEXT: duration: 7000
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<4xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<4x2xi8> (the await result, 0 uses) <- step 1
module @permute_then_gather {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0, %p1, %p2 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<4xi8> 0
  %to0 = axis.getaxis tensor<4x2xi8> 0
  %to1 = axis.getaxis tensor<4x2xi8> 1
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f3 = axis.factor %p2 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f4 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %f5 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f6 = axis.factor %p2 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f4 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>, %f5 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f6 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f7 = axis.factor %ti0 : !axis.shape_axis<tensor<4xi8>, 0> <4, 1>
  %lhs0 = axis.product (%f7 : !axis.axis_factor<!axis.shape_axis<tensor<4xi8>, 0>, 4, 1>)
  %f8 = axis.factor %to0 : !axis.shape_axis<tensor<4x2xi8>, 0> <4, 1>
  %rhs0 = axis.product (%f8 : !axis.axis_factor<!axis.shape_axis<tensor<4x2xi8>, 0>, 4, 1>)
  %f9 = axis.factor %p2 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs1 = axis.product (%f9 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f10 = axis.factor %to1 : !axis.shape_axis<tensor<4x2xi8>, 1> <2, 1>
  %rhs1 = axis.product (%f10 : !axis.axis_factor<!axis.shape_axis<tensor<4x2xi8>, 1>, 2, 1>)
  %f11 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %lhs2 = axis.product (%f11 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f12 = axis.factor %p2 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs2 = axis.product (%f12 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f13 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %lhs3 = axis.product (%f13 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>)
  %f14 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %rhs3 = axis.product (%f14 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f15 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %lhs4 = axis.product (%f15 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f16 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %rhs4 = axis.product (%f16 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>)
  %map = axis.map %lhs0, %lhs1, %lhs2, %lhs3, %lhs4 to %rhs0, %rhs1, %rhs2, %rhs3, %rhs4 : [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<4xi8>
  %h = distributed.Collective %in : tensor<4xi8> on %mesh_in : !axis.factor_group<8> to tensor<4x2xi8> on %mesh_out : !axis.factor_group<8> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4x2xi8>> -> tensor<4x2xi8>
}

// -----

// Identity collective: the mesh atom feeds itself, so the chain is empty. The ports connect directly: the input port feeds the result port, whose two uses (the add below) depend on the input.
// CHECK: chain: 0 steps, total duration 0
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<4xi8> (the collective's input_object operand) -> result port
// CHECK-NEXT: result port: tensor<4xi8> (the await result, 2 uses) <- input port
module @identity_empty_chain {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %ti0 = axis.getaxis tensor<4xi8> 0
  %to0 = axis.getaxis tensor<4xi8> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f3 = axis.factor %ti0 : !axis.shape_axis<tensor<4xi8>, 0> <4, 1>
  %lhs0 = axis.product (%f3 : !axis.axis_factor<!axis.shape_axis<tensor<4xi8>, 0>, 4, 1>)
  %f4 = axis.factor %to0 : !axis.shape_axis<tensor<4xi8>, 0> <4, 1>
  %rhs0 = axis.product (%f4 : !axis.axis_factor<!axis.shape_axis<tensor<4xi8>, 0>, 4, 1>)
  %f5 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs1 = axis.product (%f5 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f6 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs1 = axis.product (%f6 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<4xi8>
  %h = distributed.Collective %in : tensor<4xi8> on %mesh_in : !axis.factor_group<2> to tensor<4xi8> on %mesh_out : !axis.factor_group<2> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xi8>> -> tensor<4xi8>
  %u = stablehlo.add %v, %v : tensor<4xi8>
}

// -----

// Ports. plan is built from the real op, and its boundary values are read off the op rather than assumed: the first step's dependency is the collective's input_object operand (type tensor<2xi8> here), and the last step produces the distributed.Await result, whose two uses (stablehlo.add %v, %v) are the consumers of the chain. The async handle between the Collective and the Await is internal to the chain. A scheduler splices the chain in by (1) making step 0 depend on the definition of the input port and (2) making every user of the result port depend on the last step (or, for an empty chain, on the input port).
// CHECK: chain: 1 steps, total duration 6000
// CHECK-NEXT: step 0: all-gather
// CHECK-NEXT: atoms: axis0.0(x2)
// CHECK-NEXT: payload: 2 -> 4
// CHECK-NEXT: latency: 6000
// CHECK-NEXT: V: [2]
// CHECK-NEXT: rho: [1]
// CHECK-NEXT: duration: 6000
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<2xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<4xi8> (the await result, 2 uses) <- step 0
module @ports_with_consumer {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

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
  %u = stablehlo.add %v, %v : tensor<4xi8>
}

// -----

// All-reduce over f32 (n = 2, S = 16 bytes, k = 1). Both algorithms move the
// same V = 16 bytes here (transfer time rounds to 0 either way), so recursive
// doubling wins on rounds alone:
//   halving + doubling: 2 rounds, latency 5000 + 2000 = 7000, duration 7000
//   recursive doubling: 1 round, latency 5000 + 1000 = 6000, duration 6000 (chosen)
// More reduction cases are in decompose_collective_reduce.mlir.
// CHECK: chain: 1 steps, total duration 6000
// CHECK-NEXT: step 0: all-reduce
// CHECK-NEXT: atoms: axis0.0(x2)
// CHECK-NEXT: payload: 16 -> 16
// CHECK-NEXT: latency: 6000
// CHECK-NEXT: V: [16]
// CHECK-NEXT: rho: [1]
// CHECK-NEXT: duration: 6000
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<4xf32> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<4xf32> (the await result, 0 uses) <- step 0
module @all_reduce_f32 {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<4xf32> 0
  %to0 = axis.getaxis tensor<4xf32> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %red0 = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f4 = axis.factor %ti0 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %lhs0 = axis.product (%f4 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %f5 = axis.factor %to0 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %rhs0 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %f6 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %lhs1 = axis.product (%f6 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs1 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<4xf32>
  %h = distributed.Collective %in : tensor<4xf32> on %mesh_in : !axis.factor_group<2> to tensor<4xf32> on %mesh_out : !axis.factor_group<2> reduces (%red0 : !axis.factor_group<2>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
}

// -----

// Mixed mesh row, (Mesh, Tile) next to (Tile, Mesh): the input tile digit is sliced onto mesh0, mesh0's digit goes to mesh1, and mesh1's digit goes to the output tile. Each atom both releases its own digit and places another, so each is one all-to-all (a fused exchange) instead of a gather followed by a free slice. S = 2, n = 2; transfer time rounds to 0 at this payload for every candidate.
//   all-to-all mesh0 (places the input tile digit): 1 round, latency 5000 + 1000 = 6000, payload 2 -> 2
//   all-to-all mesh1 (places mesh0's digit, which mesh0's exchange made local): the same, 6000
// Total 12000. The half-split (gather + slice per atom, 6000 each) also totals 12000 at this payload; decompose_collective_meshcoupled.mlir keeps it as the baseline.
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
module @mixed_mesh_row {
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