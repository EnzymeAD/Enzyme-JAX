// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan=chain=true %s 2>&1 >/dev/null | FileCheck %s
// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true disable-all-to-all=true" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOA2A

// distributed-print-collective-plan=chain=true reports the primitive chain
// the collective decomposer chose for each collective: one block per step
// (kind, atoms, payload bytes before -> after, latency, per-axis volume V and
// demand rho, isolated duration), then `semantics: verified` when the
// independent symbolic checker confirms the chain realizes the collective,
// then the two ports that attach the chain to the surrounding program.
//
// Hand computation uses the default parameters: bandwidth 1 on every axis,
// round latency 0.01, launch latency 0.1. Formulas (CollectiveCost.h), for a
// group of n on one axis with per-device payload S and k = ceil(log2 n):
//   all-gather:  V = S (n - 1), latency = 0.1 + 0.01 k, payload out n S
//   all-to-all:  pairwise V = S (n - 1) / n, latency = 0.1 + 0.01 (n - 1);
//                Bruck V = S k / 2, latency = 0.1 + 0.01 k; the cheaper wins
//   permute:     V[a] = S (1 - 1/e) on each axis, latency = 0.1 + 0.01 x (axes)
//   local slice: free, payload S / e
// duration = latency + max_a V[a] / BW[a]. Inputs are not written in atomic
// form; every RUN line atomizes them first. Element type i8, so a payload in
// bytes equals its element count.

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

// All-gather: S = 2, n = 2, k = 1. V = 2 * 1 = 2, latency = 0.1 + 0.01 = 0.11, duration 2.11, payload 2 -> 4.
// CHECK: chain: 1 steps, total duration 2.11
// CHECK-NEXT: step 0: all-gather
// CHECK-NEXT: atoms: axis0.0(x2)
// CHECK-NEXT: payload: 2 -> 4
// CHECK-NEXT: latency: 0.11
// CHECK-NEXT: V: [2]
// CHECK-NEXT: rho: [1]
// CHECK-NEXT: duration: 2.11
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

// Single all-to-all versus gather + slice (n = 8, S = 8, one axis).
//   all-to-all, pairwise: V = 8 * 7 / 8 = 7, latency = 0.1 + 7 * 0.01 = 0.17, duration 7.17
//   all-to-all, Bruck:    V = 8 * 3 / 2 = 12, latency 0.13, duration 12.13 (not chosen)
//   gather + slice:       gather V = 8 * 7 = 56, latency 0.1 + 3 * 0.01 = 0.13, duration 56.13
//                         (payload 8 -> 64), then a free slice 64 -> 8
// The default run picks the all-to-all, 7.8x cheaper. The NOA2A run removes
// all-to-all from the candidates, which exercises the gather + slice
// alternative and its semantic check; the cost-minimizing search never picks it.
// CHECK: chain: 1 steps, total duration 7.17
// CHECK-NEXT: step 0: all-to-all
// CHECK-NEXT: atoms: axis0.0(x8)
// CHECK-NEXT: payload: 8 -> 8
// CHECK-NEXT: latency: 0.17
// CHECK-NEXT: V: [7]
// CHECK-NEXT: rho: [1]
// CHECK-NEXT: duration: 7.17
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<8x1xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<1x8xi8> (the await result, 0 uses) <- step 0
// NOA2A: chain: 2 steps, total duration 56.13
// NOA2A-NEXT: step 0: all-gather
// NOA2A-NEXT: atoms: axis0.0(x8)
// NOA2A-NEXT: payload: 8 -> 64
// NOA2A-NEXT: latency: 0.13
// NOA2A-NEXT: V: [56]
// NOA2A-NEXT: rho: [1]
// NOA2A-NEXT: duration: 56.13
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

// Two all-to-alls on different axes (16 = 2 x 4 devices): each keeps the payload at S = 16, so the two steps are independent and their order is the atom order.
//   axis0 (n = 2): pairwise V = 16 * 1 / 2 = 8, latency 0.1 + 0.01 = 0.11 -> 8.11 (Bruck ties on volume and rounds, pairwise wins ties)
//   axis1 (n = 4): pairwise V = 16 * 3 / 4 = 12, latency 0.1 + 3 * 0.01 = 0.13 -> 12.13; Bruck V = 16 * 2 / 2 = 16, latency 0.12 -> 16.12
//   total 8.11 + 12.13 = 20.24
// With all-to-all disabled: gather(axis0) V = 16 -> 16.11, slice, gather(axis1) at payload 16: V = 48, latency 0.12 -> 48.12, slice; total 64.23.
// CHECK: chain: 2 steps, total duration 20.24
// CHECK-NEXT: step 0: all-to-all
// CHECK-NEXT: atoms: axis0.0(x2)
// CHECK-NEXT: payload: 16 -> 16
// CHECK-NEXT: latency: 0.11
// CHECK-NEXT: V: [8, 0]
// CHECK-NEXT: rho: [1, 0]
// CHECK-NEXT: duration: 8.11
// CHECK-NEXT: step 1: all-to-all
// CHECK-NEXT: atoms: axis1.0(x4)
// CHECK-NEXT: payload: 16 -> 16
// CHECK-NEXT: latency: 0.13
// CHECK-NEXT: V: [0, 12]
// CHECK-NEXT: rho: [0, 1]
// CHECK-NEXT: duration: 12.13
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<16x1xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<2x8xi8> (the await result, 0 uses) <- step 1
// NOA2A: chain: 4 steps, total duration 64.23
// NOA2A-NEXT: step 0: all-gather
// NOA2A-NEXT: atoms: axis0.0(x2)
// NOA2A-NEXT: payload: 16 -> 32
// NOA2A-NEXT: latency: 0.11
// NOA2A-NEXT: V: [16, 0]
// NOA2A-NEXT: rho: [1, 0]
// NOA2A-NEXT: duration: 16.11
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
// NOA2A-NEXT: latency: 0.12
// NOA2A-NEXT: V: [0, 48]
// NOA2A-NEXT: rho: [0, 1]
// NOA2A-NEXT: duration: 48.12
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
//   gather after the slice: V = 8 * 1 = 8, latency 0.11, duration 8.11, payload 8 -> 16.
//   (gather first would cost 32 * 1 + 0.11 = 32.11.)
// CHECK: chain: 2 steps, total duration 8.11
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
// CHECK-NEXT: latency: 0.11
// CHECK-NEXT: V: [8, 0]
// CHECK-NEXT: rho: [1, 0]
// CHECK-NEXT: duration: 8.11
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

// Permute of two equal-extent atoms on different axes swapping digits: S = 4, change fraction 1 - 1/2 = 0.5 on each axis, so V = [2, 2]. Latency is launch plus both axes' round latencies: 0.1 + 0.01 + 0.01 = 0.12, duration 0.12 + 2 = 2.12.
// CHECK: chain: 1 steps, total duration 2.12
// CHECK-NEXT: step 0: permute
// CHECK-NEXT: atoms: axis0.0(x2) axis1.0(x2)
// CHECK-NEXT: payload: 4 -> 4
// CHECK-NEXT: latency: 0.12
// CHECK-NEXT: V: [2, 2]
// CHECK-NEXT: rho: [1, 1]
// CHECK-NEXT: duration: 2.12
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

// Permute and a gather on a third axis (S = 4). The permute moves S * 0.5 = 2 per axis (0.12 + 2 = 2.12) and the gather moves S * 1 = 4 (0.11 + 4 = 4.11). Permute first costs 2.12 + 4.11 = 6.23; gather first costs 4.11 + (permute at payload 8: 0.12 + 4) = 8.23, so the DP puts the permute first.
// CHECK: chain: 2 steps, total duration 6.23
// CHECK-NEXT: step 0: permute
// CHECK-NEXT: atoms: axis0.0(x2) axis1.0(x2)
// CHECK-NEXT: payload: 4 -> 4
// CHECK-NEXT: latency: 0.12
// CHECK-NEXT: V: [2, 2, 0]
// CHECK-NEXT: rho: [1, 1, 0]
// CHECK-NEXT: duration: 2.12
// CHECK-NEXT: step 1: all-gather
// CHECK-NEXT: atoms: axis2.0(x2)
// CHECK-NEXT: payload: 4 -> 8
// CHECK-NEXT: latency: 0.11
// CHECK-NEXT: V: [0, 0, 4]
// CHECK-NEXT: rho: [0, 0, 1]
// CHECK-NEXT: duration: 4.11
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
// CHECK: chain: 1 steps, total duration 2.11
// CHECK-NEXT: step 0: all-gather
// CHECK-NEXT: atoms: axis0.0(x2)
// CHECK-NEXT: payload: 2 -> 4
// CHECK-NEXT: latency: 0.11
// CHECK-NEXT: V: [2]
// CHECK-NEXT: rho: [1]
// CHECK-NEXT: duration: 2.11
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

// All-reduce over f32 (n = 2, S = 16 bytes, k = 1). Recursive doubling is the cheaper algorithm here:
//   halving + doubling: 2 rounds, V = 2 * 16 * 1 / 2 = 16, latency 0.12, duration 16.12
//   recursive doubling: 1 round, V = 16 * 1 = 16, latency 0.11, duration 16.11 (chosen)
// More reduction cases are in decompose_collective_reduce.mlir.
// CHECK: chain: 1 steps, total duration 16.11
// CHECK-NEXT: step 0: all-reduce
// CHECK-NEXT: atoms: axis0.0(x2)
// CHECK-NEXT: payload: 16 -> 16
// CHECK-NEXT: latency: 0.11
// CHECK-NEXT: V: [16]
// CHECK-NEXT: rho: [1]
// CHECK-NEXT: duration: 16.11
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

// Mixed mesh row, (Mesh, Tile) next to (Tile, Mesh): the input tile digit is sliced onto mesh0, mesh0's digit goes to mesh1, and mesh1's digit goes to the output tile. Each atom both releases its own digit and places another, so each is one all-to-all (a fused exchange) instead of a gather followed by a free slice. S = 2, n = 2.
//   all-to-all mesh0 (places the input tile digit): pairwise V = 2 * 1 / 2 = 1, latency 0.1 + 0.01 = 0.11, duration 1.11, payload 2 -> 2
//   all-to-all mesh1 (places mesh0's digit, which mesh0's exchange made local): the same, 1.11
// Total 2.22. The half-split (gather + slice per atom, 2.11 each) is 4.22; decompose_collective_meshcoupled.mlir keeps it as the baseline.
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