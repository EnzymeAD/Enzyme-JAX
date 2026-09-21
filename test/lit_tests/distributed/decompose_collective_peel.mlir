// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan=chain=true %s 2>&1 >/dev/null | FileCheck %s
// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true disable-peel-variants=true" %s 2>&1 >/dev/null | FileCheck %s

// Peel (D17) under the default, uniform (bandwidth 1 on every axis) cost
// parameters: peeling a (Reduced, Replicate) atom never beats the flat
// all-reduce here, so plan() stays flat with or without disable-peel-variants
// and both RUN lines below produce byte-identical CHECK matches. See
// decompose_collective_peel_bandwidth.mlir for the cases where a peel wins.
//
// Two independent reduced atoms (S = 64): axis0 (extent 4) is (Reduced,
// Replicate), axis1 (extent 4) is (Reduced, Tile) scattering onto a tile
// atom. The cost-minimizing order is axis1's reduce-scatter (shrinking the
// shared payload) before axis0's flat all-reduce, exactly as in
// decompose_collective_reduce.mlir's two-axis case:
//   reduce-scatter axis1 at S = 64: V = 64 * 3 / 4 = 48, k = 2, latency 0.12, duration 48.12; payload 64 -> 16
//   all-reduce axis0 at the shrunk payload 16: halving+doubling V = 2*16*3/4 = 24, latency 0.1+0.04 = 0.14, duration 24.14 (cheaper than recursive doubling V = 16*2 = 32, latency 0.12, duration 32.12)
// Total 72.26. Peeling axis0 instead (self-scatter, then axis1's
// reduce-scatter, then the closing all-gather) would cost exactly one extra
// launch latency more for the same volume (D17's two steps reproduce the
// flat all-reduce's halving-doubling volume and rounds when nothing runs
// between them), so the DP never selects it: peeling only pays off when
// something else benefits from running in the gap at a discount that exceeds
// that extra launch, which uniform bandwidth cannot provide (every axis costs
// the same either way).
// CHECK: chain: 2 steps, total duration 72.26
// CHECK-NEXT: step 0: reduce-scatter
// CHECK-NEXT: atoms: axis1.0(x4)
// CHECK-NEXT: payload: 64 -> 16
// CHECK-NEXT: latency: 0.12
// CHECK-NEXT: V: [0, 48]
// CHECK-NEXT: rho: [0, 1]
// CHECK-NEXT: duration: 48.12
// CHECK-NEXT: step 1: all-reduce
// CHECK-NEXT: atoms: axis0.0(x4)
// CHECK-NEXT: payload: 16 -> 16
// CHECK-NEXT: latency: 0.14
// CHECK-NEXT: V: [24, 0]
// CHECK-NEXT: rho: [1, 0]
// CHECK-NEXT: duration: 24.14
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<64xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<16xi8> (the await result, 0 uses) <- step 1
module @peel_not_cheaper_uniform_bandwidth {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 4>, !distributed.physical_comm_axis<4, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 4>, !distributed.physical_comm_axis<4, 1>
  %rep4 = distributed.ReplicationAxis 4 : !distributed.replication_axis<4>
  %ti0 = axis.getaxis tensor<64xi8> 0
  %to0 = axis.getaxis tensor<16xi8> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %f4 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %mesh_out = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>, %f4 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f5 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %f6 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %red0 = axis.product (%f5 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>, %f6 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f7 = axis.factor %ti0 : !axis.shape_axis<tensor<64xi8>, 0> <64, 1>
  %lhs0 = axis.product (%f7 : !axis.axis_factor<!axis.shape_axis<tensor<64xi8>, 0>, 64, 1>)
  %f8 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %f9 = axis.factor %to0 : !axis.shape_axis<tensor<16xi8>, 0> <16, 1>
  %rhs0 = axis.product (%f8 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>, %f9 : !axis.axis_factor<!axis.shape_axis<tensor<16xi8>, 0>, 16, 1>)
  %f10 = axis.factor %rep4 : !distributed.replication_axis<4> <4, 1>
  %lhs1 = axis.product (%f10 : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
  %f11 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %rhs1 = axis.product (%f11 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<64>, !axis.factor_group<4>] [!axis.factor_group<64>, !axis.factor_group<4>]

  %in = tensor.empty() : tensor<64xi8>
  %h = distributed.Collective %in : tensor<64xi8> on %mesh_in : !axis.factor_group<16> to tensor<16xi8> on %mesh_out : !axis.factor_group<16> reduces (%red0 : !axis.factor_group<16>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<i8>, %rhs_v: tensor<i8>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<i8>
    stablehlo.return %r : tensor<i8>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<16xi8>> -> tensor<16xi8>
}

// -----

// A single reduced atom on its own (n = 8, S = 8, as in
// decompose_collective_reduce.mlir's all_reduce_n8): peeling has nothing to
// interleave with, so it can only add the peel's extra launch latency over
// the flat all-reduce. The DP keeps the flat all-reduce (duration 14.16):
// the degenerate case where D17's premise, that some other unit benefits
// from the shrunk payload, does not hold.
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
module @peel_no_other_unit_to_interleave_with {
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
