// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true bandwidths=4,1" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=FAST0
// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true bandwidths=1,4" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=FAST1
// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true bandwidths=4,1 disable-peel-variants=true" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=FLAT

// Peel (D17) break-even under non-uniform bandwidth. Round latency 0.01 and
// launch latency 0.1 as elsewhere; BW as given by bandwidths=.
//
// Two independent reduced atoms (S = 64), the same collective as
// decompose_collective_peel.mlir's uniform-bandwidth case: axis0 (extent 4)
// is (Reduced, Replicate), axis1 (extent 4) is (Reduced, Tile). With axis0
// four times faster (bandwidths=4,1), peeling axis0 around axis1's
// reduce-scatter is cheaper than the flat order (axis1's reduce-scatter, then
// axis0's flat all-reduce on the shrunk payload):
//   peel:  self-scatter axis0 at S = 64: V = 64*3/4 = 48, BW 4, latency 0.12, duration 0.12+48/4 = 12.12; payload 64 -> 16
//          reduce-scatter axis1 at 16: V = 16*3/4 = 12, BW 1, latency 0.12, duration 12.12; payload 16 -> 4
//          all-gather axis0 (closing) at 4: V = 4*3 = 12, BW 4, latency 0.12, duration 0.12+12/4 = 3.12; payload 4 -> 16
//          total 27.36
//   flat:  reduce-scatter axis1 at S = 64: V = 48, BW 1, latency 0.12, duration 48.12; payload 64 -> 16
//          all-reduce axis0 at 16: halving+doubling V = 2*16*3/4 = 24, BW 4, latency 0.14, duration 0.14+24/4 = 6.14
//          total 54.26 (FLAT, disable-peel-variants=true)
// The peel moves the same total bytes as the flat all-reduce would (D17), but
// pays for axis1's reduce-scatter at 1/4 the payload instead of the full 64,
// at the cost of one extra launch latency for splitting axis0's step in two;
// here the bandwidth gap makes that trade pay off by nearly a factor of 2.
// FAST0: chain: 3 steps, total duration 27.36
// FAST0-NEXT: step 0: reduce-scatter
// FAST0-NEXT: atoms: axis0.0(x4)
// FAST0-NEXT: payload: 64 -> 16
// FAST0-NEXT: latency: 0.12
// FAST0-NEXT: V: [48, 0]
// FAST0-NEXT: rho: [1, 0]
// FAST0-NEXT: duration: 12.12
// FAST0-NEXT: step 1: reduce-scatter
// FAST0-NEXT: atoms: axis1.0(x4)
// FAST0-NEXT: payload: 16 -> 4
// FAST0-NEXT: latency: 0.12
// FAST0-NEXT: V: [0, 12]
// FAST0-NEXT: rho: [0, 1]
// FAST0-NEXT: duration: 12.12
// FAST0-NEXT: step 2: all-gather
// FAST0-NEXT: atoms: axis0.0(x4)
// FAST0-NEXT: payload: 4 -> 16
// FAST0-NEXT: latency: 0.12
// FAST0-NEXT: V: [12, 0]
// FAST0-NEXT: rho: [1, 0]
// FAST0-NEXT: duration: 3.12
// FAST0-NEXT: semantics: verified
// FAST0-NEXT: input port: tensor<64xi8> (the collective's input_object operand) -> step 0
// FAST0-NEXT: result port: tensor<16xi8> (the await result, 0 uses) <- step 2
// FLAT: chain: 2 steps, total duration 54.26
// FLAT-NEXT: step 0: reduce-scatter
// FLAT-NEXT: atoms: axis1.0(x4)
// FLAT-NEXT: payload: 64 -> 16
// FLAT-NEXT: latency: 0.12
// FLAT-NEXT: V: [0, 48]
// FLAT-NEXT: rho: [0, 1]
// FLAT-NEXT: duration: 48.12
// FLAT-NEXT: step 1: all-reduce
// FLAT-NEXT: atoms: axis0.0(x4)
// FLAT-NEXT: payload: 16 -> 16
// FLAT-NEXT: latency: 0.14
// FLAT-NEXT: V: [24, 0]
// FLAT-NEXT: rho: [1, 0]
// FLAT-NEXT: duration: 6.14
// FLAT-NEXT: semantics: verified
// FLAT-NEXT: input port: tensor<64xi8> (the collective's input_object operand) -> step 0
// FLAT-NEXT: result port: tensor<16xi8> (the await result, 0 uses) <- step 1
module @peel_break_even {
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

// Fast axis first: which atom peels follows bandwidth, not atom order. Both
// axis0 and axis1 (extent 4 each) are (Reduced, Replicate) on the same
// collective (S = 64); with nothing else to shrink the payload, the flat
// baseline is order-independent (both all-reduces run at full S, matching
// decompose_collective_bandwidth.mlir's own two-axis baselines):
// all_reduce(64,4,BW) halving+doubling V = 2*64*3/4 = 96, k = 2,
// latency 0.14, duration 0.14+96/BW.
//   bandwidths=4,1: baseline = (0.14+96/4) + (0.14+96/1) = 24.14+96.14 = 120.28
//   peeling the fast atom (axis0): self-scatter(64,4,BW4)=12.12, payload->16;
//     flat all-reduce(16,4,BW1) at the shrunk payload = 24.14; closing
//     all-gather(16,4,BW4)=12.12, payload->64; total 48.38
//   peeling the slow atom (axis1) instead is worse (its own scatter/gather
//     pay the slow bandwidth): self-scatter(64,4,BW1)=48.12; flat
//     all-reduce(16,4,BW4) at 16 = 6.14; all-gather(16,4,BW1)=48.12; total
//     102.38
// So the DP always peels whichever atom is on the faster axis, mirroring
// decompose_collective_bandwidth.mlir's order flip when bandwidths swap.
// FAST0: chain: 3 steps, total duration 48.38
// FAST0-NEXT: step 0: reduce-scatter
// FAST0-NEXT: atoms: axis0.0(x4)
// FAST0-NEXT: payload: 64 -> 16
// FAST0-NEXT: latency: 0.12
// FAST0-NEXT: V: [48, 0]
// FAST0-NEXT: rho: [1, 0]
// FAST0-NEXT: duration: 12.12
// FAST0-NEXT: step 1: all-reduce
// FAST0-NEXT: atoms: axis1.0(x4)
// FAST0-NEXT: payload: 16 -> 16
// FAST0-NEXT: latency: 0.14
// FAST0-NEXT: V: [0, 24]
// FAST0-NEXT: rho: [0, 1]
// FAST0-NEXT: duration: 24.14
// FAST0-NEXT: step 2: all-gather
// FAST0-NEXT: atoms: axis0.0(x4)
// FAST0-NEXT: payload: 16 -> 64
// FAST0-NEXT: latency: 0.12
// FAST0-NEXT: V: [48, 0]
// FAST0-NEXT: rho: [1, 0]
// FAST0-NEXT: duration: 12.12
// FAST0-NEXT: semantics: verified
// FAST0-NEXT: input port: tensor<64xi8> (the collective's input_object operand) -> step 0
// FAST0-NEXT: result port: tensor<64xi8> (the await result, 0 uses) <- step 2
// FAST1: chain: 3 steps, total duration 48.38
// FAST1-NEXT: step 0: reduce-scatter
// FAST1-NEXT: atoms: axis1.0(x4)
// FAST1-NEXT: payload: 64 -> 16
// FAST1-NEXT: latency: 0.12
// FAST1-NEXT: V: [0, 48]
// FAST1-NEXT: rho: [0, 1]
// FAST1-NEXT: duration: 12.12
// FAST1-NEXT: step 1: all-reduce
// FAST1-NEXT: atoms: axis0.0(x4)
// FAST1-NEXT: payload: 16 -> 16
// FAST1-NEXT: latency: 0.14
// FAST1-NEXT: V: [24, 0]
// FAST1-NEXT: rho: [1, 0]
// FAST1-NEXT: duration: 24.14
// FAST1-NEXT: step 2: all-gather
// FAST1-NEXT: atoms: axis1.0(x4)
// FAST1-NEXT: payload: 16 -> 64
// FAST1-NEXT: latency: 0.12
// FAST1-NEXT: V: [0, 48]
// FAST1-NEXT: rho: [0, 1]
// FAST1-NEXT: duration: 12.12
// FAST1-NEXT: semantics: verified
// FAST1-NEXT: input port: tensor<64xi8> (the collective's input_object operand) -> step 0
// FAST1-NEXT: result port: tensor<64xi8> (the await result, 0 uses) <- step 2
module @peel_fast_axis_first {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 4>, !distributed.physical_comm_axis<4, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 4>, !distributed.physical_comm_axis<4, 1>
  %rep4_0 = distributed.ReplicationAxis 4 : !distributed.replication_axis<4>
  %rep4_1 = distributed.ReplicationAxis 4 : !distributed.replication_axis<4>
  %ti0 = axis.getaxis tensor<64xi8> 0
  %to0 = axis.getaxis tensor<64xi8> 0
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
  %f8 = axis.factor %to0 : !axis.shape_axis<tensor<64xi8>, 0> <64, 1>
  %rhs0 = axis.product (%f8 : !axis.axis_factor<!axis.shape_axis<tensor<64xi8>, 0>, 64, 1>)
  %f9 = axis.factor %rep4_0 : !distributed.replication_axis<4> <4, 1>
  %lhs1 = axis.product (%f9 : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
  %f10 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %rhs1 = axis.product (%f10 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>)
  %f11 = axis.factor %rep4_1 : !distributed.replication_axis<4> <4, 1>
  %lhs2 = axis.product (%f11 : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
  %f12 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %rhs2 = axis.product (%f12 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %map = axis.map %lhs0, %lhs1, %lhs2 to %rhs0, %rhs1, %rhs2 : [!axis.factor_group<64>, !axis.factor_group<4>, !axis.factor_group<4>] [!axis.factor_group<64>, !axis.factor_group<4>, !axis.factor_group<4>]

  %in = tensor.empty() : tensor<64xi8>
  %h = distributed.Collective %in : tensor<64xi8> on %mesh_in : !axis.factor_group<16> to tensor<64xi8> on %mesh_out : !axis.factor_group<16> reduces (%red0 : !axis.factor_group<16>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<i8>, %rhs_v: tensor<i8>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<i8>
    stablehlo.return %r : tensor<i8>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<64xi8>> -> tensor<64xi8>
}
