// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true bandwidths=4,1" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=FAST0
// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true bandwidths=1,4" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=FAST1
// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true bandwidths=4,1 disable-peel-variants=true" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=FLAT

// Peel (D17) winning under non-uniform bandwidth. Round latency 1000ns and
// launch latency 5000ns as elsewhere; BW (bytes/ns) as given by bandwidths=.
// duration = latency + round(V / BW) (CollectiveCost.h's Units note).
//
// Two independent reduced atoms (S = 20480), the same shape of collective as
// decompose_collective_peel.mlir's uniform-bandwidth case: axis0 (extent 4)
// is (Reduced, Replicate), axis1 (extent 4) is (Reduced, Tile). Flat picks
// axis1's reduce-scatter (shrinking the shared payload 20480 -> 5120) before
// axis0's all-reduce on that shrunk payload:
//   reduce-scatter axis1 (BW 1): V = 20480*3/4 = 15360, k = 2 rounds,
//     duration = 7000 + 15360 = 22360
//   all-reduce axis0 (BW 4) at payload 5120: recursive doubling wins
//     (V = 5120*2 = 10240, duration = 7000 + 10240/4 = 9560) over
//     halving+doubling (V = 2*5120*3/4 = 7680, 4 rounds, 9000 + 7680/4 = 10920)
//   Flat total = 22360 + 9560 = 31920
// Peeling axis0 instead pays one extra launch latency for a self-scatter and
// a closing gather, both over axis0's fast port (BW 4), but in exchange
// axis1's own reduce-scatter runs against the payload axis0's self-scatter
// already shrank (5120 instead of 20480), at axis1's slow BW 1:
//   self-scatter axis0 (BW 4) at S = 20480: V = 20480*3/4 = 15360,
//     duration = 7000 + 15360/4 = 10840, payload 20480 -> 5120
//   reduce-scatter axis1 (BW 1) at the shrunk payload 5120: V = 5120*3/4 = 3840,
//     duration = 7000 + 3840 = 10840, payload 5120 -> 1280
//   closing gather axis0 (BW 4): V = 1280*3 = 3840,
//     duration = 7000 + 3840/4 = 7960, payload 1280 -> 5120
//   Peel total = 10840 + 10840 + 7960 = 29640
// Peel wins by 2280ns: the discount axis1's reduce-scatter gets from moving
// only a quarter of the bytes at its own slow bandwidth outweighs the extra
// launch latency peeling axis0 costs at axis0's fast bandwidth.
// FAST0: chain: 3 steps, total duration 29640
// FAST0-NEXT: step 0: reduce-scatter
// FAST0-NEXT: atoms: axis0.0(x4)
// FAST0-NEXT: payload: 20480 -> 5120
// FAST0-NEXT: latency: 7000
// FAST0-NEXT: V: [15360, 0]
// FAST0-NEXT: rho: [1, 0]
// FAST0-NEXT: duration: 10840
// FAST0-NEXT: step 1: reduce-scatter
// FAST0-NEXT: atoms: axis1.0(x4)
// FAST0-NEXT: payload: 5120 -> 1280
// FAST0-NEXT: latency: 7000
// FAST0-NEXT: V: [0, 3840]
// FAST0-NEXT: rho: [0, 1]
// FAST0-NEXT: duration: 10840
// FAST0-NEXT: step 2: all-gather
// FAST0-NEXT: atoms: axis0.0(x4)
// FAST0-NEXT: payload: 1280 -> 5120
// FAST0-NEXT: latency: 7000
// FAST0-NEXT: V: [3840, 0]
// FAST0-NEXT: rho: [1, 0]
// FAST0-NEXT: duration: 7960
// FAST0-NEXT: semantics: verified
// FAST0-NEXT: input port: tensor<20480xi8> (the collective's input_object operand) -> step 0
// FAST0-NEXT: result port: tensor<5120xi8> (the await result, 0 uses) <- step 2
// FLAT: chain: 2 steps, total duration 31920
// FLAT-NEXT: step 0: reduce-scatter
// FLAT-NEXT: atoms: axis1.0(x4)
// FLAT-NEXT: payload: 20480 -> 5120
// FLAT-NEXT: latency: 7000
// FLAT-NEXT: V: [0, 15360]
// FLAT-NEXT: rho: [0, 1]
// FLAT-NEXT: duration: 22360
// FLAT-NEXT: step 1: all-reduce
// FLAT-NEXT: atoms: axis0.0(x4)
// FLAT-NEXT: payload: 5120 -> 5120
// FLAT-NEXT: latency: 7000
// FLAT-NEXT: V: [10240, 0]
// FLAT-NEXT: rho: [1, 0]
// FLAT-NEXT: duration: 9560
// FLAT-NEXT: semantics: verified
// FLAT-NEXT: input port: tensor<20480xi8> (the collective's input_object operand) -> step 0
// FLAT-NEXT: result port: tensor<5120xi8> (the await result, 0 uses) <- step 1
module @peel_break_even {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 4>, !distributed.physical_comm_axis<4, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 4>, !distributed.physical_comm_axis<4, 1>
  %rep4 = distributed.ReplicationAxis 4 : !distributed.replication_axis<4>
  %ti0 = axis.getaxis tensor<20480xi8> 0
  %to0 = axis.getaxis tensor<5120xi8> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %f4 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %mesh_out = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>, %f4 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f5 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %f6 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %red0 = axis.product (%f5 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>, %f6 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f7 = axis.factor %ti0 : !axis.shape_axis<tensor<20480xi8>, 0> <20480, 1>
  %lhs0 = axis.product (%f7 : !axis.axis_factor<!axis.shape_axis<tensor<20480xi8>, 0>, 20480, 1>)
  %f8 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %f9 = axis.factor %to0 : !axis.shape_axis<tensor<5120xi8>, 0> <5120, 1>
  %rhs0 = axis.product (%f8 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>, %f9 : !axis.axis_factor<!axis.shape_axis<tensor<5120xi8>, 0>, 5120, 1>)
  %f10 = axis.factor %rep4 : !distributed.replication_axis<4> <4, 1>
  %lhs1 = axis.product (%f10 : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
  %f11 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %rhs1 = axis.product (%f11 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<20480>, !axis.factor_group<4>] [!axis.factor_group<20480>, !axis.factor_group<4>]

  %in = tensor.empty() : tensor<20480xi8>
  %h = distributed.Collective %in : tensor<20480xi8> on %mesh_in : !axis.factor_group<16> to tensor<5120xi8> on %mesh_out : !axis.factor_group<16> reduces (%red0 : !axis.factor_group<16>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<i8>, %rhs_v: tensor<i8>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<i8>
    stablehlo.return %r : tensor<i8>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<5120xi8>> -> tensor<5120xi8>
}

// -----

// Fast axis first: which atom would peel follows bandwidth, not atom order.
// Both axis0 and axis1 (extent 4 each) are (Reduced, Replicate) on the same
// collective (S = 8192). Flat runs two independent all-reduces at full
// payload, and their sum does not depend on order:
//   all-reduce axis0 (BW 4): recursive doubling, V = 8192*2 = 16384,
//     duration = 7000 + 16384/4 = 11096
//   all-reduce axis1 (BW 1): at this payload halving+doubling wins
//     (V = 2*8192*3/4 = 12288, 4 rounds, duration = 9000 + 12288 = 21288)
//     over recursive doubling (V = 16384, duration = 7000 + 16384 = 23384)
//   Flat total = 11096 + 21288 = 32384
// Peeling the fast axis (self-scatter + gather over BW 4) lets the slow
// axis's own all-reduce run on the shrunk quarter payload instead of the
// full one:
//   self-scatter fast axis (BW 4) at S = 8192: V = 8192*3/4 = 6144,
//     duration = 7000 + 6144/4 = 8536, payload 8192 -> 2048
//   all-reduce slow axis (BW 1) at 2048: recursive doubling wins
//     (V = 2048*2 = 4096, duration = 7000 + 4096 = 11096) over
//     halving+doubling (V = 2*2048*3/4 = 3072, 9000 + 3072 = 12072)
//   closing gather fast axis (BW 4): V = 2048*3 = 6144,
//     duration = 7000 + 6144/4 = 8536, payload 2048 -> 8192
//   Peel total = 8536 + 11096 + 8536 = 28168
// Peel wins by 4216ns either way the bandwidths are assigned (FAST0 peels
// axis0, FAST1 peels axis1), so both give the same total.
// FAST0: chain: 3 steps, total duration 28168
// FAST0-NEXT: step 0: reduce-scatter
// FAST0-NEXT: atoms: axis0.0(x4)
// FAST0-NEXT: payload: 8192 -> 2048
// FAST0-NEXT: latency: 7000
// FAST0-NEXT: V: [6144, 0]
// FAST0-NEXT: rho: [1, 0]
// FAST0-NEXT: duration: 8536
// FAST0-NEXT: step 1: all-reduce
// FAST0-NEXT: atoms: axis1.0(x4)
// FAST0-NEXT: payload: 2048 -> 2048
// FAST0-NEXT: latency: 7000
// FAST0-NEXT: V: [0, 4096]
// FAST0-NEXT: rho: [0, 1]
// FAST0-NEXT: duration: 11096
// FAST0-NEXT: step 2: all-gather
// FAST0-NEXT: atoms: axis0.0(x4)
// FAST0-NEXT: payload: 2048 -> 8192
// FAST0-NEXT: latency: 7000
// FAST0-NEXT: V: [6144, 0]
// FAST0-NEXT: rho: [1, 0]
// FAST0-NEXT: duration: 8536
// FAST0-NEXT: semantics: verified
// FAST0-NEXT: input port: tensor<8192xi8> (the collective's input_object operand) -> step 0
// FAST0-NEXT: result port: tensor<8192xi8> (the await result, 0 uses) <- step 2
// FAST1: chain: 3 steps, total duration 28168
// FAST1-NEXT: step 0: reduce-scatter
// FAST1-NEXT: atoms: axis1.0(x4)
// FAST1-NEXT: payload: 8192 -> 2048
// FAST1-NEXT: latency: 7000
// FAST1-NEXT: V: [0, 6144]
// FAST1-NEXT: rho: [0, 1]
// FAST1-NEXT: duration: 8536
// FAST1-NEXT: step 1: all-reduce
// FAST1-NEXT: atoms: axis0.0(x4)
// FAST1-NEXT: payload: 2048 -> 2048
// FAST1-NEXT: latency: 7000
// FAST1-NEXT: V: [4096, 0]
// FAST1-NEXT: rho: [1, 0]
// FAST1-NEXT: duration: 11096
// FAST1-NEXT: step 2: all-gather
// FAST1-NEXT: atoms: axis1.0(x4)
// FAST1-NEXT: payload: 2048 -> 8192
// FAST1-NEXT: latency: 7000
// FAST1-NEXT: V: [0, 6144]
// FAST1-NEXT: rho: [0, 1]
// FAST1-NEXT: duration: 8536
// FAST1-NEXT: semantics: verified
// FAST1-NEXT: input port: tensor<8192xi8> (the collective's input_object operand) -> step 0
// FAST1-NEXT: result port: tensor<8192xi8> (the await result, 0 uses) <- step 2
// FLAT: chain: 2 steps, total duration 32384
// FLAT-NEXT: step 0: all-reduce
// FLAT-NEXT: atoms: axis0.0(x4)
// FLAT-NEXT: payload: 8192 -> 8192
// FLAT-NEXT: latency: 7000
// FLAT-NEXT: V: [16384, 0]
// FLAT-NEXT: rho: [1, 0]
// FLAT-NEXT: duration: 11096
// FLAT-NEXT: step 1: all-reduce
// FLAT-NEXT: atoms: axis1.0(x4)
// FLAT-NEXT: payload: 8192 -> 8192
// FLAT-NEXT: latency: 9000
// FLAT-NEXT: V: [0, 12288]
// FLAT-NEXT: rho: [0, 1]
// FLAT-NEXT: duration: 21288
// FLAT-NEXT: semantics: verified
// FLAT-NEXT: input port: tensor<8192xi8> (the collective's input_object operand) -> step 0
// FLAT-NEXT: result port: tensor<8192xi8> (the await result, 0 uses) <- step 1
module @peel_fast_axis_first {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 4>, !distributed.physical_comm_axis<4, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 4>, !distributed.physical_comm_axis<4, 1>
  %rep4_0 = distributed.ReplicationAxis 4 : !distributed.replication_axis<4>
  %rep4_1 = distributed.ReplicationAxis 4 : !distributed.replication_axis<4>
  %ti0 = axis.getaxis tensor<8192xi8> 0
  %to0 = axis.getaxis tensor<8192xi8> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %f4 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %mesh_out = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>, %f4 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f5 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %f6 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %red0 = axis.product (%f5 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>, %f6 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f7 = axis.factor %ti0 : !axis.shape_axis<tensor<8192xi8>, 0> <8192, 1>
  %lhs0 = axis.product (%f7 : !axis.axis_factor<!axis.shape_axis<tensor<8192xi8>, 0>, 8192, 1>)
  %f8 = axis.factor %to0 : !axis.shape_axis<tensor<8192xi8>, 0> <8192, 1>
  %rhs0 = axis.product (%f8 : !axis.axis_factor<!axis.shape_axis<tensor<8192xi8>, 0>, 8192, 1>)
  %f9 = axis.factor %rep4_0 : !distributed.replication_axis<4> <4, 1>
  %lhs1 = axis.product (%f9 : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
  %f10 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %rhs1 = axis.product (%f10 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>)
  %f11 = axis.factor %rep4_1 : !distributed.replication_axis<4> <4, 1>
  %lhs2 = axis.product (%f11 : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
  %f12 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %rhs2 = axis.product (%f12 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %map = axis.map %lhs0, %lhs1, %lhs2 to %rhs0, %rhs1, %rhs2 : [!axis.factor_group<8192>, !axis.factor_group<4>, !axis.factor_group<4>] [!axis.factor_group<8192>, !axis.factor_group<4>, !axis.factor_group<4>]

  %in = tensor.empty() : tensor<8192xi8>
  %h = distributed.Collective %in : tensor<8192xi8> on %mesh_in : !axis.factor_group<16> to tensor<8192xi8> on %mesh_out : !axis.factor_group<16> reduces (%red0 : !axis.factor_group<16>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<i8>, %rhs_v: tensor<i8>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<i8>
    stablehlo.return %r : tensor<i8>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<8192xi8>> -> tensor<8192xi8>
}
