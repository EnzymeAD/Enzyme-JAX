// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true bandwidths=8,0.1,0.1" %s 2>&1 >/dev/null | FileCheck %s
// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true bandwidths=8,0.1,0.1 disable-peel-variants=true" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=FLAT

// Hierarchical multi-axis all-reduce: peeling axis0 lets axis1 and axis2's
// own reduce-scatters run on the payload axis0's self-scatter already
// shrank (see the "hierarchical multi-axis all-reduce" remark in the
// Decomposition section). S = 2048; axis0 (extent 4, BW 8 bytes/ns) is
// (Reduced, Replicate); axis1 and axis2 (extent 2 each, BW 0.1) are
// (Reduced, Tile), each scattering onto its own tile atom.
//
// Flat reduce-scatters axis1 then axis2 at full (then three-quarter) payload
// before axis0's all-reduce:
//   reduce-scatter axis1 (BW 0.1) at S = 2048: V = 2048/2 = 1024,
//     duration = 6000 + 1024/0.1 = 16240, payload 2048 -> 1024
//   reduce-scatter axis2 (BW 0.1) at 1024: V = 1024/2 = 512,
//     duration = 6000 + 512/0.1 = 11120, payload 1024 -> 512
//   all-reduce axis0 (BW 8) at 512: recursive doubling wins
//     (V = 512*2 = 1024, duration = 7000 + 1024/8 = 7128)
//   Flat total = 16240 + 11120 + 7128 = 34488
// Peeling axis0 self-scatters the full payload first, over axis0's fast
// port, so axis1 and axis2's own reduce-scatters move a quarter as many
// bytes at their slow bandwidth:
//   self-scatter axis0 (BW 8) at S = 2048: V = 2048*3/4 = 1536,
//     duration = 7000 + 1536/8 = 7192, payload 2048 -> 512
//   reduce-scatter axis1 (BW 0.1) at 512: V = 512/2 = 256,
//     duration = 6000 + 256/0.1 = 8560, payload 512 -> 256
//   reduce-scatter axis2 (BW 0.1) at 256: V = 256/2 = 128,
//     duration = 6000 + 128/0.1 = 7280, payload 256 -> 128
//   closing gather axis0 (BW 8): V = 128*3 = 384,
//     duration = 7000 + 384/8 = 7048, payload 128 -> 512
//   Peel total = 7192 + 8560 + 7280 + 7048 = 30080
// Peel wins by 4408ns: the two slow reduce-scatters each move only a
// quarter of the bytes they would flat, at their own 0.1 bytes/ns port,
// which is worth far more than the one extra launch latency (and the fast
// axis0 port's own transfer time) peeling costs.
// CHECK: chain: 4 steps, total duration 30080
// CHECK-NEXT: step 0: reduce-scatter
// CHECK-NEXT: atoms: axis0.0(x4)
// CHECK-NEXT: payload: 2048 -> 512
// CHECK-NEXT: latency: 7000
// CHECK-NEXT: V: [1536, 0, 0]
// CHECK-NEXT: rho: [1, 0, 0]
// CHECK-NEXT: duration: 7192
// CHECK-NEXT: step 1: reduce-scatter
// CHECK-NEXT: atoms: axis1.0(x2)
// CHECK-NEXT: payload: 512 -> 256
// CHECK-NEXT: latency: 6000
// CHECK-NEXT: V: [0, 256, 0]
// CHECK-NEXT: rho: [0, 1, 0]
// CHECK-NEXT: duration: 8560
// CHECK-NEXT: step 2: reduce-scatter
// CHECK-NEXT: atoms: axis2.0(x2)
// CHECK-NEXT: payload: 256 -> 128
// CHECK-NEXT: latency: 6000
// CHECK-NEXT: V: [0, 0, 128]
// CHECK-NEXT: rho: [0, 0, 1]
// CHECK-NEXT: duration: 7280
// CHECK-NEXT: step 3: all-gather
// CHECK-NEXT: atoms: axis0.0(x4)
// CHECK-NEXT: payload: 128 -> 512
// CHECK-NEXT: latency: 7000
// CHECK-NEXT: V: [384, 0, 0]
// CHECK-NEXT: rho: [1, 0, 0]
// CHECK-NEXT: duration: 7048
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<2048xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<512xi8> (the await result, 0 uses) <- step 3
// FLAT: chain: 3 steps, total duration 34488
// FLAT-NEXT: step 0: reduce-scatter
// FLAT-NEXT: atoms: axis1.0(x2)
// FLAT-NEXT: payload: 2048 -> 1024
// FLAT-NEXT: latency: 6000
// FLAT-NEXT: V: [0, 1024, 0]
// FLAT-NEXT: rho: [0, 1, 0]
// FLAT-NEXT: duration: 16240
// FLAT-NEXT: step 1: reduce-scatter
// FLAT-NEXT: atoms: axis2.0(x2)
// FLAT-NEXT: payload: 1024 -> 512
// FLAT-NEXT: latency: 6000
// FLAT-NEXT: V: [0, 0, 512]
// FLAT-NEXT: rho: [0, 0, 1]
// FLAT-NEXT: duration: 11120
// FLAT-NEXT: step 2: all-reduce
// FLAT-NEXT: atoms: axis0.0(x4)
// FLAT-NEXT: payload: 512 -> 512
// FLAT-NEXT: latency: 7000
// FLAT-NEXT: V: [1024, 0, 0]
// FLAT-NEXT: rho: [1, 0, 0]
// FLAT-NEXT: duration: 7128
// FLAT-NEXT: semantics: verified
// FLAT-NEXT: input port: tensor<2048xi8> (the collective's input_object operand) -> step 0
// FLAT-NEXT: result port: tensor<512xi8> (the await result, 0 uses) <- step 2
module @peel_hierarchical {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 4>, !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0, %p1, %p2 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 4>, !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %rep4 = distributed.ReplicationAxis 4 : !distributed.replication_axis<4>
  %ti0 = axis.getaxis tensor<2048xi8> 0
  %to0 = axis.getaxis tensor<512xi8> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f3 = axis.factor %p2 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f4 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %f5 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f6 = axis.factor %p2 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f4 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>, %f5 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f6 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %f8 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f9 = axis.factor %p2 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %red0 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>, %f8 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f9 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f10 = axis.factor %ti0 : !axis.shape_axis<tensor<2048xi8>, 0> <2048, 1>
  %lhs0 = axis.product (%f10 : !axis.axis_factor<!axis.shape_axis<tensor<2048xi8>, 0>, 2048, 1>)
  %f11 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f12 = axis.factor %p2 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f13 = axis.factor %to0 : !axis.shape_axis<tensor<512xi8>, 0> <512, 1>
  %rhs0 = axis.product (%f11 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f12 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %f13 : !axis.axis_factor<!axis.shape_axis<tensor<512xi8>, 0>, 512, 1>)
  %f14 = axis.factor %rep4 : !distributed.replication_axis<4> <4, 1>
  %lhs1 = axis.product (%f14 : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
  %f15 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %rhs1 = axis.product (%f15 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<2048>, !axis.factor_group<4>] [!axis.factor_group<2048>, !axis.factor_group<4>]

  %in = tensor.empty() : tensor<2048xi8>
  %h = distributed.Collective %in : tensor<2048xi8> on %mesh_in : !axis.factor_group<16> to tensor<512xi8> on %mesh_out : !axis.factor_group<16> reduces (%red0 : !axis.factor_group<16>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<i8>, %rhs_v: tensor<i8>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<i8>
    stablehlo.return %r : tensor<i8>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<512xi8>> -> tensor<512xi8>
}
