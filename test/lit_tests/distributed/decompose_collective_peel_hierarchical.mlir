// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true bandwidths=8,1,1" %s 2>&1 >/dev/null | FileCheck %s
// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan="chain=true bandwidths=8,1,1 disable-peel-variants=true" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=FLAT

// Hierarchical multi-axis all-reduce: peeling axis0 lets the DP interleave
// two other atoms' full units of work in the gap (see the "hierarchical
// multi-axis all-reduce" remark in the Decomposition section, which is
// exactly this ordering). S = 64; axis0 (extent 4, BW 8) is (Reduced,
// Replicate); axis1 and axis2 (extent 2 each, BW 1) are (Reduced, Tile),
// each scattering onto its own tile atom.
//   peel:  self-scatter axis0 at 64: V = 64*3/4 = 48, BW 8, latency 0.12, duration 0.12+48/8 = 6.12; payload 64 -> 16
//          reduce-scatter axis1 at 16: V = 16*1/2 = 8, BW 1, latency 0.11, duration 8.11; payload 16 -> 8
//          reduce-scatter axis2 at 8: V = 8*1/2 = 4, BW 1, latency 0.11, duration 4.11; payload 8 -> 4
//          closing all-gather axis0 at 4: V = 4*3 = 12, BW 8, latency 0.12, duration 0.12+12/8 = 1.62
//          total 19.96
//   flat:  reduce-scatter axis1 at 64: V = 32, BW 1, latency 0.11, duration 32.11; payload 64 -> 32
//          reduce-scatter axis2 at 32: V = 16, BW 1, latency 0.11, duration 16.11; payload 32 -> 16
//          all-reduce axis0 at 16: halving+doubling V = 2*16*3/4 = 24, BW 8, latency 0.14, duration 0.14+24/8 = 3.14
//          total 51.36 (FLAT, disable-peel-variants=true)
// The peel pays axis0's own cost on the fast axis while axis1 and axis2 both
// run at a quarter of the payload they would otherwise see, more than
// covering the extra launch from splitting axis0 into two steps.
// CHECK: chain: 4 steps, total duration 19.96
// CHECK-NEXT: step 0: reduce-scatter
// CHECK-NEXT: atoms: axis0.0(x4)
// CHECK-NEXT: payload: 64 -> 16
// CHECK-NEXT: latency: 0.12
// CHECK-NEXT: V: [48, 0, 0]
// CHECK-NEXT: rho: [1, 0, 0]
// CHECK-NEXT: duration: 6.12
// CHECK-NEXT: step 1: reduce-scatter
// CHECK-NEXT: atoms: axis1.0(x2)
// CHECK-NEXT: payload: 16 -> 8
// CHECK-NEXT: latency: 0.11
// CHECK-NEXT: V: [0, 8, 0]
// CHECK-NEXT: rho: [0, 1, 0]
// CHECK-NEXT: duration: 8.11
// CHECK-NEXT: step 2: reduce-scatter
// CHECK-NEXT: atoms: axis2.0(x2)
// CHECK-NEXT: payload: 8 -> 4
// CHECK-NEXT: latency: 0.11
// CHECK-NEXT: V: [0, 0, 4]
// CHECK-NEXT: rho: [0, 0, 1]
// CHECK-NEXT: duration: 4.11
// CHECK-NEXT: step 3: all-gather
// CHECK-NEXT: atoms: axis0.0(x4)
// CHECK-NEXT: payload: 4 -> 16
// CHECK-NEXT: latency: 0.12
// CHECK-NEXT: V: [12, 0, 0]
// CHECK-NEXT: rho: [1, 0, 0]
// CHECK-NEXT: duration: 1.62
// CHECK-NEXT: semantics: verified
// CHECK-NEXT: input port: tensor<64xi8> (the collective's input_object operand) -> step 0
// CHECK-NEXT: result port: tensor<16xi8> (the await result, 0 uses) <- step 3
// FLAT: chain: 3 steps, total duration 51.36
// FLAT-NEXT: step 0: reduce-scatter
// FLAT-NEXT: atoms: axis1.0(x2)
// FLAT-NEXT: payload: 64 -> 32
// FLAT-NEXT: latency: 0.11
// FLAT-NEXT: V: [0, 32, 0]
// FLAT-NEXT: rho: [0, 1, 0]
// FLAT-NEXT: duration: 32.11
// FLAT-NEXT: step 1: reduce-scatter
// FLAT-NEXT: atoms: axis2.0(x2)
// FLAT-NEXT: payload: 32 -> 16
// FLAT-NEXT: latency: 0.11
// FLAT-NEXT: V: [0, 0, 16]
// FLAT-NEXT: rho: [0, 0, 1]
// FLAT-NEXT: duration: 16.11
// FLAT-NEXT: step 2: all-reduce
// FLAT-NEXT: atoms: axis0.0(x4)
// FLAT-NEXT: payload: 16 -> 16
// FLAT-NEXT: latency: 0.14
// FLAT-NEXT: V: [24, 0, 0]
// FLAT-NEXT: rho: [1, 0, 0]
// FLAT-NEXT: duration: 3.14
// FLAT-NEXT: semantics: verified
// FLAT-NEXT: input port: tensor<64xi8> (the collective's input_object operand) -> step 0
// FLAT-NEXT: result port: tensor<16xi8> (the await result, 0 uses) <- step 2
module @peel_hierarchical {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 4>, !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0, %p1, %p2 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 4>, !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %rep4 = distributed.ReplicationAxis 4 : !distributed.replication_axis<4>
  %ti0 = axis.getaxis tensor<64xi8> 0
  %to0 = axis.getaxis tensor<16xi8> 0
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
  %f10 = axis.factor %ti0 : !axis.shape_axis<tensor<64xi8>, 0> <64, 1>
  %lhs0 = axis.product (%f10 : !axis.axis_factor<!axis.shape_axis<tensor<64xi8>, 0>, 64, 1>)
  %f11 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f12 = axis.factor %p2 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f13 = axis.factor %to0 : !axis.shape_axis<tensor<16xi8>, 0> <16, 1>
  %rhs0 = axis.product (%f11 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f12 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %f13 : !axis.axis_factor<!axis.shape_axis<tensor<16xi8>, 0>, 16, 1>)
  %f14 = axis.factor %rep4 : !distributed.replication_axis<4> <4, 1>
  %lhs1 = axis.product (%f14 : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
  %f15 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %rhs1 = axis.product (%f15 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<64>, !axis.factor_group<4>] [!axis.factor_group<64>, !axis.factor_group<4>]

  %in = tensor.empty() : tensor<64xi8>
  %h = distributed.Collective %in : tensor<64xi8> on %mesh_in : !axis.factor_group<16> to tensor<16xi8> on %mesh_out : !axis.factor_group<16> reduces (%red0 : !axis.factor_group<16>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<i8>, %rhs_v: tensor<i8>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<i8>
    stablehlo.return %r : tensor<i8>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<16xi8>> -> tensor<16xi8>
}
