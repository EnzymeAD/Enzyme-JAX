// RUN: enzymexlamlir-opt --distributed-atomize-collectives --distributed-print-collective-plan="chain=true bandwidths=4,1" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=FAST0
// RUN: enzymexlamlir-opt --distributed-atomize-collectives --distributed-print-collective-plan="chain=true bandwidths=1,4" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=FAST1
// RUN: enzymexlamlir-opt --distributed-atomize-collectives --distributed-print-collective-plan="chain=true bandwidths=4,1 disable-peel-variants=true" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=FAST0

// Regression test for a byte/element-count mix-up in the peel (D17)
// eligibility check: legality of splitting a payload across an atom's extent
// is an element-count question (a reduce-scatter's split dimension size is
// counted in elements), but the check used to test payloadBytes directly, so
// an element width sharing a factor with the extent could make the byte
// count divide evenly when the true element count did not.
//
// Same two-atom (Reduced, Replicate) structure as
// decompose_collective_peel_bandwidth.mlir's peel_fast_axis_first (S
// elements each on axis0 and axis1, both extent 4), which under a fast/slow
// bandwidth split (bandwidths=4,1 or 1,4) makes peeling the fast atom cheaper
// than the flat order whenever peeling is legal at all. Here S = 3 elements
// of f32 (4 bytes), so payloadBytes = 12: 12 % 4 == 0 (the old, wrong check
// would have offered a peel), but the true element count 3 is not divisible
// by 4, so no atom may legally be peeled. With the fix neither atom offers a
// peel, so the DP has only the flat all-reduce x2 regardless of bandwidth
// split or of disable-peel-variants -- the third RUN line (bandwidths=4,1,
// peel variants force-disabled) matches the exact same FAST0 output as the
// first, proof the atom is excluded from candidacy outright rather than
// merely outcompeted on cost.
//
// Both all-reduces are neutral-tier (they leave the payload at 12), so their
// order does not change the total (D3): V = 2 * 12 * 3/4 = 18, latency
// 0.1 + 2*0.02 = 0.14, duration 0.14 + 18/BW, total 0.14+18/4 + 0.14+18/1 =
// 4.64 + 18.14 = 22.78 under bandwidths=4,1 (FAST0) and the same total with
// the two durations swapped under bandwidths=1,4 (FAST1).
// FAST0: chain: 2 steps, total duration 22.78
// FAST0-NEXT: step 0: all-reduce
// FAST0-NEXT: atoms: axis0.0(x4)
// FAST0-NEXT: payload: 12 -> 12
// FAST0-NEXT: latency: 0.14
// FAST0-NEXT: V: [18, 0]
// FAST0-NEXT: rho: [1, 0]
// FAST0-NEXT: duration: 4.64
// FAST0-NEXT: step 1: all-reduce
// FAST0-NEXT: atoms: axis1.0(x4)
// FAST0-NEXT: payload: 12 -> 12
// FAST0-NEXT: latency: 0.14
// FAST0-NEXT: V: [0, 18]
// FAST0-NEXT: rho: [0, 1]
// FAST0-NEXT: duration: 18.14
// FAST0-NEXT: semantics: verified
// FAST0-NEXT: input port: tensor<3xf32> (the collective's input_object operand) -> step 0
// FAST0-NEXT: result port: tensor<3xf32> (the await result, 0 uses) <- step 1
// FAST1: chain: 2 steps, total duration 22.78
// FAST1-NEXT: step 0: all-reduce
// FAST1-NEXT: atoms: axis0.0(x4)
// FAST1-NEXT: payload: 12 -> 12
// FAST1-NEXT: latency: 0.14
// FAST1-NEXT: V: [18, 0]
// FAST1-NEXT: rho: [1, 0]
// FAST1-NEXT: duration: 18.14
// FAST1-NEXT: step 1: all-reduce
// FAST1-NEXT: atoms: axis1.0(x4)
// FAST1-NEXT: payload: 12 -> 12
// FAST1-NEXT: latency: 0.14
// FAST1-NEXT: V: [0, 18]
// FAST1-NEXT: rho: [0, 1]
// FAST1-NEXT: duration: 4.64
// FAST1-NEXT: semantics: verified
// FAST1-NEXT: input port: tensor<3xf32> (the collective's input_object operand) -> step 0
// FAST1-NEXT: result port: tensor<3xf32> (the await result, 0 uses) <- step 1
module @peel_element_count_mismatch {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 4>, !distributed.physical_comm_axis<4, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 4>, !distributed.physical_comm_axis<4, 1>
  %rep4_0 = distributed.ReplicationAxis 4 : !distributed.replication_axis<4>
  %rep4_1 = distributed.ReplicationAxis 4 : !distributed.replication_axis<4>
  %ti0 = axis.getaxis tensor<3xf32> 0
  %to0 = axis.getaxis tensor<3xf32> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %f4 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %mesh_out = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>, %f4 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f5 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %f6 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %red0 = axis.product (%f5 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>, %f6 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f7 = axis.factor %ti0 : !axis.shape_axis<tensor<3xf32>, 0> <3, 1>
  %lhs0 = axis.product (%f7 : !axis.axis_factor<!axis.shape_axis<tensor<3xf32>, 0>, 3, 1>)
  %f8 = axis.factor %to0 : !axis.shape_axis<tensor<3xf32>, 0> <3, 1>
  %rhs0 = axis.product (%f8 : !axis.axis_factor<!axis.shape_axis<tensor<3xf32>, 0>, 3, 1>)
  %f9 = axis.factor %rep4_0 : !distributed.replication_axis<4> <4, 1>
  %lhs1 = axis.product (%f9 : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
  %f10 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %rhs1 = axis.product (%f10 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>)
  %f11 = axis.factor %rep4_1 : !distributed.replication_axis<4> <4, 1>
  %lhs2 = axis.product (%f11 : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
  %f12 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %rhs2 = axis.product (%f12 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %map = axis.map %lhs0, %lhs1, %lhs2 to %rhs0, %rhs1, %rhs2 : [!axis.factor_group<3>, !axis.factor_group<4>, !axis.factor_group<4>] [!axis.factor_group<3>, !axis.factor_group<4>, !axis.factor_group<4>]

  %in = tensor.empty() : tensor<3xf32>
  %h = distributed.Collective %in : tensor<3xf32> on %mesh_in : !axis.factor_group<16> to tensor<3xf32> on %mesh_out : !axis.factor_group<16> reduces (%red0 : !axis.factor_group<16>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<3xf32>> -> tensor<3xf32>
}
