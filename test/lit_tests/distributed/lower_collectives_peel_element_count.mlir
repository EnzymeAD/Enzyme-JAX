// RUN: enzymexlamlir-opt --distributed-atomize-collectives --distributed-lower-collectives %s | FileCheck %s

// End-to-end regression test for the byte/element-count mix-up in the peel
// (D17) eligibility check described in decompose_collective_peel_element_count.mlir:
// legality of splitting a payload across an atom's extent is an
// element-count question, but the check used to test payloadBytes directly.
// Reproduces a real crash: a payload too small to actually divide (1 f32
// element, 4 bytes, over an extent-4 atom: 4 % 4 == 0 by bytes, but 1 % 4 !=
// 0 by elements) used to still be offered a peel, and since the mesh's
// axis_bandwidths = [4, 1] makes peeling the fast atom look cheap, the DP
// selected it, then distributed-lower-collectives crashed trying to build
// the peel's stablehlo.reduce_scatter and stablehlo.all_gather with a
// scatter/gather dimension of size 1/4 = 0 ("result dimension size at
// scatter_dimension cannot be zero", "dimension size of operand at
// 'all_gather_dim' cannot be zero"). With the fix neither mesh atom offers a
// peel, so this lowers to the flat two-step chain below (order fixed by the
// bandwidth split: the fast atom's all-reduce runs first, matching Smith's
// rule for two neutral-tier reductions of equal volume run back to back).
// CHECK-LABEL: module @peel_element_count_mismatch {
// CHECK: %[[IN:.+]] = tensor.empty() : tensor<1xf32>
// CHECK: %[[START0:.+]] = "stablehlo.async_start"(%[[IN]]) (
// CHECK: ^bb0(%{{.+}}: tensor<1xf32>):
// CHECK:   "stablehlo.all_reduce"
// CHECK: }) {distributed.primitive_cost = {isolated_duration = {{.*}} : f64, kind = "all-reduce", latency = {{.*}} : f64, payload_in = 4 : i64, payload_out = 4 : i64
// CHECK: %[[DONE0:.+]] = "stablehlo.async_done"(%[[START0]]) : (!stablehlo.future<tensor<1xf32>>) -> tensor<1xf32>
// CHECK: %[[START1:.+]] = "stablehlo.async_start"(%[[DONE0]]) (
// CHECK: ^bb0(%{{.+}}: tensor<1xf32>):
// CHECK:   "stablehlo.all_reduce"
// CHECK: }) {distributed.primitive_cost = {isolated_duration = {{.*}} : f64, kind = "all-reduce", latency = {{.*}} : f64, payload_in = 4 : i64, payload_out = 4 : i64
// CHECK: %{{.+}} = "stablehlo.async_done"(%[[START1]]) : (!stablehlo.future<tensor<1xf32>>) -> tensor<1xf32>
// CHECK-NOT: stablehlo.reduce_scatter
// CHECK-NOT: stablehlo.all_gather
// CHECK-NOT: distributed.Collective
// CHECK-NOT: distributed.Await
module @peel_element_count_mismatch {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 4>, !distributed.physical_comm_axis<4, 1>] {axis_bandwidths = [4.0e0, 1.0e0]}

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 4>, !distributed.physical_comm_axis<4, 1>
  %rep4_0 = distributed.ReplicationAxis 4 : !distributed.replication_axis<4>
  %rep4_1 = distributed.ReplicationAxis 4 : !distributed.replication_axis<4>
  %ti0 = axis.getaxis tensor<1xf32> 0
  %to0 = axis.getaxis tensor<1xf32> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %f4 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %mesh_out = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>, %f4 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f5 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %f6 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %red0 = axis.product (%f5 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>, %f6 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f7 = axis.factor %ti0 : !axis.shape_axis<tensor<1xf32>, 0> <1, 1>
  %lhs0 = axis.product (%f7 : !axis.axis_factor<!axis.shape_axis<tensor<1xf32>, 0>, 1, 1>)
  %f8 = axis.factor %to0 : !axis.shape_axis<tensor<1xf32>, 0> <1, 1>
  %rhs0 = axis.product (%f8 : !axis.axis_factor<!axis.shape_axis<tensor<1xf32>, 0>, 1, 1>)
  %f9 = axis.factor %rep4_0 : !distributed.replication_axis<4> <4, 1>
  %lhs1 = axis.product (%f9 : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
  %f10 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %rhs1 = axis.product (%f10 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>)
  %f11 = axis.factor %rep4_1 : !distributed.replication_axis<4> <4, 1>
  %lhs2 = axis.product (%f11 : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
  %f12 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %rhs2 = axis.product (%f12 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %map = axis.map %lhs0, %lhs1, %lhs2 to %rhs0, %rhs1, %rhs2 : [!axis.factor_group<1>, !axis.factor_group<4>, !axis.factor_group<4>] [!axis.factor_group<1>, !axis.factor_group<4>, !axis.factor_group<4>]

  %in = tensor.empty() : tensor<1xf32>
  %h = distributed.Collective %in : tensor<1xf32> on %mesh_in : !axis.factor_group<16> to tensor<1xf32> on %mesh_out : !axis.factor_group<16> reduces (%red0 : !axis.factor_group<16>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<1xf32>> -> tensor<1xf32>
}
