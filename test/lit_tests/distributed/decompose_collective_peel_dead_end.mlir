// RUN: enzymexlamlir-opt --split-input-file --distributed-print-collective-plan="chain=true" %s 2>&1 >/dev/null | FileCheck %s

// A step that splits the tile is offered only where the per-device element
// count divides by its extent. In the first three cases an 8-element tile is
// reduce-scattered onto p1 (extent 8) and all-reduced over p0 (extent 2). A
// peel of p0 leaves 4 elements, which p1's reduce-scatter (or its all-reduce
// and slice) can't divide by 8, and running p1 first leaves one element the
// peel can't scatter. Every peel order dead-ends, so the search discards them
// and plans the flat chain.

// Fast p0, where a peel would be cheapest if it could finish.
// CHECK-LABEL: payload bytes: 32
// CHECK: chain: 2 steps
// CHECK-NEXT: step 0: reduce-scatter
// CHECK-NEXT: atoms: axis1.0(x8)
// CHECK: step 1: all-reduce
// CHECK-NEXT: atoms: axis0.0(x2)
// CHECK: semantics: verified
module @peel_scatter_dead_end {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 8>, !distributed.physical_comm_axis<8, 1>] {axis_bandwidths = [8.0, 0.1]}

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 8>, !distributed.physical_comm_axis<8, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<8xf32> 0
  %to0 = axis.getaxis tensor<1xf32> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 8> <2, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %f3 = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 8>, 2, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %f4 = axis.factor %p0 : !distributed.physical_comm_axis<2, 8> <2, 1>
  %f5 = axis.factor %p1 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %f6 = axis.product (%f4 : !axis.axis_factor<!distributed.physical_comm_axis<2, 8>, 2, 1>, %f5 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 8> <2, 1>
  %f8 = axis.factor %p1 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %f9 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 8>, 2, 1>, %f8 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %f10 = axis.factor %ti0 : !axis.shape_axis<tensor<8xf32>, 0> <8, 1>
  %f11 = axis.product (%f10 : !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 8, 1>)
  %f12 = axis.factor %p1 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %f13 = axis.product (%f12 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %f14 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %f15 = axis.product (%f14 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f16 = axis.factor %p0 : !distributed.physical_comm_axis<2, 8> <2, 1>
  %f17 = axis.product (%f16 : !axis.axis_factor<!distributed.physical_comm_axis<2, 8>, 2, 1>)
  %map = axis.map %f11, %f15 to %f13, %f17 : [!axis.factor_group<8>, !axis.factor_group<2>] [!axis.factor_group<8>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<8xf32>
  %h = distributed.Collective %in : tensor<8xf32> on %f3 : !axis.factor_group<16> to tensor<1xf32> on %f6 : !axis.factor_group<16> reduces (%f9 : !axis.factor_group<16>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<1xf32>> -> tensor<1xf32>
}

// -----

// Slow p0, where a dead end scored as a finished chain would undercut the flat one.
// CHECK-LABEL: payload bytes: 32
// CHECK: chain: 2 steps
// CHECK-NEXT: step 0: reduce-scatter
// CHECK-NEXT: atoms: axis1.0(x8)
// CHECK: step 1: all-reduce
// CHECK-NEXT: atoms: axis0.0(x2)
// CHECK: semantics: verified
module @peel_scatter_dead_end_flat {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 8>, !distributed.physical_comm_axis<8, 1>] {axis_bandwidths = [0.1, 8.0]}

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 8>, !distributed.physical_comm_axis<8, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<8xf32> 0
  %to0 = axis.getaxis tensor<1xf32> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 8> <2, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %f3 = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 8>, 2, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %f4 = axis.factor %p0 : !distributed.physical_comm_axis<2, 8> <2, 1>
  %f5 = axis.factor %p1 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %f6 = axis.product (%f4 : !axis.axis_factor<!distributed.physical_comm_axis<2, 8>, 2, 1>, %f5 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 8> <2, 1>
  %f8 = axis.factor %p1 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %f9 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 8>, 2, 1>, %f8 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %f10 = axis.factor %ti0 : !axis.shape_axis<tensor<8xf32>, 0> <8, 1>
  %f11 = axis.product (%f10 : !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 8, 1>)
  %f12 = axis.factor %p1 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %f13 = axis.product (%f12 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %f14 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %f15 = axis.product (%f14 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f16 = axis.factor %p0 : !distributed.physical_comm_axis<2, 8> <2, 1>
  %f17 = axis.product (%f16 : !axis.axis_factor<!distributed.physical_comm_axis<2, 8>, 2, 1>)
  %map = axis.map %f11, %f15 to %f13, %f17 : [!axis.factor_group<8>, !axis.factor_group<2>] [!axis.factor_group<8>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<8xf32>
  %h = distributed.Collective %in : tensor<8xf32> on %f3 : !axis.factor_group<16> to tensor<1xf32> on %f6 : !axis.factor_group<16> reduces (%f9 : !axis.factor_group<16>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<1xf32>> -> tensor<1xf32>
}

// -----

// One-byte elements, where p1's reduce-scatter after a peel would run on 4 bytes.
// CHECK-LABEL: payload bytes: 8
// CHECK: chain: 2 steps
// CHECK-NEXT: step 0: reduce-scatter
// CHECK-NEXT: atoms: axis1.0(x8)
// CHECK: step 1: all-reduce
// CHECK-NEXT: atoms: axis0.0(x2)
// CHECK: semantics: verified
module @peel_scatter_dead_end_bytes {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 8>, !distributed.physical_comm_axis<8, 1>] {axis_bandwidths = [8.0, 0.1]}

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 8>, !distributed.physical_comm_axis<8, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<8xi8> 0
  %to0 = axis.getaxis tensor<1xi8> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 8> <2, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %f3 = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 8>, 2, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %f4 = axis.factor %p0 : !distributed.physical_comm_axis<2, 8> <2, 1>
  %f5 = axis.factor %p1 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %f6 = axis.product (%f4 : !axis.axis_factor<!distributed.physical_comm_axis<2, 8>, 2, 1>, %f5 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 8> <2, 1>
  %f8 = axis.factor %p1 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %f9 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 8>, 2, 1>, %f8 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %f10 = axis.factor %ti0 : !axis.shape_axis<tensor<8xi8>, 0> <8, 1>
  %f11 = axis.product (%f10 : !axis.axis_factor<!axis.shape_axis<tensor<8xi8>, 0>, 8, 1>)
  %f12 = axis.factor %p1 : !distributed.physical_comm_axis<8, 1> <8, 1>
  %f13 = axis.product (%f12 : !axis.axis_factor<!distributed.physical_comm_axis<8, 1>, 8, 1>)
  %f14 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %f15 = axis.product (%f14 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f16 = axis.factor %p0 : !distributed.physical_comm_axis<2, 8> <2, 1>
  %f17 = axis.product (%f16 : !axis.axis_factor<!distributed.physical_comm_axis<2, 8>, 2, 1>)
  %map = axis.map %f11, %f15 to %f13, %f17 : [!axis.factor_group<8>, !axis.factor_group<2>] [!axis.factor_group<8>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<8xi8>
  %h = distributed.Collective %in : tensor<8xi8> on %f3 : !axis.factor_group<16> to tensor<1xi8> on %f6 : !axis.factor_group<16> reduces (%f9 : !axis.factor_group<16>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<i8>, %rhs_v: tensor<i8>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<i8>
    stablehlo.return %r : tensor<i8>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<1xi8>> -> tensor<1xi8>
}

// -----

// An all-to-all on p1 is not offered on the one element a peel of p0 leaves.
// CHECK-LABEL: payload bytes: 8
// CHECK: chain: 2 steps
// CHECK-NEXT: step 0: all-reduce
// CHECK-NEXT: atoms: axis0.0(x2)
// CHECK: step 1: all-to-all
// CHECK-NEXT: atoms: axis1.0(x2)
// CHECK: semantics: verified
module @peel_all_to_all_needs_elements {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>] {axis_bandwidths = [8.0, 0.1]}

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<2xf32> 0
  %to0 = axis.getaxis tensor<2xf32> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f3 = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f4 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f5 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f6 = axis.product (%f4 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f5 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f8 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f9 = axis.factor %ti0 : !axis.shape_axis<tensor<2xf32>, 0> <2, 1>
  %f10 = axis.product (%f9 : !axis.axis_factor<!axis.shape_axis<tensor<2xf32>, 0>, 2, 1>)
  %f11 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f12 = axis.product (%f11 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f13 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %f14 = axis.product (%f13 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f15 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f16 = axis.product (%f15 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f17 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f18 = axis.product (%f17 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f19 = axis.factor %to0 : !axis.shape_axis<tensor<2xf32>, 0> <2, 1>
  %f20 = axis.product (%f19 : !axis.axis_factor<!axis.shape_axis<tensor<2xf32>, 0>, 2, 1>)
  %map = axis.map %f10, %f14, %f18 to %f12, %f16, %f20 : [!axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<2xf32>
  %h = distributed.Collective %in : tensor<2xf32> on %f3 : !axis.factor_group<4> to tensor<2xf32> on %f6 : !axis.factor_group<4> reduces (%f8 : !axis.factor_group<2>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<2xf32>> -> tensor<2xf32>
}
