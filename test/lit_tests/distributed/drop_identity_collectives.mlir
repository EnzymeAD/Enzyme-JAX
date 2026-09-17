// RUN: enzymexlamlir-opt %s --drop-identity-collectives | FileCheck %s

// A collective with no reduction, structurally-identical input/output
// meshes, and a fully identity mapping performs no real communication:
// its distributed.Await use is replaced directly by input_object, and both
// the collective and the await are erased. Neither side's mapping group
// bothers representing the (extent-1) output tensor's own shape axis:
// areFactorIndexSpacesEqual ignores extent-1 factors on either side, so
// DistributedCollectiveOp::verify() accepts this without it -- see
// test/lit_tests/axis/canonicalize.mlir for axis.product's own unit-factor
// dropping, which is what makes such factors unnecessary to write out here.
// CHECK-LABEL: module @drops_identity_collective
// CHECK: %[[LOCAL:.*]] = distributed.CastGlobalToLocal
// CHECK-NOT: distributed.Collective
// CHECK-NOT: distributed.Await
// CHECK: distributed.DistributedKernel (%[[LOCAL]] : tensor<1xf32>)
module @drops_identity_collective {
  func.func @main() {
    return
  }
  %l = distributed.LogicalMeshAxes 2 : !distributed.logical_mesh_axis<2>
  %rf = axis.factor %l : !distributed.logical_mesh_axis<2><2, 1>

  %global = tensor.empty() : tensor<2xf32>
  %cast_axes = axis.product (%rf : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>)
  %local = distributed.CastGlobalToLocal %global axes (%cast_axes : !axis.factor_group<2>) : tensor<2xf32> -> tensor<1xf32>

  %mesh_in = axis.product (%rf : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>)
  %mesh_out = axis.product (%rf : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>)

  %lhs_b = axis.product (%rf : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>)
  %rhs_b = axis.product (%rf : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>)
  %map = axis.map %lhs_b to %rhs_b : [!axis.factor_group<2>] [!axis.factor_group<2>]

  %h = distributed.Collective %local : tensor<1xf32> on %mesh_in : !axis.factor_group<2> to tensor<1xf32> on %mesh_out : !axis.factor_group<2> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<1xf32>> -> tensor<1xf32>
  %keep = distributed.DistributedKernel (%v : tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    -> (tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    axes () {
  ^bb0(%arg0: tensor<1xf32>):
    distributed.DistributedYield (%arg0 : tensor<1xf32>)
  }
}

// A collective with a real reduction performs genuine communication and
// must be left untouched.
// CHECK-LABEL: module @keeps_collective_with_reduction
// CHECK: distributed.Collective
// CHECK: distributed.Await
module @keeps_collective_with_reduction {
  func.func @main() {
    return
  }
  %l0 = distributed.LogicalMeshAxes 2 : !distributed.logical_mesh_axis<2>
  %l1 = distributed.LogicalMeshAxes 2 : !distributed.logical_mesh_axis<2>
  %lf0 = axis.factor %l0 : !distributed.logical_mesh_axis<2> <2, 1>
  %lf1 = axis.factor %l1 : !distributed.logical_mesh_axis<2> <2, 1>
  %ta = axis.getaxis tensor<8xf32> 0
  %ta_out = axis.getaxis tensor<4xf32> 0
  %tf_out = axis.factor %ta_out : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %tf_to_mesh = axis.factor %ta : !axis.shape_axis<tensor<8xf32>, 0> <4, 2>
  %tf_remain = axis.factor %ta : !axis.shape_axis<tensor<8xf32>, 0> <2, 1>
  %mesh_in = axis.product (%lf0 : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>, %lf1 : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>)
  %mesh_out = axis.product (%lf0 : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>, %lf1 : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>)
  %reduction = axis.product (%lf1 : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>)
  %lhs_group_1 = axis.product (%tf_to_mesh : !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 4, 2>)
  %lhs_group_2 = axis.product (%lf0 : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>, %tf_remain : !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 2, 1>)
  %rhs_group_2 = axis.product (%tf_out : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %mapping = axis.map %lhs_group_1, %lhs_group_2 to %mesh_out, %rhs_group_2 : [!axis.factor_group<4>, !axis.factor_group<4>] [!axis.factor_group<4>, !axis.factor_group<4>]
  %input = tensor.empty() : tensor<8xf32>

  %h = distributed.Collective %input : tensor<8xf32> on %mesh_in : !axis.factor_group<4> to tensor<4xf32> on %mesh_out : !axis.factor_group<4> reduces (%reduction : !axis.factor_group<2>) maps %mapping : !axis.map {
  ^bb0(%lhs: f32, %rhs: f32):
    %sum = arith.addf %lhs, %rhs : f32
    distributed.DistributedYield (%sum : f32)
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
}
