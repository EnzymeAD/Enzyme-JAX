// RUN: true
// Temporarily disabled: enzymexlamlir-opt --split-input-file --naive-logical-to-physical-mesh --canonicalize --distributed-to-hlo %s | FileCheck %s

module {

  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 2>, !distributed.physical_comm_axis<2, 1>]

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 2>, !distributed.physical_comm_axis<2, 1>

  %l0, %l1 = distributed.LogicalMeshAxes [4, 2] : !distributed.logical_mesh_axis<4>, !distributed.logical_mesh_axis<2>
  %lf0_upper = axis.factor %l0 : !distributed.logical_mesh_axis<4> <2, 2>
  %lf0_lower = axis.factor %l0 : !distributed.logical_mesh_axis<4> <2, 1>
  %lf1 = axis.factor %l1 : !distributed.logical_mesh_axis<2> <2, 1>

  %ta = axis.getaxis tensor<8xf32> 0
  %tf0 = axis.factor %ta : !axis.shape_axis<tensor<8xf32>, 0> <2, 1>
  %tf1 = axis.factor %ta : !axis.shape_axis<tensor<8xf32>, 0> <2, 2>
  %tf2 = axis.factor %ta : !axis.shape_axis<tensor<8xf32>, 0> <2, 4>

  %ta_out = axis.getaxis tensor<2xf32> 0
  %tf_out = axis.factor %ta_out : !axis.shape_axis<tensor<2xf32>, 0> <2, 1>

  %mesh_in = axis.product (%lf0_upper : !axis.axis_factor<!distributed.logical_mesh_axis<4>, 2, 2>, %lf0_lower : !axis.axis_factor<!distributed.logical_mesh_axis<4>, 2, 1>, %lf1 : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>)
  %mesh_out = axis.product (%lf0_upper : !axis.axis_factor<!distributed.logical_mesh_axis<4>, 2, 2>, %lf0_lower : !axis.axis_factor<!distributed.logical_mesh_axis<4>, 2, 1>, %lf1 : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>)
  %reduction = axis.product (%lf0_upper : !axis.axis_factor<!distributed.logical_mesh_axis<4>, 2, 2>, %lf1 : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>)

  %lhs_group_0 = axis.product (%tf0 : !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 2, 1>)
  %rhs_group_0 = axis.product (%tf_out : !axis.axis_factor<!axis.shape_axis<tensor<2xf32>, 0>, 2, 1>)

  // rhs group 1 = reduction
  %lhs_group_1 = axis.product (%tf1 : !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 2, 2>, %tf2 : !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 2, 4>)

  %lhs_group_2 = axis.product (%lf0_lower : !axis.axis_factor<!distributed.logical_mesh_axis<4>, 2, 1>)
  %rhs_group_2 = axis.product (%lf0_lower : !axis.axis_factor<!distributed.logical_mesh_axis<4>, 2, 1>)

  %mapping = axis.map %lhs_group_0, %lhs_group_1, %lhs_group_2 to %rhs_group_0, %reduction, %rhs_group_2 : [!axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<2>]

  %c = stablehlo.constant dense<0.0> : tensor<8xf32>
  %h = distributed.Collective %c : tensor<8xf32> on %mesh_in : !axis.factor_group<8> to tensor<2xf32> on %mesh_out : !axis.factor_group<8> reduces (%reduction : !axis.factor_group<4>) maps %mapping : !axis.map {
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<2xf32>> -> tensor<2xf32>
}

// CHECK-LABEL: module {
// CHECK: stablehlo.async_start
// CHECK: stablehlo.all_reduce
// CHECK: stablehlo.add
// CHECK: stablehlo.async_done
// CHECK-NOT: distributed.Collective
// CHECK-NOT: distributed.Await
