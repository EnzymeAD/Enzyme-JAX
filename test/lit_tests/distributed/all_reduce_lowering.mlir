// RUN: enzymexlamlir-opt --split-input-file --naive-logical-to-physical-mesh --canonicalize --distributed-to-hlo %s | FileCheck %s

module {

  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 2>, !distributed.physical_comm_axis<2, 1>]

  distributed.MeshComputation @mc mesh @mesh0 {
      func.func @add(%lhs: tensor<f32>, %rhs: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %lhs, %rhs : tensor<f32>
    return %0 : tensor<f32>
  }
    %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 2>, !distributed.physical_comm_axis<2, 1>

    %l0, %l1 = distributed.LogicalMeshAxes [4, 2] : !distributed.logical_mesh_axis<4>, !distributed.logical_mesh_axis<2>
    %lf0_upper = axis.factor %l0 : !distributed.logical_mesh_axis<4> <2, 2>
    %lf0_lower = axis.factor %l0 : !distributed.logical_mesh_axis<4> <2, 1>
    %lf1 = axis.factor %l1 : !distributed.logical_mesh_axis<2> <2, 1>

    %ta = axis.getaxis tensor<8xf32> 0
    %tf0 = axis.factor %ta : !axis.shape_axis<tensor<8xf32>, 0> <8, 1>

    %r0 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
    %rf0 = axis.factor %r0 : !distributed.replication_axis<2> <2, 1>

    %mesh_in = axis.product %lf0_upper, %lf0_lower, %lf1 : !axis.axis_factor<!distributed.logical_mesh_axis<4>, 2, 2>, !axis.axis_factor<!distributed.logical_mesh_axis<4>, 2, 1>, !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>
    %mesh_out = axis.product %lf0_upper, %lf0_lower, %lf1 : !axis.axis_factor<!distributed.logical_mesh_axis<4>, 2, 2>, !axis.axis_factor<!distributed.logical_mesh_axis<4>, 2, 1>, !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>
    %reduction = axis.product %lf0_upper : !axis.axis_factor<!distributed.logical_mesh_axis<4>, 2, 2>

    %lhs_group_0 = axis.product %rf0 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>
    %rhs_group_0 = axis.product %lf0_upper : !axis.axis_factor<!distributed.logical_mesh_axis<4>, 2, 2>

    %lhs_group_1 = axis.product %lf0_lower : !axis.axis_factor<!distributed.logical_mesh_axis<4>, 2, 1>
    %rhs_group_1 = axis.product %lf0_lower : !axis.axis_factor<!distributed.logical_mesh_axis<4>, 2, 1>

    %lhs_group_2 = axis.product %lf1 : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>
    %rhs_group_2 = axis.product %lf1 : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>

    %lhs_group_3 = axis.product %tf0 : !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 8, 1>
    %rhs_group_3 = axis.product %tf0 : !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 8, 1>

    %mapping = axis.map %lhs_group_0, %lhs_group_1, %lhs_group_2, %lhs_group_3 to %rhs_group_0, %rhs_group_1, %rhs_group_2, %rhs_group_3 : [!axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<8>] [!axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<8>]

    distributed.Function @collective context %mesh_in : !axis.factor_group<8> arg_types [tensor<8xf32>] ret_types [tensor<8xf32>] {
    ^bb0(%arg0: tensor<8xf32>):
      %h = distributed.Collective %arg0 : tensor<8xf32> on %mesh_in : !axis.factor_group<8> to tensor<8xf32> on %mesh_out : !axis.factor_group<8> reduces (%reduction @add) : !axis.factor_group<2> maps %mapping : !axis.map
      %v = distributed.Await %h : !distributed.asynch_handle<tensor<8xf32>> -> tensor<8xf32>
      distributed.DistributedYield %v tensor<8xf32>
    }
  }
}

// CHECK-LABEL: module {
// CHECK: stablehlo.async_start
// CHECK: stablehlo.all_reduce
// CHECK: stablehlo.add
// CHECK: stablehlo.async_done
// CHECK-NOT: distributed.Collective
// CHECK-NOT: distributed.Await
