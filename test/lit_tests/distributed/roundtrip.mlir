// RUN: enzymexlamlir-opt --split-input-file %s | FileCheck %s

// Roundtrip distributed.Function / distributed.DistributedCall /
// distributed.DistributedYield.
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2>, !distributed.physical_comm_axis<3>]

  distributed.MeshComputation @mc mesh @mesh0 {
    %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2>, !distributed.physical_comm_axis<3>

    %axis = axis.getaxis tensor<12xf32> 0
    %f0 = axis.factor %axis : (!axis.shape_axis<tensor<12xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<12xf32>, 0>, 2, 6>
    %f1 = axis.factor %axis : (!axis.shape_axis<tensor<12xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<12xf32>, 0>, 2, 3>
    %f2 = axis.factor %axis : (!axis.shape_axis<tensor<12xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<12xf32>, 0>, 3, 1>
    %ctx_callee = axis.product %f0, %f1 : !axis.axis_factor<!axis.shape_axis<tensor<12xf32>, 0>, 2, 6>, !axis.axis_factor<!axis.shape_axis<tensor<12xf32>, 0>, 2, 3>
    %ctx_rep = axis.product %f2 : !axis.axis_factor<!axis.shape_axis<tensor<12xf32>, 0>, 3, 1>
    %ctx_caller = axis.product %f0, %f1, %f2 : !axis.axis_factor<!axis.shape_axis<tensor<12xf32>, 0>, 2, 6>, !axis.axis_factor<!axis.shape_axis<tensor<12xf32>, 0>, 2, 3>, !axis.axis_factor<!axis.shape_axis<tensor<12xf32>, 0>, 3, 1>

    distributed.Function @callee context %ctx_callee : !axis.factor_group<4> arg_types [i32] ret_types [i32] {
    ^bb0(%arg0: i32):
      distributed.DistributedYield %arg0 i32
    }

    distributed.Function @caller context %ctx_caller : !axis.factor_group<12> arg_types [i32] ret_types [i32] {
    ^bb0(%arg0: i32):
      %r = distributed.DistributedCall @callee replicate_over %ctx_rep %arg0 : !axis.factor_group<3>, i32 -> i32
      distributed.DistributedYield %r i32
    }
  }
}

// CHECK-LABEL: module {
// CHECK: distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2>, !distributed.physical_comm_axis<3>]
// CHECK: %{{.*}}:2 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2>, !distributed.physical_comm_axis<3>
// CHECK: distributed.Function @callee context
// CHECK: distributed.Function @caller context
// CHECK: distributed.DistributedCall @callee replicate_over

// -----

// Roundtrip distributed.Collective / distributed.Await.
module {
  func.func @add(%lhs: f32, %rhs: f32) -> f32 {
    %0 = arith.addf %lhs, %rhs : f32
    return %0 : f32
  }

  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2>, !distributed.physical_comm_axis<2>]

  distributed.MeshComputation @mc mesh @mesh0 {
    %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2>, !distributed.physical_comm_axis<2>

    %l0, %l1 = distributed.LogicalMeshAxes [2, 2] : !distributed.logical_mesh_axis<2>, !distributed.logical_mesh_axis<2>
    %lf0 = axis.factor %l0 : (!distributed.logical_mesh_axis<2>) -> !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>
    %lf1 = axis.factor %l1 : (!distributed.logical_mesh_axis<2>) -> !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>
    %ta = axis.getaxis tensor<8xf32> 0
    %ta_out = axis.getaxis tensor<4xf32> 0
    %tf_out = axis.factor %ta_out : (!axis.shape_axis<tensor<4xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>
    %tf_to_mesh = axis.factor %ta : (!axis.shape_axis<tensor<8xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 4, 2>
    %tf_remain = axis.factor %ta : (!axis.shape_axis<tensor<8xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 2, 1>
    %mesh_in = axis.product %lf0, %lf1 : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>, !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>
    %mesh_out = axis.product %lf0, %lf1 : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>, !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>
    %reduction = axis.product %lf1 : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>
    %lhs_group_1 = axis.product %tf_to_mesh : !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 4, 2>
    // rhs_group_1 = %mesh_out
    %lhs_group_2 = axis.product %lf0, %tf_remain : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>, !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 2, 1>
    %rhs_group_2 = axis.product %tf_out : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>
    %mapping = axis.map %lhs_group_1, %lhs_group_2 to %mesh_out, %rhs_group_2 : [!axis.factor_group<4>, !axis.factor_group<4>] [!axis.factor_group<4>, !axis.factor_group<4>]
    
    distributed.Function @collective context %mesh_in : !axis.factor_group<4> arg_types [tensor<8xf32>] ret_types [tensor<4xf32>] {
    ^bb0(%arg0: tensor<8xf32>):
      %h = distributed.Collective %arg0 : tensor<8xf32> on %mesh_in : !axis.factor_group<4> to tensor<4xf32> on %mesh_out : !axis.factor_group<4> reduces (%reduction @add) : !axis.factor_group<2> maps %mapping : !axis.map
      %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
      distributed.DistributedYield %v tensor<4xf32>
    }
  }
}

// CHECK: func.func @add(%{{.*}}: f32, %{{.*}}: f32) -> f32
// CHECK: distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2>, !distributed.physical_comm_axis<2>]
// CHECK: axis.map %{{.*}}, %{{.*}} to %{{.*}}, %{{.*}} : [!axis.factor_group<4>, !axis.factor_group<4>] [!axis.factor_group<4>, !axis.factor_group<4>]
// CHECK: %{{.*}} = distributed.Collective %{{.*}} : tensor<8xf32> on %{{.*}} : <4> to tensor<4xf32> on %{{.*}} : <4> reduces (%{{.*}} @add) : !axis.factor_group<2> maps %{{.*}} : !axis.map
// CHECK: %{{.*}} = distributed.Await %{{.*}} : <tensor<4xf32>> -> tensor<4xf32>

// -----

// Roundtrip distributed.MeshComputation with a minimal metadata-only body.
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4>]

  distributed.MeshComputation @mc mesh @mesh0 {
    %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4>
    %a = axis.getaxis tensor<4xf32> 0
  }
}

// CHECK: distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4>]
// CHECK: distributed.MeshComputation @mc mesh @mesh0
// CHECK: %{{.*}} = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4>
