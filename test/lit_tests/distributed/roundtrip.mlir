// RUN: enzymexlamlir-opt --split-input-file %s | FileCheck %s

// Roundtrip mesh metadata, distributed.DistributedFunction, and
// distributed.DistributedYield.
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 3>, !distributed.physical_comm_axis<3, 1>]

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 3>, !distributed.physical_comm_axis<3, 1>
  %l0, %l1 = distributed.LogicalMeshAxes [2, 3] : !distributed.logical_mesh_axis<2>, !distributed.logical_mesh_axis<3>
  %r0 = distributed.ReplicationAxis 4 : !distributed.replication_axis<4>

  %axis = axis.getaxis tensor<12xf32> 0
  %f0 = axis.factor %axis : !axis.shape_axis<tensor<12xf32>, 0> <2, 6>
  %f1 = axis.factor %axis : !axis.shape_axis<tensor<12xf32>, 0> <2, 3>
  %ctx_callee = axis.product %f0, %f1 : !axis.axis_factor<!axis.shape_axis<tensor<12xf32>, 0>, 2, 6>, !axis.axis_factor<!axis.shape_axis<tensor<12xf32>, 0>, 2, 3>

  "distributed.DistributedFunction"(%ctx_callee) <{argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>, function_type = (tensor<12xf32>) -> tensor<12xf32>, output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = [0]>]>, sym_name = "identity"}> ({
  ^bb0(%arg0: tensor<12xf32>):
    distributed.DistributedYield %arg0 tensor<12xf32>
  }) : (!axis.factor_group<4>) -> ()
}

// CHECK-LABEL: module {
// CHECK: distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 3>, !distributed.physical_comm_axis<3, 1>]
// CHECK: %{{.*}}:2 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 3>, !distributed.physical_comm_axis<3, 1>
// CHECK: %{{.*}}:2 = distributed.LogicalMeshAxes [2, 3] : !distributed.logical_mesh_axis<2>, !distributed.logical_mesh_axis<3>
// CHECK: %{{.*}} = distributed.ReplicationAxis 4 : <4>
// CHECK: "distributed.DistributedFunction"(%{{.*}}) <{argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = {{\[\[}}0{{\]\]}} : unreduced_axes = []>]>, function_type = (tensor<12xf32>) -> tensor<12xf32>, output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = {{\[\[}}0{{\]\]}} : unreduced_axes = [0]>]>, sym_name = "identity"}> ({
// CHECK: distributed.DistributedYield %{{.*}} tensor<12xf32>
// CHECK: }) : (!axis.factor_group<4>) -> ()

// -----

// Roundtrip distributed.DistributedKernel custom assembly.
module {
  %axis = axis.getaxis tensor<8xf32> 0
  %factor = axis.factor %axis : !axis.shape_axis<tensor<8xf32>, 0> <2, 4>
  %ctx = axis.product %factor : !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 2, 4>
  %input = tensor.empty() : tensor<8xf32>

  %result = distributed.DistributedKernel %input : tensor<8xf32> #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>
    -> tensor<8xf32> #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>
    axes %ctx : !axis.factor_group<2> {
  ^bb0(%arg0: tensor<8xf32>):
    distributed.DistributedYield %arg0 tensor<8xf32>
  }
}

// CHECK-LABEL: module {
// CHECK: %{{.*}} = distributed.DistributedKernel %{{.*}} : tensor<8xf32> <[<dim_partitioning_axes = {{\[\[}}0{{\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: -> tensor<8xf32> <[<dim_partitioning_axes = {{\[\[}}0{{\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: axes %{{.*}} : !axis.factor_group<2> {
// CHECK: distributed.DistributedYield %{{.*}} tensor<8xf32>

// -----

// Roundtrip distributed.Collective / distributed.Await.
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>

  %l0, %l1 = distributed.LogicalMeshAxes [2, 2] : !distributed.logical_mesh_axis<2>, !distributed.logical_mesh_axis<2>
  %lf0 = axis.factor %l0 : !distributed.logical_mesh_axis<2> <2, 1>
  %lf1 = axis.factor %l1 : !distributed.logical_mesh_axis<2> <2, 1>
  %ta = axis.getaxis tensor<8xf32> 0
  %ta_out = axis.getaxis tensor<4xf32> 0
  %tf_out = axis.factor %ta_out : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %tf_to_mesh = axis.factor %ta : !axis.shape_axis<tensor<8xf32>, 0> <4, 2>
  %tf_remain = axis.factor %ta : !axis.shape_axis<tensor<8xf32>, 0> <2, 1>
  %mesh_in = axis.product %lf0, %lf1 : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>, !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>
  %mesh_out = axis.product %lf0, %lf1 : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>, !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>
  %reduction = axis.product %lf1 : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>
  %lhs_group_1 = axis.product %tf_to_mesh : !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 4, 2>
  // rhs_group_1 = %mesh_out
  %lhs_group_2 = axis.product %lf0, %tf_remain : !axis.axis_factor<!distributed.logical_mesh_axis<2>, 2, 1>, !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 2, 1>
  %rhs_group_2 = axis.product %tf_out : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>
  %mapping = axis.map %lhs_group_1, %lhs_group_2 to %mesh_out, %rhs_group_2 : [!axis.factor_group<4>, !axis.factor_group<4>] [!axis.factor_group<4>, !axis.factor_group<4>]
  %input = tensor.empty() : tensor<8xf32>

  %h = distributed.Collective %input : tensor<8xf32> on %mesh_in : !axis.factor_group<4> to tensor<4xf32> on %mesh_out : !axis.factor_group<4> reduces (%reduction : !axis.factor_group<2>) maps %mapping : !axis.map {
  ^bb0(%lhs: f32, %rhs: f32):
    %sum = arith.addf %lhs, %rhs : f32
    distributed.DistributedYield %sum f32
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
}

// CHECK: distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]
// CHECK: axis.map %{{.*}}, %{{.*}} to %{{.*}}, %{{.*}} : [!axis.factor_group<4>, !axis.factor_group<4>] [!axis.factor_group<4>, !axis.factor_group<4>]
// CHECK: %{{.*}} = distributed.Collective %{{.*}} : tensor<8xf32> on %{{.*}} : <4> to tensor<4xf32> on %{{.*}} : <4> reduces (%{{.*}} : !axis.factor_group<2>) maps %{{.*}} : !axis.map {
// CHECK: arith.addf
// CHECK: distributed.DistributedYield %{{.*}} f32
// CHECK: %{{.*}} = distributed.Await %{{.*}} : <tensor<4xf32>> -> tensor<4xf32>

// -----

// Roundtrip module-level metadata-only body.
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 1>]

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 1>
  %a = axis.getaxis tensor<4xf32> 0
}

// CHECK: distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 1>]
// CHECK: %{{.*}} = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 1>
