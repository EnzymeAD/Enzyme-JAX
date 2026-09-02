// RUN: enzymexlamlir-opt --naive-logical-to-physical-mesh %s | FileCheck %s

module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 1>]
  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 1>
  %l0 = distributed.LogicalMeshAxes [4] : !distributed.logical_mesh_axis<4>
  %lf0 = axis.factor %l0 : !distributed.logical_mesh_axis<4> <4, 1>
  %ctx = axis.product (%lf0 : !axis.axis_factor<!distributed.logical_mesh_axis<4>, 4, 1>)

  "distributed.DistributedFunction"(%ctx) <{argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>, function_type = (tensor<4xf32>) -> tensor<4xf32>, output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<4xf32>):
    distributed.DistributedYield %arg0 tensor<4xf32>
  }) : (!axis.factor_group<4>) -> ()
}

// CHECK: distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 1>]
// CHECK: %[[P:.*]] = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 1>
// CHECK-NOT: distributed.LogicalMeshAxes
// CHECK: %[[F:.*]] = axis.factor %[[P]] : !distributed.physical_comm_axis<4, 1><4, 1>
// CHECK: %[[CTX:.*]] = axis.product (%[[F]] : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
// CHECK: "distributed.DistributedFunction"(%[[CTX]])
