// RUN: enzymexlamlir-opt --naive-logical-to-physical-mesh %s | FileCheck %s

module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 1>]
  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 1>
  %l0 = distributed.LogicalMeshAxes [4] : !distributed.logical_mesh_axis<4>
  %lf0 = axis.factor %l0 : !distributed.logical_mesh_axis<4> <4, 1>
  %ctx = axis.product %lf0 : !axis.axis_factor<!distributed.logical_mesh_axis<4>, 4, 1>

  distributed.Function @main context %ctx : !axis.factor_group<4> arg_types [i32] ret_types [i32] {
  ^bb0(%arg0: i32):
    distributed.DistributedYield %arg0 i32
  }
}

// CHECK: distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 1>]
// CHECK: %[[P:.*]] = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 1>
// CHECK-NOT: distributed.LogicalMeshAxes
// CHECK: %[[F:.*]] = axis.factor %[[P]] : !distributed.physical_comm_axis<4, 1><4, 1>
// CHECK: distributed.Function @main context
