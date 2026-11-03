// RUN: enzymexlamlir-opt %s --canonicalize -split-input-file | FileCheck %s

// A local-global-local round trip over two equal factors of one mesh axis folds.
// CHECK-LABEL: func.func @fold_global_to_local_round_trip
// CHECK-SAME: (%[[LOCAL:.*]]: tensor<4x8xf32>)
// CHECK-NOT: distributed.Cast
// CHECK: %[[ANCHOR:.*]] = distributed.AnchorPartitioning %[[LOCAL]]
// CHECK-NOT: distributed.Cast
// CHECK: return %[[ANCHOR]]
distributed.PhysicalMesh @mesh device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]
func.func @fold_global_to_local_round_trip(%local: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %mesh:2 = distributed.GetPhysicalMeshAxes @mesh : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %first = axis.factor %mesh#0 : !distributed.physical_comm_axis<2, 2><2, 1>
  %second = axis.factor %mesh#0 : !distributed.physical_comm_axis<2, 2><2, 1>
  %first_group = axis.product (%first : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %second_group = axis.product (%second : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %unsplit = axis.product ()
  %global = distributed.CastLocalToGlobal %local axes (%first_group : !axis.factor_group<2>, %unsplit : !axis.factor_group<1>) : tensor<4x8xf32> -> tensor<8x8xf32>
  %back = distributed.CastGlobalToLocal %global axes (%second_group : !axis.factor_group<2>, %unsplit : !axis.factor_group<1>) : tensor<8x8xf32> -> tensor<4x8xf32>
  return %back : tensor<4x8xf32>
}

// -----

// A global-local-global round trip over factors from two mesh-axis lookups folds.
// CHECK-LABEL: func.func @fold_local_to_global_round_trip
// CHECK-SAME: (%[[GLOBAL:.*]]: tensor<8x8xf32>)
// CHECK-NOT: distributed.Cast
// CHECK: %[[ANCHOR:.*]] = distributed.AnchorPartitioning %[[GLOBAL]]
// CHECK-NOT: distributed.Cast
// CHECK: return %[[ANCHOR]]
distributed.PhysicalMesh @mesh device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]
func.func @fold_local_to_global_round_trip(%global: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %first_mesh:2 = distributed.GetPhysicalMeshAxes @mesh : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %second_mesh:2 = distributed.GetPhysicalMeshAxes @mesh : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %first = axis.factor %first_mesh#1 : !distributed.physical_comm_axis<2, 1><2, 1>
  %second = axis.factor %second_mesh#1 : !distributed.physical_comm_axis<2, 1><2, 1>
  %first_group = axis.product (%first : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %second_group = axis.product (%second : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %unsplit = axis.product ()
  %local = distributed.CastGlobalToLocal %global axes (%unsplit : !axis.factor_group<1>, %first_group : !axis.factor_group<2>) : tensor<8x8xf32> -> tensor<8x4xf32>
  %back = distributed.CastLocalToGlobal %local axes (%unsplit : !axis.factor_group<1>, %second_group : !axis.factor_group<2>) : tensor<8x4xf32> -> tensor<8x8xf32>
  return %back : tensor<8x8xf32>
}

// -----

// Negative: factors of equal extent but different strides do not fold.
// CHECK-LABEL: func.func @no_fold_round_trip_different_stride
// CHECK: distributed.CastLocalToGlobal
// CHECK: distributed.CastGlobalToLocal
// CHECK-NOT: distributed.AnchorPartitioning
distributed.PhysicalMesh @mesh device_target "cpu" axes [!distributed.physical_comm_axis<4, 1>]
func.func @no_fold_round_trip_different_stride(%local: tensor<4xf32>) -> tensor<4xf32> {
  %mesh = distributed.GetPhysicalMeshAxes @mesh : !distributed.physical_comm_axis<4, 1>
  %low = axis.factor %mesh : !distributed.physical_comm_axis<4, 1><2, 1>
  %high = axis.factor %mesh : !distributed.physical_comm_axis<4, 1><2, 2>
  %low_group = axis.product (%low : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 2, 1>)
  %high_group = axis.product (%high : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 2, 2>)
  %global = distributed.CastLocalToGlobal %local axes (%low_group : !axis.factor_group<2>) : tensor<4xf32> -> tensor<8xf32>
  %back = distributed.CastGlobalToLocal %global axes (%high_group : !axis.factor_group<2>) : tensor<8xf32> -> tensor<4xf32>
  return %back : tensor<4xf32>
}

// -----

// Negative: the same two factors listed in opposite orders do not fold.
// CHECK-LABEL: func.func @no_fold_round_trip_different_factor_order
// CHECK: distributed.CastLocalToGlobal
// CHECK: distributed.CastGlobalToLocal
// CHECK-NOT: distributed.AnchorPartitioning
distributed.PhysicalMesh @mesh device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]
func.func @no_fold_round_trip_different_factor_order(%local: tensor<2xf32>) -> tensor<2xf32> {
  %mesh:2 = distributed.GetPhysicalMeshAxes @mesh : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %major = axis.factor %mesh#0 : !distributed.physical_comm_axis<2, 2><2, 1>
  %minor = axis.factor %mesh#1 : !distributed.physical_comm_axis<2, 1><2, 1>
  %major_first = axis.product (%major : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %minor : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %minor_first = axis.product (%minor : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %major : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %global = distributed.CastLocalToGlobal %local axes (%major_first : !axis.factor_group<4>) : tensor<2xf32> -> tensor<8xf32>
  %back = distributed.CastGlobalToLocal %global axes (%minor_first : !axis.factor_group<4>) : tensor<8xf32> -> tensor<2xf32>
  return %back : tensor<2xf32>
}

// -----

// Negative: factors of two distinct DeviceLocalAxis declarations do not fold.
// CHECK-LABEL: func.func @no_fold_round_trip_distinct_device_local_axes
// CHECK: distributed.CastLocalToGlobal
// CHECK: distributed.CastGlobalToLocal
// CHECK-NOT: distributed.AnchorPartitioning
func.func @no_fold_round_trip_distinct_device_local_axes(%local: tensor<1xf32>) -> tensor<1xf32> {
  %first_axis = distributed.DeviceLocalAxis 4 : !distributed.device_local_axis<4>
  %second_axis = distributed.DeviceLocalAxis 4 : !distributed.device_local_axis<4>
  %first = axis.factor %first_axis : !distributed.device_local_axis<4><4, 1>
  %second = axis.factor %second_axis : !distributed.device_local_axis<4><4, 1>
  %first_group = axis.product (%first : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>)
  %second_group = axis.product (%second : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>)
  %global = distributed.CastLocalToGlobal %local axes (%first_group : !axis.factor_group<4>) : tensor<1xf32> -> tensor<4xf32>
  %back = distributed.CastGlobalToLocal %global axes (%second_group : !axis.factor_group<4>) : tensor<4xf32> -> tensor<1xf32>
  return %back : tensor<1xf32>
}
