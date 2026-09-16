// RUN: enzymexlamlir-opt --canonicalize --cse %s | FileCheck %s

// General home for testing --canonicalize/--cse behavior on the distributed
// dialect's own ops (as opposed to a specific pass like
// canonicalize-sharded-factor-order.mlir).

// LogicalMeshAxesOp and DeviceLocalAxisOp share the exact same
// canonicalization story (see PruneUnusedMetadataAxis in AxisOps.cpp):
// DeclarativeMetadataTrait opts each op out of Pure, and so out of MLIR's
// generic DCE/CSE too, since both key off the same memory-effect-free check.
// An unused declaration is pruned by a dedicated canonicalizer instead.
// CHECK-LABEL: func.func @unused_logical_mesh_axis_pruned
// CHECK-NOT: distributed.LogicalMeshAxes
func.func @unused_logical_mesh_axis_pruned() {
  %l = distributed.LogicalMeshAxes 4 : !distributed.logical_mesh_axis<4>
  return
}

// CHECK-LABEL: func.func @unused_device_local_axis_pruned
// CHECK-NOT: distributed.DeviceLocalAxis
func.func @unused_device_local_axis_pruned() {
  %d = distributed.DeviceLocalAxis 4 : !distributed.device_local_axis<4>
  return
}

// Two textually-identical declarations of either op must NOT be CSE'd
// together: each denotes a distinct axis, and identity is the SSA value
// itself, not the extent -- an extent-only equivalence can't tell "the same
// axis reused" apart from "two unrelated axes that happen to have the same
// size" (see LogicalMeshAxisType::equivalent/DeviceLocalAxisType::equivalent).
// CHECK-LABEL: func.func @two_distinct_logical_mesh_axes_not_cse
// CHECK: %[[L1:.*]] = distributed.LogicalMeshAxes 4
// CHECK: %[[L2:.*]] = distributed.LogicalMeshAxes 4
// CHECK: axis.factor %[[L1]]
// CHECK: axis.factor %[[L2]]
func.func @two_distinct_logical_mesh_axes_not_cse() -> !axis.map {
  %l1 = distributed.LogicalMeshAxes 4 : !distributed.logical_mesh_axis<4>
  %l2 = distributed.LogicalMeshAxes 4 : !distributed.logical_mesh_axis<4>
  %f1 = axis.factor %l1 : !distributed.logical_mesh_axis<4><4, 1>
  %f2 = axis.factor %l2 : !distributed.logical_mesh_axis<4><4, 1>
  %p1 = axis.product (%f1 : !axis.axis_factor<!distributed.logical_mesh_axis<4>, 4, 1>)
  %p2 = axis.product (%f2 : !axis.axis_factor<!distributed.logical_mesh_axis<4>, 4, 1>)
  %m = axis.map %p1 to %p2 : [!axis.factor_group<4>] [!axis.factor_group<4>]
  return %m : !axis.map
}

// CHECK-LABEL: func.func @two_distinct_device_local_axes_not_cse
// CHECK: %[[D1:.*]] = distributed.DeviceLocalAxis 4
// CHECK: %[[D2:.*]] = distributed.DeviceLocalAxis 4
// CHECK: axis.factor %[[D1]]
// CHECK: axis.factor %[[D2]]
func.func @two_distinct_device_local_axes_not_cse() -> !axis.map {
  %d1 = distributed.DeviceLocalAxis 4 : !distributed.device_local_axis<4>
  %d2 = distributed.DeviceLocalAxis 4 : !distributed.device_local_axis<4>
  %f1 = axis.factor %d1 : !distributed.device_local_axis<4><4, 1>
  %f2 = axis.factor %d2 : !distributed.device_local_axis<4><4, 1>
  %p1 = axis.product (%f1 : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>)
  %p2 = axis.product (%f2 : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>)
  %m = axis.map %p1 to %p2 : [!axis.factor_group<4>] [!axis.factor_group<4>]
  return %m : !axis.map
}
