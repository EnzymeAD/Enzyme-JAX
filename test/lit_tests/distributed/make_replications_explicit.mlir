// RUN: enzymexlamlir-opt --split-input-file --distributed-make-replications-explicit %s | FileCheck %s

// A physical axis (axis1) entirely absent from both input_mesh and
// output_mesh gets an explicit ReplicationAxis factor folded into each
// mesh operand independently, plus one new mapping pair per direction
// (Replicate-->realAxis1 from the "missing from output" fix,
// realAxis1-->Replicate from the "missing from input" fix) -- two
// independent fixes, not a single merged pair, per this pass's own design
// (see MakeReplicationsExplicit.cpp's top-of-file rationale).
// CHECK-LABEL: module @missing_from_both
// CHECK: distributed.ReplicationAxis 2
// CHECK: distributed.ReplicationAxis 2
// CHECK: distributed.Collective {{.*}} on %{{.*}} : <4> to {{.*}} on %{{.*}} : <4>
module @missing_from_both {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %f0 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>

  %mesh_in = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %mesh_out = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %lhs = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %rhs = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %map = axis.map %lhs to %rhs : [!axis.factor_group<2>] [!axis.factor_group<2>]

  %global = tensor.empty() : tensor<2xf32>
  %local = distributed.CastGlobalToLocal %global axes (%mesh_in : !axis.factor_group<2>) : tensor<2xf32> -> tensor<1xf32>
  %h = distributed.Collective %local : tensor<1xf32> on %mesh_in : !axis.factor_group<2> to tensor<1xf32> on %mesh_out : !axis.factor_group<2> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<1xf32>> -> tensor<1xf32>
}

// -----

// A physical axis missing from only one side (here: axis1, legitimately
// consumed by reduction_groups on the input side, so output_mesh correctly
// never mentions it) gets exactly one new Replicate factor -- only the
// "missing from output" direction fires, since input_mesh already fully
// accounts for both axes.
// CHECK-LABEL: module @missing_from_output_only
// CHECK: distributed.ReplicationAxis 2
// CHECK-NOT: distributed.ReplicationAxis
// CHECK: distributed.Collective {{.*}} on %{{.*}} : <4> to {{.*}} on %{{.*}} : <4>
module @missing_from_output_only {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %f0 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f1 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>

  %mesh_in = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %mesh_out = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %reduction = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)

  %lhs = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %rhs = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %map = axis.map %lhs to %rhs : [!axis.factor_group<2>] [!axis.factor_group<2>]

  %global = tensor.empty() : tensor<4xf32>
  %local = distributed.CastGlobalToLocal %global axes (%mesh_in : !axis.factor_group<4>) : tensor<4xf32> -> tensor<1xf32>
  %h = distributed.Collective %local : tensor<1xf32> on %mesh_in : !axis.factor_group<4> to tensor<1xf32> on %mesh_out : !axis.factor_group<2> reduces (%reduction : !axis.factor_group<2>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %sum = stablehlo.add %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<1xf32>> -> tensor<1xf32>
}

// -----

// Every physical axis already explicit on both mesh operands: idempotent
// no-op, no ReplicationAxis is synthesized.
// CHECK-LABEL: module @already_explicit
// CHECK-NOT: distributed.ReplicationAxis
// CHECK: distributed.Collective {{.*}} on %{{.*}} : <4> to {{.*}} on %{{.*}} : <4>
module @already_explicit {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %f0 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f1 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>

  %mesh_in = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %mesh_out = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)

  %lhs0 = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %rhs0 = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %lhs1 = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %rhs1 = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<2>, !axis.factor_group<2>]

  %global = tensor.empty() : tensor<4xf32>
  %local = distributed.CastGlobalToLocal %global axes (%mesh_in : !axis.factor_group<4>) : tensor<4xf32> -> tensor<1xf32>
  %h = distributed.Collective %local : tensor<1xf32> on %mesh_in : !axis.factor_group<4> to tensor<1xf32> on %mesh_out : !axis.factor_group<4> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<1xf32>> -> tensor<1xf32>
}

// -----

// A physical axis only partially covered by the collective's own mesh
// operand (here: a stride-2 half of a 4-way physical axis) requires
// subtractSpace to synthesize a genuinely new complementary sub-factor,
// rather than passing an existing whole-axis factor through unchanged --
// this is the path that needs the missing factor spliced into the module
// before being used (see MakeReplicationsExplicit.cpp's materialize step).
// CHECK-LABEL: module @missing_partial_factor
// CHECK: distributed.ReplicationAxis 2
// CHECK: distributed.ReplicationAxis 2
// CHECK: distributed.Collective {{.*}} on %{{.*}} : <4> to {{.*}} on %{{.*}} : <4>
module @missing_partial_factor {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 1>
  %f0 = axis.factor %p0 : !distributed.physical_comm_axis<4, 1> <2, 2>

  %mesh_in = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 2, 2>)
  %mesh_out = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 2, 2>)
  %lhs = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 2, 2>)
  %rhs = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 2, 2>)
  %map = axis.map %lhs to %rhs : [!axis.factor_group<2>] [!axis.factor_group<2>]

  %global = tensor.empty() : tensor<2xf32>
  %local = distributed.CastGlobalToLocal %global axes (%mesh_in : !axis.factor_group<2>) : tensor<2xf32> -> tensor<1xf32>
  %h = distributed.Collective %local : tensor<1xf32> on %mesh_in : !axis.factor_group<2> to tensor<1xf32> on %mesh_out : !axis.factor_group<2> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<1xf32>> -> tensor<1xf32>
}
