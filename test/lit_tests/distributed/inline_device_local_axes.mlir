// RUN: enzymexlamlir-opt --split-input-file --inline-device-local-axes %s | FileCheck %s

// A DeviceLocalAxis factor that only ever appears alongside reduction_groups
// (never in the mapping) means the whole reduction group is purely local.
// Where does the "growth" for that factor actually show up, if not on the
// collective? A cast grows its own result -- the real, non-degenerate
// tensor<4xf32> local chunk of the reduction axis -- and a kernel's own
// local `stablehlo.reduce` genuinely consumes that whole chunk, producing a
// real scalar (not a placeholder awaiting growth: an actual reduction
// happened). By the time that scalar reaches the collective, there's
// nothing left of this factor to grow anywhere; the collective's job is
// just to drop it from its own reduction_groups/input_mesh bookkeeping and
// keep only the genuinely cross-device (physical) factor.
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]
  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %rf = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>

  %d = distributed.DeviceLocalAxis 4 : !distributed.device_local_axis<4>
  %df = axis.factor %d : !distributed.device_local_axis<4> <4, 1>

  %global = stablehlo.constant dense<0.0> : tensor<4xf32>
  %cast_axes = axis.product (%df : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>)
  %local = distributed.CastGlobalToLocal %global axes (%cast_axes : !axis.factor_group<4>) : tensor<4xf32> -> tensor<1xf32>

  %sum = "distributed.DistributedKernel"(%local, %cast_axes) <{
      argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>,
      operandSegmentSizes = array<i32: 1, 1>,
      output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [] : unreduced_axes = []>]>}> ({
  ^bb0(%arg0: tensor<4xf32>):
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%arg0 init: %z) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    "distributed.DistributedYield"(%r) : (tensor<f32>) -> ()
  }) : (tensor<1xf32>, !axis.factor_group<4>) -> tensor<f32>

  %mesh_in = axis.product (%rf : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %df : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>)
  %mesh_out = axis.product (%rf : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %reduction = axis.product (%df : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>)

  %lhs_a = axis.product (%rf : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %rhs_a = axis.product (%rf : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map = axis.map %lhs_a to %rhs_a : [!axis.factor_group<2>] [!axis.factor_group<2>]

  %h = distributed.Collective %sum : tensor<f32> on %mesh_in : !axis.factor_group<8> to tensor<f32> on %mesh_out : !axis.factor_group<2> reduces (%reduction : !axis.factor_group<4>) maps %map : !axis.map {
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %s = stablehlo.add %lhs, %rhs : tensor<f32>
    stablehlo.return %s : tensor<f32>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<f32>> -> tensor<f32>
}

// CHECK-LABEL: module {
// CHECK-DAG: %[[RF:.*]] = axis.factor %{{.*}} : !distributed.physical_comm_axis<2, 1><2, 1>
// CHECK-DAG: %[[MESH_IN:.*]] = axis.product (%[[RF]] : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
// CHECK: %[[LOCAL:.*]] = distributed.CastGlobalToLocal %{{.*}} axes (%{{.*}} : !axis.factor_group<1>) : tensor<4xf32> -> tensor<4xf32>
// The kernel's own operand is already grown by the cast above -- no remark
// about it, and no remark about the dropped reduction group either, since
// its producer really is a distributed.DistributedKernel.
// CHECK-NOT: remark: inline-device-local-axes
// CHECK: %[[SUM:.*]] = distributed.DistributedKernel (%[[LOCAL]] : tensor<4xf32>)
// CHECK-DAG: %[[MESH_OUT:.*]] = axis.product (%[[RF]] : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
// CHECK-DAG: %[[MAP:.*]] = axis.map %{{.*}} to %{{.*}} : [!axis.factor_group<2>] [!axis.factor_group<2>]
// CHECK: distributed.Collective %[[SUM]] : tensor<f32> on %[[MESH_IN]] : <2> to tensor<f32> on %[[MESH_OUT]] : <2> reduces () maps %[[MAP]] : !axis.map

// -----

// A DeviceLocalAxis factor that passes through the mapping (present in both
// input_mesh/output_mesh, paired via mapping_lhs/mapping_rhs, each anchored
// by the real ShapeAxis factor for the tensor dimension it belongs to --
// exactly how MaterializeDistributedCollectives.cpp's
// toLocallyTypedAxisProduct builds a collective's mapping) is never dropped
// or merged into the anchor: it's replaced, in the same position, by a
// same-extent ShapeAxisType sub-factor for the dimension it belongs to, and
// the dimension's original anchor keeps its own original extent but shifts
// to a stride above it (this pass's own contiguous/minor-most
// precondition). Preserving that position/stride distinction -- rather than
// collapsing everything into one grown anchor -- is what would let a real
// scatter/gather still be told apart from an ordinary pass-through by
// something reading the mapping downstream. On top of that, output_type/
// async_handle (and any DistributedAwait using it) grow to reflect the
// same factor. input_object is fed by a DistributedCastGlobalToLocal using
// the *same* DeviceLocalAxis factor, which this pass also grows -- growing
// input_object changes its actual type, which makes mapping_lhs's own
// (otherwise-untouched) anchor stale too, exercising the same rebuild path
// on both sides of the mapping, not just rhs.
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]
  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %rf = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>

  %d = distributed.DeviceLocalAxis 4 : !distributed.device_local_axis<4>
  %df = axis.factor %d : !distributed.device_local_axis<4> <4, 1>

  %ta = axis.getaxis tensor<1xf32> 0
  %taf_in = axis.factor %ta : !axis.shape_axis<tensor<1xf32>, 0> <1, 1>
  %taf_out = axis.factor %ta : !axis.shape_axis<tensor<1xf32>, 0> <1, 1>

  %global = stablehlo.constant dense<0.0> : tensor<8xf32>
  %cast_axes = axis.product (%rf : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %df : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>)
  %local = distributed.CastGlobalToLocal %global axes (%cast_axes : !axis.factor_group<8>) : tensor<8xf32> -> tensor<1xf32>

  %mesh_in = axis.product (%rf : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %df : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>)
  %mesh_out = axis.product (%rf : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %df : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>)

  %lhs_b = axis.product (%rf : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %df : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>, %taf_in : !axis.axis_factor<!axis.shape_axis<tensor<1xf32>, 0>, 1, 1>)
  %rhs_b = axis.product (%rf : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %df : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>, %taf_out : !axis.axis_factor<!axis.shape_axis<tensor<1xf32>, 0>, 1, 1>)
  %map = axis.map %lhs_b to %rhs_b : [!axis.factor_group<8>] [!axis.factor_group<8>]

  %h = distributed.Collective %local : tensor<1xf32> on %mesh_in : !axis.factor_group<8> to tensor<1xf32> on %mesh_out : !axis.factor_group<8> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<1xf32>> -> tensor<1xf32>
}

// CHECK-LABEL: module {
// CHECK-DAG: %[[RF:.*]] = axis.factor %{{.*}} : !distributed.physical_comm_axis<2, 1><2, 1>
// CHECK-DAG: %[[MESH_IN:.*]] = axis.product (%[[RF]] : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
// CHECK-DAG: %[[MESH_OUT:.*]] = axis.product (%[[RF]] : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
// The DeviceLocalAxis factor is replaced by a same-extent (4, stride 1)
// sub-factor, and the original anchor (extent 1, unchanged) shifts to
// stride 4 above it -- not merged into one grown anchor.
// CHECK-DAG: %[[LHS_AXIS:.*]] = axis.getaxis tensor<4xf32> 0
// CHECK-DAG: %[[LHS_LOCAL:.*]] = axis.factor %[[LHS_AXIS]] : !axis.shape_axis<tensor<4xf32>, 0><4, 1>
// CHECK-DAG: %[[LHS_ANCHOR:.*]] = axis.factor %[[LHS_AXIS]] : !axis.shape_axis<tensor<4xf32>, 0><1, 4>
// CHECK-DAG: %[[LHS:.*]] = axis.product (%[[RF]] : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %[[LHS_LOCAL]] : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>, %[[LHS_ANCHOR]] : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 1, 4>)
// CHECK-DAG: %[[RHS_AXIS:.*]] = axis.getaxis tensor<4xf32> 0
// CHECK-DAG: %[[RHS_LOCAL:.*]] = axis.factor %[[RHS_AXIS]] : !axis.shape_axis<tensor<4xf32>, 0><4, 1>
// CHECK-DAG: %[[RHS_ANCHOR:.*]] = axis.factor %[[RHS_AXIS]] : !axis.shape_axis<tensor<4xf32>, 0><1, 4>
// CHECK-DAG: %[[RHS:.*]] = axis.product (%[[RF]] : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %[[RHS_LOCAL]] : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>, %[[RHS_ANCHOR]] : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 1, 4>)
// CHECK-DAG: %[[MAP:.*]] = axis.map %[[LHS]] to %[[RHS]] : [!axis.factor_group<8>] [!axis.factor_group<8>]
// CHECK: %[[LOCAL:.*]] = distributed.CastGlobalToLocal %{{.*}} axes (%{{.*}} : !axis.factor_group<2>) : tensor<8xf32> -> tensor<4xf32>
// CHECK: %[[H:.*]] = distributed.Collective %[[LOCAL]] : tensor<4xf32> on %[[MESH_IN]] : <2> to tensor<4xf32> on %[[MESH_OUT]] : <2> reduces () maps %[[MAP]] : !axis.map
// CHECK: distributed.Await %[[H]] : <tensor<4xf32>> -> tensor<4xf32>

// -----

// A DeviceLocalAxis factor feeding a distributed.DistributedCall through its
// argument cast must be reconciled on the call's own argument_shardings/
// partitioning_axes, not just on the cast: the call is its own local/global
// boundary (like a DistributedKernelOp), checked against its callee's fixed
// GLOBAL-scope function-type input/result types, so its own declared
// binding has to shrink in step with the cast that grew its operand or the
// two disagree about how big the growth already was.
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]
  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %rf = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>

  %d = distributed.DeviceLocalAxis 4 : !distributed.device_local_axis<4>
  %df = axis.factor %d : !distributed.device_local_axis<4> <4, 1>
  %axes_grp = axis.product (%df : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>)

  "distributed.DistributedFunction"(%axes_grp) <{
    argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>,
    function_type = (tensor<4xf32>) -> tensor<4xf32>,
    output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>,
    sym_name = "identity",
    sym_visibility = "private"
  }> ({
  ^bb0(%arg0: tensor<4xf32>):
    distributed.DistributedYield (%arg0 : tensor<4xf32>)
  }) : (!axis.factor_group<4>) -> ()

  %global = stablehlo.constant dense<0.0> : tensor<4xf32>
  %local = distributed.CastGlobalToLocal %global axes (%axes_grp : !axis.factor_group<4>) : tensor<4xf32> -> tensor<1xf32>
  %call = distributed.DistributedCall @identity (%local : tensor<1xf32>) <[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>
      -> (tensor<1xf32>) <[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>
      axes (%axes_grp : !axis.factor_group<4>)
}

// CHECK-NOT: remark: inline-device-local-axes
// CHECK: %[[LOCAL:.*]] = distributed.CastGlobalToLocal %{{.*}} axes (%{{.*}} : !axis.factor_group<1>) : tensor<4xf32> -> tensor<4xf32>
// CHECK: distributed.DistributedCall @identity (%[[LOCAL]] : tensor<4xf32>) <[<dim_partitioning_axes = {{\[\[0\]\]}} : unreduced_axes = []>]>
// CHECK-NEXT: -> (tensor<4xf32>)
// CHECK-NEXT: axes (%{{.*}} : !axis.factor_group<1>)
