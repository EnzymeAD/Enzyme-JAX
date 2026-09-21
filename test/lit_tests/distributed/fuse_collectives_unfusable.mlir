// RUN: enzymexlamlir-opt --split-input-file --distributed-fuse-collectives %s 2>&1 >/dev/null | FileCheck %s --check-prefix=REMARK
// RUN: enzymexlamlir-opt --split-input-file --distributed-fuse-collectives %s | FileCheck %s --check-prefix=IR

// Pairs distributed-fuse-collectives must leave alone for reasons that the
// print pass cannot report on (it asserts on a reduction over a tile atom, and
// distributed-atomize-collectives needs explicit replication), which is why
// these two live apart from fuse_collectives.mlir: only the fusion pass runs.

// The second collective sums over a piece of the tile (a device-local reduction) that the first collective produced from its input tile. The reduction
// model only covers mesh atoms, so the fused collective would reduce something outside it: left alone.
// IR-LABEL: module @second_reduces_local_tile
// IR: distributed.Collective
// IR: distributed.Collective
// IR-NOT: distributed.Collective
// REMARK: remark: fuse-collectives: not fused with the preceding collective (the second collective reduces over a device-local tile atom of the first collective's input)
module @second_reduces_local_tile {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>

  %ax1 = axis.getaxis tensor<4xf32> 0
  %ax2 = axis.getaxis tensor<4xf32> 0
  %ax3 = axis.getaxis tensor<2xf32> 0
  %in = tensor.empty() : tensor<4xf32>
  %f4 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh5 = axis.product (%f4 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f6 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh7 = axis.product (%f6 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f8 = axis.factor %ax1 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %f9 = axis.factor %ax2 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %f10 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f11 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs12 = axis.product (%f8 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %rhs13 = axis.product (%f9 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %lhs14 = axis.product (%f10 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %rhs15 = axis.product (%f11 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map16 = axis.map %lhs12, %lhs14 to %rhs13, %rhs15 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]
  %h17 = distributed.Collective %in : tensor<4xf32> on %mesh5 : !axis.factor_group<2> to tensor<4xf32> on %mesh7 : !axis.factor_group<2> reduces () maps %map16 : !axis.map
  %v18 = distributed.Await %h17 : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
  %f19 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh20 = axis.product (%f19 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f21 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh22 = axis.product (%f21 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f23 = axis.factor %ax2 : !axis.shape_axis<tensor<4xf32>, 0> <2, 2>
  %red24 = axis.product (%f23 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 2, 2>)
  %f25 = axis.factor %ax2 : !axis.shape_axis<tensor<4xf32>, 0> <2, 1>
  %f26 = axis.factor %ax3 : !axis.shape_axis<tensor<2xf32>, 0> <2, 1>
  %f27 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f28 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs29 = axis.product (%f25 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 2, 1>)
  %rhs30 = axis.product (%f26 : !axis.axis_factor<!axis.shape_axis<tensor<2xf32>, 0>, 2, 1>)
  %lhs31 = axis.product (%f27 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %rhs32 = axis.product (%f28 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map33 = axis.map %lhs29, %lhs31 to %rhs30, %rhs32 : [!axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<2>, !axis.factor_group<2>]
  %h34 = distributed.Collective %v18 : tensor<4xf32> on %mesh20 : !axis.factor_group<2> to tensor<2xf32> on %mesh22 : !axis.factor_group<2> reduces (%red24 : !axis.factor_group<2>) maps %map33 : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }
  %v35 = distributed.Await %h34 : !distributed.asynch_handle<tensor<2xf32>> -> tensor<2xf32>
}

// -----

// Not in explicit-replication form: the first collective only mentions mesh axis 0 and the second only axis 1, so the mesh axes the two touch differ. Fusing
// would need identity pairs for the missing axes; the pass declines. After distributed-make-replications-explicit both would mention every axis.
// IR-LABEL: module @different_mesh_axes
// IR: distributed.Collective
// IR: distributed.Collective
// IR-NOT: distributed.Collective
// REMARK: remark: fuse-collectives: not fused with the preceding collective (the collectives touch different mesh axes)
module @different_mesh_axes {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>

  %ax1 = axis.getaxis tensor<4xf32> 0
  %ax2 = axis.getaxis tensor<4xf32> 0
  %ax3 = axis.getaxis tensor<4xf32> 0
  %in = tensor.empty() : tensor<4xf32>
  %f4 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %mesh5 = axis.product (%f4 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f6 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %mesh7 = axis.product (%f6 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f8 = axis.factor %ax1 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %f9 = axis.factor %ax2 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %f10 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f11 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %lhs12 = axis.product (%f8 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %rhs13 = axis.product (%f9 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %lhs14 = axis.product (%f10 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %rhs15 = axis.product (%f11 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %map16 = axis.map %lhs12, %lhs14 to %rhs13, %rhs15 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]
  %h17 = distributed.Collective %in : tensor<4xf32> on %mesh5 : !axis.factor_group<2> to tensor<4xf32> on %mesh7 : !axis.factor_group<2> reduces () maps %map16 : !axis.map
  %v18 = distributed.Await %h17 : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
  %f19 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh20 = axis.product (%f19 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f21 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh22 = axis.product (%f21 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f23 = axis.factor %ax2 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %f24 = axis.factor %ax3 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %f25 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f26 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs27 = axis.product (%f23 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %rhs28 = axis.product (%f24 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %lhs29 = axis.product (%f25 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %rhs30 = axis.product (%f26 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map31 = axis.map %lhs27, %lhs29 to %rhs28, %rhs30 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]
  %h32 = distributed.Collective %v18 : tensor<4xf32> on %mesh20 : !axis.factor_group<2> to tensor<4xf32> on %mesh22 : !axis.factor_group<2> reduces () maps %map31 : !axis.map
  %v33 = distributed.Await %h32 : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
}
