// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-lower-collectives %s | FileCheck %s

// distributed-lower-collectives runs a collective's chain on its tile in
// digit shape, one dimension per local digit. Each step slices, scatters,
// gathers or exchanges the digit the collective's map names, on that digit's
// own dimension, and a gathered digit lands where the output tile needs it.
// So no step moves a digit before it acts, and only a device-local
// relabeling ends in a transpose. The RUN line atomizes the input first, since
// planning requires an atomic collective.

// A local slice of the column digit of a 4x16 tile: the slice cuts
// dimension 1 of the 4x4x4 digit view, keeping every row.
// CHECK-LABEL: module @slice_column_digit {
// CHECK: %[[DIGITS:.+]] = stablehlo.reshape %{{.+}} : (tensor<4x16xf32>) -> tensor<4x4x4xf32>
// CHECK: %[[SLICED:.+]] = stablehlo.dynamic_slice %[[DIGITS]], [[ZERO:%[^,]+]], {{%[^,]+}}, [[ZERO]], sizes = [4, 1, 4]
// CHECK: stablehlo.reshape %[[SLICED]] : (tensor<4x1x4xf32>) -> tensor<4x4xf32>
// CHECK-NOT: stablehlo.transpose
module @slice_column_digit {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 1>
  %rep4 = distributed.ReplicationAxis 4 : !distributed.replication_axis<4>
  %ti0 = axis.getaxis tensor<4x16xf32> 0
  %ti1 = axis.getaxis tensor<4x16xf32> 1
  %to0 = axis.getaxis tensor<4x4xf32> 0
  %to1 = axis.getaxis tensor<4x4xf32> 1
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %mesh_out = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %lhs0 = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f4 = axis.factor %rep4 : !distributed.replication_axis<4> <4, 1>
  %rhs0 = axis.product (%f4 : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
  %f5 = axis.factor %ti0 : !axis.shape_axis<tensor<4x16xf32>, 0> <4, 1>
  %lhs1 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<4x16xf32>, 0>, 4, 1>)
  %f6 = axis.factor %to0 : !axis.shape_axis<tensor<4x4xf32>, 0> <4, 1>
  %rhs1 = axis.product (%f6 : !axis.axis_factor<!axis.shape_axis<tensor<4x4xf32>, 0>, 4, 1>)
  %f7 = axis.factor %ti1 : !axis.shape_axis<tensor<4x16xf32>, 1> <16, 1>
  %lhs2 = axis.product (%f7 : !axis.axis_factor<!axis.shape_axis<tensor<4x16xf32>, 1>, 16, 1>)
  %f8 = axis.factor %p0 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %f9 = axis.factor %to1 : !axis.shape_axis<tensor<4x4xf32>, 1> <4, 1>
  %rhs2 = axis.product (%f8 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>, %f9 : !axis.axis_factor<!axis.shape_axis<tensor<4x4xf32>, 1>, 4, 1>)
  %map = axis.map %lhs0, %lhs1, %lhs2 to %rhs0, %rhs1, %rhs2 : [!axis.factor_group<4>, !axis.factor_group<4>, !axis.factor_group<16>] [!axis.factor_group<4>, !axis.factor_group<4>, !axis.factor_group<16>]

  %in = tensor.empty() : tensor<4x16xf32>
  %h = distributed.Collective %in : tensor<4x16xf32> on %mesh_in : !axis.factor_group<4> to tensor<4x4xf32> on %mesh_out : !axis.factor_group<4> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4x4xf32>> -> tensor<4x4xf32>
}

// -----

// An all-to-all that moves a mesh digit from the rows to the columns: it
// splits the column's major digit (dimension 2 of the view) onto the mesh
// atom and concatenates the mesh digit at dimension 0, the output rows' major
// digit.
// CHECK-LABEL: module @all_to_all_column_digit {
// CHECK: %[[DIGITS:.+]] = stablehlo.reshape %{{.+}} : (tensor<4x4xf32>) -> tensor<1x4x2x2xf32>
// CHECK: %[[EXCHANGED:.+]] = "stablehlo.all_to_all"(%[[DIGITS]])
// CHECK-SAME: concat_dimension = 0
// CHECK-SAME: split_dimension = 2
// CHECK-SAME: (tensor<1x4x2x2xf32>) -> tensor<2x4x1x2xf32>
// CHECK: stablehlo.reshape %[[EXCHANGED]] : (tensor<2x4x1x2xf32>) -> tensor<8x2xf32>
// CHECK-NOT: stablehlo.transpose
module @all_to_all_column_digit {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %ti0 = axis.getaxis tensor<4x4xf32> 0
  %ti1 = axis.getaxis tensor<4x4xf32> 1
  %to0 = axis.getaxis tensor<8x2xf32> 0
  %to1 = axis.getaxis tensor<8x2xf32> 1
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f4 = axis.factor %ti0 : !axis.shape_axis<tensor<4x4xf32>, 0> <4, 1>
  %lhs0 = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %f4 : !axis.axis_factor<!axis.shape_axis<tensor<4x4xf32>, 0>, 4, 1>)
  %f5 = axis.factor %to0 : !axis.shape_axis<tensor<8x2xf32>, 0> <8, 1>
  %rhs0 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<8x2xf32>, 0>, 8, 1>)
  %f6 = axis.factor %ti1 : !axis.shape_axis<tensor<4x4xf32>, 1> <4, 1>
  %lhs1 = axis.product (%f6 : !axis.axis_factor<!axis.shape_axis<tensor<4x4xf32>, 1>, 4, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f8 = axis.factor %to1 : !axis.shape_axis<tensor<8x2xf32>, 1> <2, 1>
  %rhs1 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %f8 : !axis.axis_factor<!axis.shape_axis<tensor<8x2xf32>, 1>, 2, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<8>, !axis.factor_group<4>] [!axis.factor_group<8>, !axis.factor_group<4>]

  %in = tensor.empty() : tensor<4x4xf32>
  %h = distributed.Collective %in : tensor<4x4xf32> on %mesh_in : !axis.factor_group<2> to tensor<8x2xf32> on %mesh_out : !axis.factor_group<2> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<8x2xf32>> -> tensor<8x2xf32>
}

// -----

// A reduce-scatter onto the column's major digit: it scatters dimension 1 of
// the 4x2x2 view.
// CHECK-LABEL: module @reduce_scatter_column_digit {
// CHECK: %[[DIGITS:.+]] = stablehlo.reshape %{{.+}} : (tensor<4x4xf32>) -> tensor<4x2x2xf32>
// CHECK: %[[SCATTERED:.+]] = "stablehlo.reduce_scatter"(%[[DIGITS]])
// CHECK-SAME: scatter_dimension = 1
// CHECK: }) {{.*}}: (tensor<4x2x2xf32>) -> tensor<4x1x2xf32>
// CHECK: stablehlo.reshape %[[SCATTERED]] : (tensor<4x1x2xf32>) -> tensor<4x2xf32>
// CHECK-NOT: stablehlo.transpose
module @reduce_scatter_column_digit {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %ti0 = axis.getaxis tensor<4x4xf32> 0
  %ti1 = axis.getaxis tensor<4x4xf32> 1
  %to0 = axis.getaxis tensor<4x2xf32> 0
  %to1 = axis.getaxis tensor<4x2xf32> 1
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %red0 = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f4 = axis.factor %ti0 : !axis.shape_axis<tensor<4x4xf32>, 0> <4, 1>
  %lhs0 = axis.product (%f4 : !axis.axis_factor<!axis.shape_axis<tensor<4x4xf32>, 0>, 4, 1>)
  %f5 = axis.factor %to0 : !axis.shape_axis<tensor<4x2xf32>, 0> <4, 1>
  %rhs0 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<4x2xf32>, 0>, 4, 1>)
  %f6 = axis.factor %ti1 : !axis.shape_axis<tensor<4x4xf32>, 1> <4, 1>
  %lhs1 = axis.product (%f6 : !axis.axis_factor<!axis.shape_axis<tensor<4x4xf32>, 1>, 4, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f8 = axis.factor %to1 : !axis.shape_axis<tensor<4x2xf32>, 1> <2, 1>
  %rhs1 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %f8 : !axis.axis_factor<!axis.shape_axis<tensor<4x2xf32>, 1>, 2, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<4>, !axis.factor_group<4>] [!axis.factor_group<4>, !axis.factor_group<4>]

  %in = tensor.empty() : tensor<4x4xf32>
  %h = distributed.Collective %in : tensor<4x4xf32> on %mesh_in : !axis.factor_group<2> to tensor<4x2xf32> on %mesh_out : !axis.factor_group<2> reduces (%red0 : !axis.factor_group<2>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4x2xf32>> -> tensor<4x2xf32>
}

// -----

// Two mesh atoms gathered into one output dimension whose index is
// 2 * p0 + p1. The chain gathers p0 first (groups [[0, 2], [1, 3]]); p1,
// gathered second, goes minor to it at dimension 1.
// CHECK-LABEL: module @gather_two_atoms_p0_major {
// CHECK: "stablehlo.all_gather"
// CHECK-SAME: all_gather_dim = 0
// CHECK-SAME: replica_groups = dense<{{\[}}[0, 2], [1, 3]]>
// CHECK-SAME: (tensor<1xf32>) -> tensor<2xf32>
// CHECK: %[[VIEW:.+]] = stablehlo.reshape %{{.+}} : (tensor<2xf32>) -> tensor<2x1xf32>
// CHECK: %[[GATHERED:.+]] = "stablehlo.all_gather"(%[[VIEW]])
// CHECK-SAME: all_gather_dim = 1
// CHECK-SAME: replica_groups = dense<{{\[}}[0, 1], [2, 3]]>
// CHECK: stablehlo.reshape %[[GATHERED]] : (tensor<2x2xf32>) -> tensor<4xf32>
// CHECK-NOT: stablehlo.transpose
module @gather_two_atoms_p0_major {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %rep_a = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %rep_b = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %to0 = axis.getaxis tensor<4xf32> 0
  %m0_in = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %m1_in = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%m0_in : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %m1_in : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %m0_out = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %m1_out = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%m0_out : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %m1_out : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %t0 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %t1 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs0 = axis.product (%t0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %t1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f5 = axis.factor %to0 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %rhs0 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %ra = axis.factor %rep_a : !distributed.replication_axis<2> <2, 1>
  %lhs1 = axis.product (%ra : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %r0 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %rhs1 = axis.product (%r0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %rb = axis.factor %rep_b : !distributed.replication_axis<2> <2, 1>
  %lhs2 = axis.product (%rb : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %r1 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs2 = axis.product (%r1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map = axis.map %lhs0, %lhs1, %lhs2 to %rhs0, %rhs1, %rhs2 : [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<1xf32>
  %h = distributed.Collective %in : tensor<1xf32> on %mesh_in : !axis.factor_group<4> to tensor<4xf32> on %mesh_out : !axis.factor_group<4> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
}

// -----

// The same chain for an output index of 2 * p1 + p0: p1, gathered second,
// goes major to p0 at dimension 0.
// CHECK-LABEL: module @gather_two_atoms_p1_major {
// CHECK: "stablehlo.all_gather"
// CHECK-SAME: all_gather_dim = 0
// CHECK-SAME: replica_groups = dense<{{\[}}[0, 2], [1, 3]]>
// CHECK: %[[VIEW:.+]] = stablehlo.reshape %{{.+}} : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK: %[[GATHERED:.+]] = "stablehlo.all_gather"(%[[VIEW]])
// CHECK-SAME: all_gather_dim = 0
// CHECK-SAME: replica_groups = dense<{{\[}}[0, 1], [2, 3]]>
// CHECK: stablehlo.reshape %[[GATHERED]] : (tensor<2x2xf32>) -> tensor<4xf32>
// CHECK-NOT: stablehlo.transpose
module @gather_two_atoms_p1_major {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %rep_a = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %rep_b = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %to0 = axis.getaxis tensor<4xf32> 0
  %m0_in = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %m1_in = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%m0_in : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %m1_in : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %m0_out = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %m1_out = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%m0_out : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %m1_out : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %t0 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %t1 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs0 = axis.product (%t1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %t0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f5 = axis.factor %to0 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %rhs0 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %ra = axis.factor %rep_a : !distributed.replication_axis<2> <2, 1>
  %lhs1 = axis.product (%ra : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %r0 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %rhs1 = axis.product (%r0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %rb = axis.factor %rep_b : !distributed.replication_axis<2> <2, 1>
  %lhs2 = axis.product (%rb : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %r1 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs2 = axis.product (%r1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map = axis.map %lhs0, %lhs1, %lhs2 to %rhs0, %rhs1, %rhs2 : [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<1xf32>
  %h = distributed.Collective %in : tensor<1xf32> on %mesh_in : !axis.factor_group<4> to tensor<4xf32> on %mesh_out : !axis.factor_group<4> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
}

// -----

// One mesh atom gathered minor to the input tile digit: output index
// 2 * t + p0, so the gather inserts the mesh digit at dimension 1.
// CHECK-LABEL: module @gather_mesh_digit_below_tile_digit {
// CHECK: %[[VIEW:.+]] = stablehlo.reshape %{{.+}} : (tensor<2xf32>) -> tensor<2x1xf32>
// CHECK: %[[GATHERED:.+]] = "stablehlo.all_gather"(%[[VIEW]])
// CHECK-SAME: all_gather_dim = 1
// CHECK: stablehlo.reshape %[[GATHERED]] : (tensor<2x2xf32>) -> tensor<4xf32>
// CHECK-NOT: stablehlo.transpose
module @gather_mesh_digit_below_tile_digit {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<2xf32> 0
  %to0 = axis.getaxis tensor<4xf32> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f4 = axis.factor %ti0 : !axis.shape_axis<tensor<2xf32>, 0> <2, 1>
  %lhs0 = axis.product (%f4 : !axis.axis_factor<!axis.shape_axis<tensor<2xf32>, 0>, 2, 1>, %f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f5 = axis.factor %to0 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %rhs0 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %f6 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %lhs1 = axis.product (%f6 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs1 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<2xf32>
  %h = distributed.Collective %in : tensor<2xf32> on %mesh_in : !axis.factor_group<2> to tensor<4xf32> on %mesh_out : !axis.factor_group<2> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
}

// -----

// A peel of the fast atom p0 around reduce-scatters on p1 and p2. Its
// scatter shards the 512-element digit, the only one that stays local until
// the peel's gather. The two digits that p1 and p2 place are also divisible by
// the atom's extent, but they leave the tile before the gather. The gather
// restores the same dimension.
// CHECK-LABEL: module @peel_shards_digit_that_stays_local {
// CHECK: %[[DIGITS:.+]] = stablehlo.reshape %{{.+}} : (tensor<2048xi8>) -> tensor<2x2x512xi8>
// CHECK: "stablehlo.reduce_scatter"(%[[DIGITS]])
// CHECK-SAME: replica_groups = dense<{{\[}}[0, 4], [1, 5], [2, 6], [3, 7]]>
// CHECK-SAME: scatter_dimension = 2
// CHECK: }) {{.*}}: (tensor<2x2x512xi8>) -> tensor<2x2x256xi8>
// CHECK: "stablehlo.reduce_scatter"
// CHECK-SAME: scatter_dimension = 0
// CHECK: "stablehlo.reduce_scatter"
// CHECK-SAME: scatter_dimension = 0
// CHECK: "stablehlo.all_gather"
// CHECK-SAME: all_gather_dim = 0
// CHECK-SAME: replica_groups = dense<{{\[}}[0, 4], [1, 5], [2, 6], [3, 7]]>
// CHECK-SAME: (tensor<256xi8>) -> tensor<512xi8>
module @peel_shards_digit_that_stays_local {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>] {axis_bandwidths = [8.0e0, 1.0e-1, 1.0e-1]}

  func.func @main() {
    return
  }

  %p0, %p1, %p2 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<2048xi8> 0
  %to0 = axis.getaxis tensor<512xi8> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f3 = axis.factor %p2 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f4 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %f5 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f6 = axis.factor %p2 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f4 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>, %f5 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f6 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %f8 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f9 = axis.factor %p2 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %red0 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>, %f8 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f9 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f10 = axis.factor %ti0 : !axis.shape_axis<tensor<2048xi8>, 0> <2048, 1>
  %lhs0 = axis.product (%f10 : !axis.axis_factor<!axis.shape_axis<tensor<2048xi8>, 0>, 2048, 1>)
  %f11 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f12 = axis.factor %p2 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f13 = axis.factor %to0 : !axis.shape_axis<tensor<512xi8>, 0> <512, 1>
  %rhs0 = axis.product (%f11 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f12 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %f13 : !axis.axis_factor<!axis.shape_axis<tensor<512xi8>, 0>, 512, 1>)
  %f14 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %lhs1 = axis.product (%f14 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f15 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %rhs1 = axis.product (%f15 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<2048>, !axis.factor_group<2>] [!axis.factor_group<2048>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<2048xi8>
  %h = distributed.Collective %in : tensor<2048xi8> on %mesh_in : !axis.factor_group<8> to tensor<512xi8> on %mesh_out : !axis.factor_group<8> reduces (%red0 : !axis.factor_group<8>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<i8>, %rhs_v: tensor<i8>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<i8>
    stablehlo.return %r : tensor<i8>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<512xi8>> -> tensor<512xi8>
}

// -----

// A device-local relabeling that swaps the tile's two digits has an empty
// chain, and lowers to the one transpose into the output's order.
// CHECK-LABEL: module @relabel_transposes {
// CHECK: %[[IN:.+]] = tensor.empty() : tensor<2x3xf32>
// CHECK: stablehlo.transpose %[[IN]], dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
// CHECK-NOT: distributed.Collective
module @relabel_transposes {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %ti0 = axis.getaxis tensor<2x3xf32> 0
  %ti1 = axis.getaxis tensor<2x3xf32> 1
  %to0 = axis.getaxis tensor<3x2xf32> 0
  %to1 = axis.getaxis tensor<3x2xf32> 1
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %a0 = axis.factor %ti0 : !axis.shape_axis<tensor<2x3xf32>, 0> <2, 1>
  %lhs0 = axis.product (%a0 : !axis.axis_factor<!axis.shape_axis<tensor<2x3xf32>, 0>, 2, 1>)
  %b1 = axis.factor %to1 : !axis.shape_axis<tensor<3x2xf32>, 1> <2, 1>
  %rhs0 = axis.product (%b1 : !axis.axis_factor<!axis.shape_axis<tensor<3x2xf32>, 1>, 2, 1>)
  %a1 = axis.factor %ti1 : !axis.shape_axis<tensor<2x3xf32>, 1> <3, 1>
  %lhs1 = axis.product (%a1 : !axis.axis_factor<!axis.shape_axis<tensor<2x3xf32>, 1>, 3, 1>)
  %b0 = axis.factor %to0 : !axis.shape_axis<tensor<3x2xf32>, 0> <3, 1>
  %rhs1 = axis.product (%b0 : !axis.axis_factor<!axis.shape_axis<tensor<3x2xf32>, 0>, 3, 1>)
  %m0 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs2 = axis.product (%m0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %m1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs2 = axis.product (%m1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map = axis.map %lhs0, %lhs1, %lhs2 to %rhs0, %rhs1, %rhs2 : [!axis.factor_group<2>, !axis.factor_group<3>, !axis.factor_group<2>] [!axis.factor_group<2>, !axis.factor_group<3>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<2x3xf32>
  %h = distributed.Collective %in : tensor<2x3xf32> on %mesh_in : !axis.factor_group<2> to tensor<3x2xf32> on %mesh_out : !axis.factor_group<2> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<3x2xf32>> -> tensor<3x2xf32>
}
