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

// A peel of the extent-4 atom p0 around a reduce-scatter on p1, over a 2x2x2
// tile. No digit has extent 4, so the peel's shard merges the two digits that
// stay local (dimensions 1 and 2, already adjacent) into one dimension, and its
// gather restores both.
// CHECK-LABEL: module @peel_merges_adjacent_digits {
// CHECK: %[[MERGED:.+]] = stablehlo.reshape %{{.+}} : (tensor<2x2x2xi8>) -> tensor<2x4xi8>
// CHECK: "stablehlo.reduce_scatter"(%[[MERGED]])
// CHECK-SAME: replica_groups = dense<{{\[}}[0, 2, 4, 6], [1, 3, 5, 7]]>
// CHECK-SAME: scatter_dimension = 1
// CHECK: }) {{.*}}: (tensor<2x4xi8>) -> tensor<2x1xi8>
// CHECK: "stablehlo.reduce_scatter"
// CHECK-SAME: scatter_dimension = 0
// CHECK: %[[GATHERED:.+]] = "stablehlo.all_gather"
// CHECK-SAME: all_gather_dim = 0
// CHECK-SAME: replica_groups = dense<{{\[}}[0, 2, 4, 6], [1, 3, 5, 7]]>
// CHECK-SAME: (tensor<1xi8>) -> tensor<4xi8>
// CHECK: stablehlo.reshape %[[GATHERED]] : (tensor<4xi8>) -> tensor<2x2xi8>
// CHECK-NOT: stablehlo.transpose
module @peel_merges_adjacent_digits {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 2>, !distributed.physical_comm_axis<2, 1>] {axis_bandwidths = [8.0, 0.1]}

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 2>, !distributed.physical_comm_axis<2, 1>
  %rep4 = distributed.ReplicationAxis 4 : !distributed.replication_axis<4>
  %ti0 = axis.getaxis tensor<2x2x2xi8> 0
  %ti1 = axis.getaxis tensor<2x2x2xi8> 1
  %ti2 = axis.getaxis tensor<2x2x2xi8> 2
  %to0 = axis.getaxis tensor<2x2xi8> 0
  %to1 = axis.getaxis tensor<2x2xi8> 1
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<4, 2> <4, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f3 = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 4, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f4 = axis.factor %p0 : !distributed.physical_comm_axis<4, 2> <4, 1>
  %f5 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f6 = axis.product (%f4 : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 4, 1>, %f5 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<4, 2> <4, 1>
  %f8 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f9 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 4, 1>, %f8 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f10 = axis.factor %ti0 : !axis.shape_axis<tensor<2x2x2xi8>, 0> <2, 1>
  %f11 = axis.product (%f10 : !axis.axis_factor<!axis.shape_axis<tensor<2x2x2xi8>, 0>, 2, 1>)
  %f12 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f13 = axis.product (%f12 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f14 = axis.factor %ti1 : !axis.shape_axis<tensor<2x2x2xi8>, 1> <2, 1>
  %f15 = axis.product (%f14 : !axis.axis_factor<!axis.shape_axis<tensor<2x2x2xi8>, 1>, 2, 1>)
  %f16 = axis.factor %to0 : !axis.shape_axis<tensor<2x2xi8>, 0> <2, 1>
  %f17 = axis.product (%f16 : !axis.axis_factor<!axis.shape_axis<tensor<2x2xi8>, 0>, 2, 1>)
  %f18 = axis.factor %ti2 : !axis.shape_axis<tensor<2x2x2xi8>, 2> <2, 1>
  %f19 = axis.product (%f18 : !axis.axis_factor<!axis.shape_axis<tensor<2x2x2xi8>, 2>, 2, 1>)
  %f20 = axis.factor %to1 : !axis.shape_axis<tensor<2x2xi8>, 1> <2, 1>
  %f21 = axis.product (%f20 : !axis.axis_factor<!axis.shape_axis<tensor<2x2xi8>, 1>, 2, 1>)
  %f22 = axis.factor %rep4 : !distributed.replication_axis<4> <4, 1>
  %f23 = axis.product (%f22 : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
  %f24 = axis.factor %p0 : !distributed.physical_comm_axis<4, 2> <4, 1>
  %f25 = axis.product (%f24 : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 4, 1>)
  %map = axis.map %f11, %f15, %f19, %f23 to %f13, %f17, %f21, %f25 : [!axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<4>] [!axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<4>]

  %in = tensor.empty() : tensor<2x2x2xi8>
  %h = distributed.Collective %in : tensor<2x2x2xi8> on %f3 : !axis.factor_group<8> to tensor<2x2xi8> on %f6 : !axis.factor_group<8> reduces (%f9 : !axis.factor_group<8>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<i8>, %rhs_v: tensor<i8>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<i8>
    stablehlo.return %r : tensor<i8>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<2x2xi8>> -> tensor<2x2xi8>
}

// -----

// A peel of extent 4 whose only free digit has extent 8: the scatter takes
// the digit's major factor of 4 and leaves the minor 2 in place.
// CHECK-LABEL: module @peel_splits_one_digit {
// CHECK: "stablehlo.reduce_scatter"
// CHECK-SAME: replica_groups = dense<{{\[}}[0, 2, 4, 6], [1, 3, 5, 7]]>
// CHECK-SAME: scatter_dimension = 1
// CHECK: }) {{.*}}: (tensor<2x8xi8>) -> tensor<2x2xi8>
// CHECK: "stablehlo.reduce_scatter"
// CHECK-SAME: scatter_dimension = 0
// CHECK: "stablehlo.all_gather"
// CHECK-SAME: all_gather_dim = 0
// CHECK-SAME: replica_groups = dense<{{\[}}[0, 2, 4, 6], [1, 3, 5, 7]]>
// CHECK-SAME: (tensor<2xi8>) -> tensor<8xi8>
// CHECK-NOT: stablehlo.transpose
module @peel_splits_one_digit {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 2>, !distributed.physical_comm_axis<2, 1>] {axis_bandwidths = [8.0, 0.1]}

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 2>, !distributed.physical_comm_axis<2, 1>
  %rep4 = distributed.ReplicationAxis 4 : !distributed.replication_axis<4>
  %ti0 = axis.getaxis tensor<2x8xi8> 0
  %ti1 = axis.getaxis tensor<2x8xi8> 1
  %to0 = axis.getaxis tensor<8xi8> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<4, 2> <4, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f3 = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 4, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f4 = axis.factor %p0 : !distributed.physical_comm_axis<4, 2> <4, 1>
  %f5 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f6 = axis.product (%f4 : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 4, 1>, %f5 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<4, 2> <4, 1>
  %f8 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f9 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 4, 1>, %f8 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f10 = axis.factor %ti0 : !axis.shape_axis<tensor<2x8xi8>, 0> <2, 1>
  %f11 = axis.product (%f10 : !axis.axis_factor<!axis.shape_axis<tensor<2x8xi8>, 0>, 2, 1>)
  %f12 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f13 = axis.product (%f12 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f14 = axis.factor %ti1 : !axis.shape_axis<tensor<2x8xi8>, 1> <8, 1>
  %f15 = axis.product (%f14 : !axis.axis_factor<!axis.shape_axis<tensor<2x8xi8>, 1>, 8, 1>)
  %f16 = axis.factor %to0 : !axis.shape_axis<tensor<8xi8>, 0> <8, 1>
  %f17 = axis.product (%f16 : !axis.axis_factor<!axis.shape_axis<tensor<8xi8>, 0>, 8, 1>)
  %f18 = axis.factor %rep4 : !distributed.replication_axis<4> <4, 1>
  %f19 = axis.product (%f18 : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
  %f20 = axis.factor %p0 : !distributed.physical_comm_axis<4, 2> <4, 1>
  %f21 = axis.product (%f20 : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 4, 1>)
  %map = axis.map %f11, %f15, %f19 to %f13, %f17, %f21 : [!axis.factor_group<2>, !axis.factor_group<8>, !axis.factor_group<4>] [!axis.factor_group<2>, !axis.factor_group<8>, !axis.factor_group<4>]

  %in = tensor.empty() : tensor<2x8xi8>
  %h = distributed.Collective %in : tensor<2x8xi8> on %f3 : !axis.factor_group<8> to tensor<8xi8> on %f6 : !axis.factor_group<8> reduces (%f9 : !axis.factor_group<8>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<i8>, %rhs_v: tensor<i8>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<i8>
    stablehlo.return %r : tensor<i8>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<8xi8>> -> tensor<8xi8>
}

// -----

// A peel of extent 4 over a 2x3x2x2 tile prefers the two adjacent extent-2
// digits (dimensions 2 and 3) over the first ones it could combine, so it
// merges them without a transpose.
// CHECK-LABEL: module @peel_prefers_adjacent_digits {
// CHECK: %[[MERGED:.+]] = stablehlo.reshape %{{.+}} : (tensor<2x3x2x2xi8>) -> tensor<2x3x4xi8>
// CHECK: "stablehlo.reduce_scatter"(%[[MERGED]])
// CHECK-SAME: scatter_dimension = 2
// CHECK: }) {{.*}}: (tensor<2x3x4xi8>) -> tensor<2x3x1xi8>
// CHECK: "stablehlo.all_reduce"
// CHECK: %[[GATHERED:.+]] = "stablehlo.all_gather"
// CHECK-SAME: all_gather_dim = 2
// CHECK-SAME: (tensor<2x3x1xi8>) -> tensor<2x3x4xi8>
// CHECK: stablehlo.reshape %[[GATHERED]] : (tensor<2x3x4xi8>) -> tensor<2x3x2x2xi8>
// CHECK-NOT: stablehlo.transpose
module @peel_prefers_adjacent_digits {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 2>, !distributed.physical_comm_axis<2, 1>] {axis_bandwidths = [8.0, 0.1]}

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 2>, !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %rep4 = distributed.ReplicationAxis 4 : !distributed.replication_axis<4>
  %ti0 = axis.getaxis tensor<2x3x2x2xi8> 0
  %ti1 = axis.getaxis tensor<2x3x2x2xi8> 1
  %ti2 = axis.getaxis tensor<2x3x2x2xi8> 2
  %ti3 = axis.getaxis tensor<2x3x2x2xi8> 3
  %to0 = axis.getaxis tensor<2x3x2x2xi8> 0
  %to1 = axis.getaxis tensor<2x3x2x2xi8> 1
  %to2 = axis.getaxis tensor<2x3x2x2xi8> 2
  %to3 = axis.getaxis tensor<2x3x2x2xi8> 3
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<4, 2> <4, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f3 = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 4, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f4 = axis.factor %p0 : !distributed.physical_comm_axis<4, 2> <4, 1>
  %f5 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f6 = axis.product (%f4 : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 4, 1>, %f5 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<4, 2> <4, 1>
  %f8 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f9 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 4, 1>, %f8 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f10 = axis.factor %ti0 : !axis.shape_axis<tensor<2x3x2x2xi8>, 0> <2, 1>
  %f11 = axis.product (%f10 : !axis.axis_factor<!axis.shape_axis<tensor<2x3x2x2xi8>, 0>, 2, 1>)
  %f12 = axis.factor %to0 : !axis.shape_axis<tensor<2x3x2x2xi8>, 0> <2, 1>
  %f13 = axis.product (%f12 : !axis.axis_factor<!axis.shape_axis<tensor<2x3x2x2xi8>, 0>, 2, 1>)
  %f14 = axis.factor %ti1 : !axis.shape_axis<tensor<2x3x2x2xi8>, 1> <3, 1>
  %f15 = axis.product (%f14 : !axis.axis_factor<!axis.shape_axis<tensor<2x3x2x2xi8>, 1>, 3, 1>)
  %f16 = axis.factor %to1 : !axis.shape_axis<tensor<2x3x2x2xi8>, 1> <3, 1>
  %f17 = axis.product (%f16 : !axis.axis_factor<!axis.shape_axis<tensor<2x3x2x2xi8>, 1>, 3, 1>)
  %f18 = axis.factor %ti2 : !axis.shape_axis<tensor<2x3x2x2xi8>, 2> <2, 1>
  %f19 = axis.product (%f18 : !axis.axis_factor<!axis.shape_axis<tensor<2x3x2x2xi8>, 2>, 2, 1>)
  %f20 = axis.factor %to2 : !axis.shape_axis<tensor<2x3x2x2xi8>, 2> <2, 1>
  %f21 = axis.product (%f20 : !axis.axis_factor<!axis.shape_axis<tensor<2x3x2x2xi8>, 2>, 2, 1>)
  %f22 = axis.factor %ti3 : !axis.shape_axis<tensor<2x3x2x2xi8>, 3> <2, 1>
  %f23 = axis.product (%f22 : !axis.axis_factor<!axis.shape_axis<tensor<2x3x2x2xi8>, 3>, 2, 1>)
  %f24 = axis.factor %to3 : !axis.shape_axis<tensor<2x3x2x2xi8>, 3> <2, 1>
  %f25 = axis.product (%f24 : !axis.axis_factor<!axis.shape_axis<tensor<2x3x2x2xi8>, 3>, 2, 1>)
  %f26 = axis.factor %rep4 : !distributed.replication_axis<4> <4, 1>
  %f27 = axis.product (%f26 : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
  %f28 = axis.factor %p0 : !distributed.physical_comm_axis<4, 2> <4, 1>
  %f29 = axis.product (%f28 : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 4, 1>)
  %f30 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %f31 = axis.product (%f30 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f32 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f33 = axis.product (%f32 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map = axis.map %f11, %f15, %f19, %f23, %f27, %f31 to %f13, %f17, %f21, %f25, %f29, %f33 : [!axis.factor_group<2>, !axis.factor_group<3>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<2>, !axis.factor_group<3>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<2x3x2x2xi8>
  %h = distributed.Collective %in : tensor<2x3x2x2xi8> on %f3 : !axis.factor_group<8> to tensor<2x3x2x2xi8> on %f6 : !axis.factor_group<8> reduces (%f9 : !axis.factor_group<8>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<i8>, %rhs_v: tensor<i8>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<i8>
    stablehlo.return %r : tensor<i8>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<2x3x2x2xi8>> -> tensor<2x3x2x2xi8>
}

// -----

// A peel of extent 4 over a 2x3x5x2 tile takes the two extent-2 digits, which
// the 3 and the 5 separate. A transpose brings them together before the merge,
// and the inverse transpose after the gather puts them back.
// CHECK-LABEL: module @peel_transposes_separated_factors {
// CHECK: %[[TOGETHER:.+]] = stablehlo.transpose %{{.+}}, dims = [1, 2, 0, 3] : (tensor<2x3x5x2xi8>) -> tensor<3x5x2x2xi8>
// CHECK: %[[MERGED:.+]] = stablehlo.reshape %[[TOGETHER]] : (tensor<3x5x2x2xi8>) -> tensor<3x5x4xi8>
// CHECK: "stablehlo.reduce_scatter"(%[[MERGED]])
// CHECK-SAME: scatter_dimension = 2
// CHECK: }) {{.*}}: (tensor<3x5x4xi8>) -> tensor<3x5x1xi8>
// CHECK: "stablehlo.all_reduce"
// CHECK: %[[GATHERED:.+]] = "stablehlo.all_gather"
// CHECK-SAME: all_gather_dim = 2
// CHECK-SAME: (tensor<3x5x1xi8>) -> tensor<3x5x4xi8>
// CHECK: %[[SPLIT:.+]] = stablehlo.reshape %[[GATHERED]] : (tensor<3x5x4xi8>) -> tensor<3x5x2x2xi8>
// CHECK: stablehlo.transpose %[[SPLIT]], dims = [2, 0, 1, 3] : (tensor<3x5x2x2xi8>) -> tensor<2x3x5x2xi8>
module @peel_transposes_separated_factors {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 2>, !distributed.physical_comm_axis<2, 1>] {axis_bandwidths = [8.0, 0.1]}

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 2>, !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %rep4 = distributed.ReplicationAxis 4 : !distributed.replication_axis<4>
  %ti0 = axis.getaxis tensor<2x3x5x2xi8> 0
  %ti1 = axis.getaxis tensor<2x3x5x2xi8> 1
  %ti2 = axis.getaxis tensor<2x3x5x2xi8> 2
  %ti3 = axis.getaxis tensor<2x3x5x2xi8> 3
  %to0 = axis.getaxis tensor<2x3x5x2xi8> 0
  %to1 = axis.getaxis tensor<2x3x5x2xi8> 1
  %to2 = axis.getaxis tensor<2x3x5x2xi8> 2
  %to3 = axis.getaxis tensor<2x3x5x2xi8> 3
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<4, 2> <4, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f3 = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 4, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f4 = axis.factor %p0 : !distributed.physical_comm_axis<4, 2> <4, 1>
  %f5 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f6 = axis.product (%f4 : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 4, 1>, %f5 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<4, 2> <4, 1>
  %f8 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f9 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 4, 1>, %f8 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f10 = axis.factor %ti0 : !axis.shape_axis<tensor<2x3x5x2xi8>, 0> <2, 1>
  %f11 = axis.product (%f10 : !axis.axis_factor<!axis.shape_axis<tensor<2x3x5x2xi8>, 0>, 2, 1>)
  %f12 = axis.factor %to0 : !axis.shape_axis<tensor<2x3x5x2xi8>, 0> <2, 1>
  %f13 = axis.product (%f12 : !axis.axis_factor<!axis.shape_axis<tensor<2x3x5x2xi8>, 0>, 2, 1>)
  %f14 = axis.factor %ti1 : !axis.shape_axis<tensor<2x3x5x2xi8>, 1> <3, 1>
  %f15 = axis.product (%f14 : !axis.axis_factor<!axis.shape_axis<tensor<2x3x5x2xi8>, 1>, 3, 1>)
  %f16 = axis.factor %to1 : !axis.shape_axis<tensor<2x3x5x2xi8>, 1> <3, 1>
  %f17 = axis.product (%f16 : !axis.axis_factor<!axis.shape_axis<tensor<2x3x5x2xi8>, 1>, 3, 1>)
  %f18 = axis.factor %ti2 : !axis.shape_axis<tensor<2x3x5x2xi8>, 2> <5, 1>
  %f19 = axis.product (%f18 : !axis.axis_factor<!axis.shape_axis<tensor<2x3x5x2xi8>, 2>, 5, 1>)
  %f20 = axis.factor %to2 : !axis.shape_axis<tensor<2x3x5x2xi8>, 2> <5, 1>
  %f21 = axis.product (%f20 : !axis.axis_factor<!axis.shape_axis<tensor<2x3x5x2xi8>, 2>, 5, 1>)
  %f22 = axis.factor %ti3 : !axis.shape_axis<tensor<2x3x5x2xi8>, 3> <2, 1>
  %f23 = axis.product (%f22 : !axis.axis_factor<!axis.shape_axis<tensor<2x3x5x2xi8>, 3>, 2, 1>)
  %f24 = axis.factor %to3 : !axis.shape_axis<tensor<2x3x5x2xi8>, 3> <2, 1>
  %f25 = axis.product (%f24 : !axis.axis_factor<!axis.shape_axis<tensor<2x3x5x2xi8>, 3>, 2, 1>)
  %f26 = axis.factor %rep4 : !distributed.replication_axis<4> <4, 1>
  %f27 = axis.product (%f26 : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
  %f28 = axis.factor %p0 : !distributed.physical_comm_axis<4, 2> <4, 1>
  %f29 = axis.product (%f28 : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 4, 1>)
  %f30 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %f31 = axis.product (%f30 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f32 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f33 = axis.product (%f32 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map = axis.map %f11, %f15, %f19, %f23, %f27, %f31 to %f13, %f17, %f21, %f25, %f29, %f33 : [!axis.factor_group<2>, !axis.factor_group<3>, !axis.factor_group<5>, !axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<2>, !axis.factor_group<3>, !axis.factor_group<5>, !axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<2x3x5x2xi8>
  %h = distributed.Collective %in : tensor<2x3x5x2xi8> on %f3 : !axis.factor_group<8> to tensor<2x3x5x2xi8> on %f6 : !axis.factor_group<8> reduces (%f9 : !axis.factor_group<8>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<i8>, %rhs_v: tensor<i8>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<i8>
    stablehlo.return %r : tensor<i8>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<2x3x5x2xi8>> -> tensor<2x3x5x2xi8>
}

// -----

// Two nested peels, on p0 and p1, around an all-reduce on p2, share the tile's
// one digit through disjoint factors: each takes a factor of 2 from what the
// other left, and each gather restores its own.
// CHECK-LABEL: module @peels_share_one_digit {
// CHECK: "stablehlo.reduce_scatter"
// CHECK-SAME: replica_groups = dense<{{\[}}[0, 4], [1, 5], [2, 6], [3, 7]]>
// CHECK: }) {{.*}}: (tensor<4096xi8>) -> tensor<2048xi8>
// CHECK: "stablehlo.reduce_scatter"
// CHECK-SAME: replica_groups = dense<{{\[}}[0, 2], [1, 3], [4, 6], [5, 7]]>
// CHECK: }) {{.*}}: (tensor<2048xi8>) -> tensor<1024xi8>
// CHECK: "stablehlo.all_reduce"
// CHECK: "stablehlo.all_gather"
// CHECK-SAME: replica_groups = dense<{{\[}}[0, 2], [1, 3], [4, 6], [5, 7]]>
// CHECK-SAME: (tensor<1024xi8>) -> tensor<2048xi8>
// CHECK: "stablehlo.all_gather"
// CHECK-SAME: replica_groups = dense<{{\[}}[0, 4], [1, 5], [2, 6], [3, 7]]>
// CHECK-SAME: (tensor<2048xi8>) -> tensor<4096xi8>
module @peels_share_one_digit {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>] {axis_bandwidths = [8.0, 1.0, 0.1]}

  func.func @main() {
    return
  }

  %p0, %p1, %p2 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<4096xi8> 0
  %to0 = axis.getaxis tensor<4096xi8> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f3 = axis.factor %p2 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f4 = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f5 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %f6 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f7 = axis.factor %p2 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f8 = axis.product (%f5 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>, %f6 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f9 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %f10 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f11 = axis.factor %p2 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f12 = axis.product (%f9 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>, %f10 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f11 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f13 = axis.factor %ti0 : !axis.shape_axis<tensor<4096xi8>, 0> <4096, 1>
  %f14 = axis.product (%f13 : !axis.axis_factor<!axis.shape_axis<tensor<4096xi8>, 0>, 4096, 1>)
  %f15 = axis.factor %to0 : !axis.shape_axis<tensor<4096xi8>, 0> <4096, 1>
  %f16 = axis.product (%f15 : !axis.axis_factor<!axis.shape_axis<tensor<4096xi8>, 0>, 4096, 1>)
  %f17 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %f18 = axis.product (%f17 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f19 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %f20 = axis.product (%f19 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>)
  %f21 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %f22 = axis.product (%f21 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f23 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f24 = axis.product (%f23 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f25 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %f26 = axis.product (%f25 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f27 = axis.factor %p2 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f28 = axis.product (%f27 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map = axis.map %f14, %f18, %f22, %f26 to %f16, %f20, %f24, %f28 : [!axis.factor_group<4096>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<4096>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<4096xi8>
  %h = distributed.Collective %in : tensor<4096xi8> on %f4 : !axis.factor_group<8> to tensor<4096xi8> on %f8 : !axis.factor_group<8> reduces (%f12 : !axis.factor_group<8>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<i8>, %rhs_v: tensor<i8>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<i8>
    stablehlo.return %r : tensor<i8>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4096xi8>> -> tensor<4096xi8>
}

// -----

// Two nested peels that each take a whole factor: the outer one, on p0, takes
// all of the extent-2 row digit (merged with the column digit after it), and
// the inner one, on p1, takes the major 2 of the column digit. The outer
// peel's shard survives the inner one closing, and its gather restores it.
// CHECK-LABEL: module @peels_shard_separate_digits {
// CHECK: %[[MERGED:.+]] = stablehlo.reshape %{{.+}} : (tensor<2x2048xi8>) -> tensor<4096xi8>
// CHECK: "stablehlo.reduce_scatter"(%[[MERGED]])
// CHECK-SAME: replica_groups = dense<{{\[}}[0, 4], [1, 5], [2, 6], [3, 7]]>
// CHECK: }) {{.*}}: (tensor<4096xi8>) -> tensor<2048xi8>
// CHECK: "stablehlo.reduce_scatter"
// CHECK-SAME: replica_groups = dense<{{\[}}[0, 2], [1, 3], [4, 6], [5, 7]]>
// CHECK: }) {{.*}}: (tensor<2048xi8>) -> tensor<1024xi8>
// CHECK: "stablehlo.all_reduce"
// CHECK: "stablehlo.all_gather"
// CHECK-SAME: replica_groups = dense<{{\[}}[0, 2], [1, 3], [4, 6], [5, 7]]>
// CHECK-SAME: (tensor<1024xi8>) -> tensor<2048xi8>
// CHECK: %[[GATHERED:.+]] = "stablehlo.all_gather"
// CHECK-SAME: replica_groups = dense<{{\[}}[0, 4], [1, 5], [2, 6], [3, 7]]>
// CHECK-SAME: (tensor<2048xi8>) -> tensor<4096xi8>
// CHECK: stablehlo.reshape %[[GATHERED]] : (tensor<4096xi8>) -> tensor<2x2048xi8>
module @peels_shard_separate_digits {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>] {axis_bandwidths = [8.0, 1.0, 0.1]}

  func.func @main() {
    return
  }

  %p0, %p1, %p2 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<2x2048xi8> 0
  %ti1 = axis.getaxis tensor<2x2048xi8> 1
  %to0 = axis.getaxis tensor<2x2048xi8> 0
  %to1 = axis.getaxis tensor<2x2048xi8> 1
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f3 = axis.factor %p2 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f4 = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f5 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %f6 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f7 = axis.factor %p2 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f8 = axis.product (%f5 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>, %f6 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f9 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %f10 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f11 = axis.factor %p2 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f12 = axis.product (%f9 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>, %f10 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f11 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f13 = axis.factor %ti0 : !axis.shape_axis<tensor<2x2048xi8>, 0> <2, 1>
  %f14 = axis.product (%f13 : !axis.axis_factor<!axis.shape_axis<tensor<2x2048xi8>, 0>, 2, 1>)
  %f15 = axis.factor %to0 : !axis.shape_axis<tensor<2x2048xi8>, 0> <2, 1>
  %f16 = axis.product (%f15 : !axis.axis_factor<!axis.shape_axis<tensor<2x2048xi8>, 0>, 2, 1>)
  %f17 = axis.factor %ti1 : !axis.shape_axis<tensor<2x2048xi8>, 1> <2048, 1>
  %f18 = axis.product (%f17 : !axis.axis_factor<!axis.shape_axis<tensor<2x2048xi8>, 1>, 2048, 1>)
  %f19 = axis.factor %to1 : !axis.shape_axis<tensor<2x2048xi8>, 1> <2048, 1>
  %f20 = axis.product (%f19 : !axis.axis_factor<!axis.shape_axis<tensor<2x2048xi8>, 1>, 2048, 1>)
  %f21 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %f22 = axis.product (%f21 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f23 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %f24 = axis.product (%f23 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>)
  %f25 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %f26 = axis.product (%f25 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f27 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f28 = axis.product (%f27 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f29 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %f30 = axis.product (%f29 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f31 = axis.factor %p2 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f32 = axis.product (%f31 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map = axis.map %f14, %f18, %f22, %f26, %f30 to %f16, %f20, %f24, %f28, %f32 : [!axis.factor_group<2>, !axis.factor_group<2048>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<2>, !axis.factor_group<2048>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<2x2048xi8>
  %h = distributed.Collective %in : tensor<2x2048xi8> on %f4 : !axis.factor_group<8> to tensor<2x2048xi8> on %f8 : !axis.factor_group<8> reduces (%f12 : !axis.factor_group<8>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<i8>, %rhs_v: tensor<i8>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<i8>
    stablehlo.return %r : tensor<i8>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<2x2048xi8>> -> tensor<2x2048xi8>
}

// -----

// A peel on p0 around all-to-alls on p1 and p2 that move both of the tile's
// digits out: no digit stays local to hold the shard, so the peel lowers as an
// all-reduce over p0 and its closing gather emits nothing.
// CHECK-LABEL: module @peel_falls_back_to_all_reduce {
// CHECK-NOT: "stablehlo.reduce_scatter"
// CHECK: "stablehlo.all_reduce"
// CHECK-SAME: replica_groups = dense<{{\[}}[0, 4], [1, 5], [2, 6], [3, 7]]>
// CHECK: }) {{.*}}kind = "reduce-scatter"{{.*}}: (tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NOT: "stablehlo.reduce_scatter"
// CHECK: "stablehlo.all_to_all"
// CHECK: "stablehlo.all_to_all"
// CHECK-NOT: "stablehlo.all_gather"
// CHECK-NOT: "stablehlo.reduce_scatter"
module @peel_falls_back_to_all_reduce {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>] {axis_bandwidths = [8.0, 0.1, 0.1]}

  func.func @main() {
    return
  }

  %p0, %p1, %p2 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<2x2xf32> 0
  %ti1 = axis.getaxis tensor<2x2xf32> 1
  %to0 = axis.getaxis tensor<2x2xf32> 0
  %to1 = axis.getaxis tensor<2x2xf32> 1
  %i0 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %i1 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %i2 = axis.factor %p2 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %in_mesh = axis.product (%i0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>, %i1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %i2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %o0 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %o1 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %o2 = axis.factor %p2 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %out_mesh = axis.product (%o0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>, %o1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %o2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %r0 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %reduced = axis.product (%r0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>)
  %a0 = axis.factor %ti0 : !axis.shape_axis<tensor<2x2xf32>, 0> <2, 1>
  %l0 = axis.product (%a0 : !axis.axis_factor<!axis.shape_axis<tensor<2x2xf32>, 0>, 2, 1>)
  %a1 = axis.factor %ti1 : !axis.shape_axis<tensor<2x2xf32>, 1> <2, 1>
  %l1 = axis.product (%a1 : !axis.axis_factor<!axis.shape_axis<tensor<2x2xf32>, 1>, 2, 1>)
  %ar = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %l2 = axis.product (%ar : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %a3 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %l3 = axis.product (%a3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %a4 = axis.factor %p2 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %l4 = axis.product (%a4 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %b0 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %r_0 = axis.product (%b0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %b1 = axis.factor %p2 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %r_1 = axis.product (%b1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %b2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %r_2 = axis.product (%b2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>)
  %b3 = axis.factor %to0 : !axis.shape_axis<tensor<2x2xf32>, 0> <2, 1>
  %r_3 = axis.product (%b3 : !axis.axis_factor<!axis.shape_axis<tensor<2x2xf32>, 0>, 2, 1>)
  %b4 = axis.factor %to1 : !axis.shape_axis<tensor<2x2xf32>, 1> <2, 1>
  %r_4 = axis.product (%b4 : !axis.axis_factor<!axis.shape_axis<tensor<2x2xf32>, 1>, 2, 1>)
  %map = axis.map %l0, %l1, %l2, %l3, %l4 to %r_0, %r_1, %r_2, %r_3, %r_4 : [!axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<2x2xf32>
  %h = distributed.Collective %in : tensor<2x2xf32> on %in_mesh : !axis.factor_group<8> to tensor<2x2xf32> on %out_mesh : !axis.factor_group<8> reduces (%reduced : !axis.factor_group<2>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<2x2xf32>> -> tensor<2x2xf32>
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
