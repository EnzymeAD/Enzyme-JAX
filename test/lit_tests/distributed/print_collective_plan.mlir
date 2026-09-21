// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-print-collective-plan %s 2>&1 >/dev/null | FileCheck %s
// RUN: enzymexlamlir-opt --split-input-file --distributed-print-collective-plan %s 2>&1 >/dev/null | FileCheck %s --check-prefix=RAW

// distributed-print-collective-plan reports a collective's normal form as a
// remark (remarks go to stderr, so the RUN lines discard stdout). Each mesh
// atom line reads `mesh<axis>.<atom>: extent E stride S in <role> out <role>
// => <primitive>`, where `in` is where the atom's input digit goes and `out`
// is where its output digit comes from; tile atoms list the atom they pair
// with; the payload is the per-device input tile in bytes. One module per
// role pair. Inputs are not written in atomic form; the first RUN line
// atomizes them first.

// Reduced -> replicate: the reduced atom's output is drawn from a replication, so each device ends with the full sum (all-reduce).
// CHECK: mesh0.0: extent 2 stride 1 in reduced out replicate => all-reduce
// CHECK-NEXT: in0.0: extent 4 pairs out0.0
// CHECK-NEXT: out0.0: extent 4 pairs in0.0
// CHECK-NEXT: reduction: add
// CHECK-NEXT: payload bytes: 16
// RAW: mesh0.0: extent 2 stride 1 in reduced out replicate => all-reduce
// RAW-NEXT: in0.0: extent 4 pairs out0.0
// RAW-NEXT: out0.0: extent 4 pairs in0.0
// RAW-NEXT: reduction: add
// RAW-NEXT: payload bytes: 16
module @all_reduce {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<4xf32> 0
  %to0 = axis.getaxis tensor<4xf32> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %red0 = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f4 = axis.factor %ti0 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %lhs0 = axis.product (%f4 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %f5 = axis.factor %to0 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %rhs0 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %f6 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %lhs1 = axis.product (%f6 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs1 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<4xf32>
  %h = distributed.Collective %in : tensor<4xf32> on %mesh_in : !axis.factor_group<2> to tensor<4xf32> on %mesh_out : !axis.factor_group<2> reduces (%red0 : !axis.factor_group<2>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
}

// -----

// Same as all_reduce but the reduction body is not a recognized associative op, so its kind is reported as unknown.
// CHECK: mesh0.0: extent 2 stride 1 in reduced out replicate => all-reduce
// CHECK-NEXT: in0.0: extent 4 pairs out0.0
// CHECK-NEXT: out0.0: extent 4 pairs in0.0
// CHECK-NEXT: reduction: unknown
// CHECK-NEXT: payload bytes: 16
// RAW: mesh0.0: extent 2 stride 1 in reduced out replicate => all-reduce
// RAW-NEXT: in0.0: extent 4 pairs out0.0
// RAW-NEXT: out0.0: extent 4 pairs in0.0
// RAW-NEXT: reduction: unknown
// RAW-NEXT: payload bytes: 16
module @all_reduce_unrecognized_body {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<4xf32> 0
  %to0 = axis.getaxis tensor<4xf32> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %red0 = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f4 = axis.factor %ti0 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %lhs0 = axis.product (%f4 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %f5 = axis.factor %to0 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %rhs0 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %f6 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %lhs1 = axis.product (%f6 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs1 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<4xf32>
  %h = distributed.Collective %in : tensor<4xf32> on %mesh_in : !axis.factor_group<2> to tensor<4xf32> on %mesh_out : !axis.factor_group<2> reduces (%red0 : !axis.factor_group<2>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %r = stablehlo.subtract %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
}

// -----

// Reduced -> tile: the reduced atom's output digit is a piece of the input tile dimension, so the sum is scattered (reduce-scatter). The in-tile pair 4 -> (mesh 2, out tile 2) atomizes into in0.0 -> mesh0.0 and in0.1 -> out0.0.
// CHECK: mesh0.0: extent 2 stride 1 in reduced out tile(in0.0) => reduce-scatter
// CHECK-NEXT: in0.0: extent 2 pairs mesh0.0
// CHECK-NEXT: in0.1: extent 2 pairs out0.0
// CHECK-NEXT: out0.0: extent 2 pairs in0.1
// CHECK-NEXT: reduction: max
// CHECK-NEXT: payload bytes: 16
// RAW: print-collective-plan: no plan (collective is not atomic)
module @reduce_scatter {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %ti0 = axis.getaxis tensor<4xf32> 0
  %to0 = axis.getaxis tensor<2xf32> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %red0 = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f4 = axis.factor %ti0 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %lhs0 = axis.product (%f4 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %f5 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f6 = axis.factor %to0 : !axis.shape_axis<tensor<2xf32>, 0> <2, 1>
  %rhs0 = axis.product (%f5 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %f6 : !axis.axis_factor<!axis.shape_axis<tensor<2xf32>, 0>, 2, 1>)
  %map = axis.map %lhs0 to %rhs0 : [!axis.factor_group<4>] [!axis.factor_group<4>]

  %in = tensor.empty() : tensor<4xf32>
  %h = distributed.Collective %in : tensor<4xf32> on %mesh_in : !axis.factor_group<2> to tensor<2xf32> on %mesh_out : !axis.factor_group<2> reduces (%red0 : !axis.factor_group<2>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %r = stablehlo.maximum %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<2xf32>> -> tensor<2xf32>
}

// -----

// Reduced -> mesh: mesh1 is reduced and its output comes from mesh0, whose own input goes to mesh1 (mesh0 needs a permute, and mesh1 reduces then permutes).
// CHECK: mesh0.0: extent 2 stride 1 in mesh(mesh1.0) out replicate => permute
// CHECK-NEXT: mesh1.0: extent 2 stride 1 in reduced out mesh(mesh0.0) => reduce-then-permute
// CHECK-NEXT: in0.0: extent 4 pairs out0.0
// CHECK-NEXT: out0.0: extent 4 pairs in0.0
// CHECK-NEXT: reduction: add
// CHECK-NEXT: payload bytes: 16
// RAW: mesh0.0: extent 2 stride 1 in mesh(mesh1.0) out replicate => permute
// RAW-NEXT: mesh1.0: extent 2 stride 1 in reduced out mesh(mesh0.0) => reduce-then-permute
// RAW-NEXT: in0.0: extent 4 pairs out0.0
// RAW-NEXT: out0.0: extent 4 pairs in0.0
// RAW-NEXT: reduction: add
// RAW-NEXT: payload bytes: 16
module @reduce_then_permute {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<4xf32> 0
  %to0 = axis.getaxis tensor<4xf32> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f4 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f4 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f5 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %red0 = axis.product (%f5 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f6 = axis.factor %ti0 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %lhs0 = axis.product (%f6 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %f7 = axis.factor %to0 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %rhs0 = axis.product (%f7 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %f8 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %lhs1 = axis.product (%f8 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f9 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs1 = axis.product (%f9 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f10 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %lhs2 = axis.product (%f10 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f11 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %rhs2 = axis.product (%f11 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %map = axis.map %lhs0, %lhs1, %lhs2 to %rhs0, %rhs1, %rhs2 : [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<4xf32>
  %h = distributed.Collective %in : tensor<4xf32> on %mesh_in : !axis.factor_group<4> to tensor<4xf32> on %mesh_out : !axis.factor_group<4> reduces (%red0 : !axis.factor_group<2>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
}

// -----

// Tile -> replicate: the mesh atom's input digit becomes part of the larger output tile, and every device holds the result (all-gather).
// CHECK: mesh0.0: extent 2 stride 1 in tile(out0.0) out replicate => all-gather
// CHECK-NEXT: in0.0: extent 2 pairs out0.1
// CHECK-NEXT: out0.0: extent 2 pairs mesh0.0
// CHECK-NEXT: out0.1: extent 2 pairs in0.0
// CHECK-NEXT: payload bytes: 8
// RAW: print-collective-plan: no plan (collective is not atomic)
module @all_gather {
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
  %lhs0 = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %f4 : !axis.axis_factor<!axis.shape_axis<tensor<2xf32>, 0>, 2, 1>)
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

// Tile -> tile: the mesh atom moves from input dim 1 to output dim 0's split (all-to-all).
// CHECK: mesh0.0: extent 2 stride 1 in tile(out1.0) out tile(in0.0) => tile-to-tile
// CHECK-NEXT: in0.0: extent 2 pairs mesh0.0
// CHECK-NEXT: in0.1: extent 2 pairs out0.0
// CHECK-NEXT: out0.0: extent 2 pairs in0.1
// CHECK-NEXT: out1.0: extent 2 pairs mesh0.0
// CHECK-NEXT: payload bytes: 16
// RAW: print-collective-plan: no plan (collective is not atomic)
module @tile_to_tile {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %ti0 = axis.getaxis tensor<4x1xf32> 0
  %ti1 = axis.getaxis tensor<4x1xf32> 1
  %to0 = axis.getaxis tensor<2x2xf32> 0
  %to1 = axis.getaxis tensor<2x2xf32> 1
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f3 = axis.factor %ti0 : !axis.shape_axis<tensor<4x1xf32>, 0> <4, 1>
  %lhs0 = axis.product (%f3 : !axis.axis_factor<!axis.shape_axis<tensor<4x1xf32>, 0>, 4, 1>)
  %f4 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f5 = axis.factor %to0 : !axis.shape_axis<tensor<2x2xf32>, 0> <2, 1>
  %rhs0 = axis.product (%f4 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %f5 : !axis.axis_factor<!axis.shape_axis<tensor<2x2xf32>, 0>, 2, 1>)
  %f6 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs1 = axis.product (%f6 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f7 = axis.factor %to1 : !axis.shape_axis<tensor<2x2xf32>, 1> <2, 1>
  %rhs1 = axis.product (%f7 : !axis.axis_factor<!axis.shape_axis<tensor<2x2xf32>, 1>, 2, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<4x1xf32>
  %h = distributed.Collective %in : tensor<4x1xf32> on %mesh_in : !axis.factor_group<2> to tensor<2x2xf32> on %mesh_out : !axis.factor_group<2> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<2x2xf32>> -> tensor<2x2xf32>
}

// -----

// Replicate -> tile: the input is replicated and each device keeps its own slice of the output tile (free).
// CHECK: mesh0.0: extent 2 stride 1 in replicate out tile(in0.0) => local-slice
// CHECK-NEXT: in0.0: extent 2 pairs mesh0.0
// CHECK-NEXT: in0.1: extent 2 pairs out0.0
// CHECK-NEXT: out0.0: extent 2 pairs in0.1
// CHECK-NEXT: payload bytes: 16
// RAW: print-collective-plan: no plan (collective is not atomic)
module @local_slice {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<4xf32> 0
  %to0 = axis.getaxis tensor<2xf32> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs0 = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f4 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %rhs0 = axis.product (%f4 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f5 = axis.factor %ti0 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %lhs1 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %f6 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %f7 = axis.factor %to0 : !axis.shape_axis<tensor<2xf32>, 0> <2, 1>
  %rhs1 = axis.product (%f6 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>, %f7 : !axis.axis_factor<!axis.shape_axis<tensor<2xf32>, 0>, 2, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<2>, !axis.factor_group<4>] [!axis.factor_group<2>, !axis.factor_group<4>]

  %in = tensor.empty() : tensor<4xf32>
  %h = distributed.Collective %in : tensor<4xf32> on %mesh_in : !axis.factor_group<2> to tensor<2xf32> on %mesh_out : !axis.factor_group<2> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<2xf32>> -> tensor<2xf32>
}

// -----

// Replicate -> replicate: replicated in, replicated out; no communication.
// CHECK: mesh0.0: extent 2 stride 1 in replicate out replicate => no-op
// CHECK-NEXT: in0.0: extent 4 pairs out0.0
// CHECK-NEXT: out0.0: extent 4 pairs in0.0
// CHECK-NEXT: payload bytes: 16
// RAW: mesh0.0: extent 2 stride 1 in replicate out replicate => no-op
// RAW-NEXT: in0.0: extent 4 pairs out0.0
// RAW-NEXT: out0.0: extent 4 pairs in0.0
// RAW-NEXT: payload bytes: 16
module @replicate_to_replicate {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<4xf32> 0
  %to0 = axis.getaxis tensor<4xf32> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f3 = axis.factor %ti0 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %lhs0 = axis.product (%f3 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %f4 = axis.factor %to0 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %rhs0 = axis.product (%f4 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %f5 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs1 = axis.product (%f5 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f6 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %rhs1 = axis.product (%f6 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f7 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %lhs2 = axis.product (%f7 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f8 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs2 = axis.product (%f8 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map = axis.map %lhs0, %lhs1, %lhs2 to %rhs0, %rhs1, %rhs2 : [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<4xf32>
  %h = distributed.Collective %in : tensor<4xf32> on %mesh_in : !axis.factor_group<2> to tensor<4xf32> on %mesh_out : !axis.factor_group<2> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
}

// -----

// Mesh -> same mesh atom: each device keeps its own digit; no communication.
// CHECK: mesh0.0: extent 2 stride 1 in mesh(mesh0.0) out mesh(mesh0.0) => no-op
// CHECK-NEXT: in0.0: extent 4 pairs out0.0
// CHECK-NEXT: out0.0: extent 4 pairs in0.0
// CHECK-NEXT: payload bytes: 16
// RAW: mesh0.0: extent 2 stride 1 in mesh(mesh0.0) out mesh(mesh0.0) => no-op
// RAW-NEXT: in0.0: extent 4 pairs out0.0
// RAW-NEXT: out0.0: extent 4 pairs in0.0
// RAW-NEXT: payload bytes: 16
module @identity {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %ti0 = axis.getaxis tensor<4xf32> 0
  %to0 = axis.getaxis tensor<4xf32> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f3 = axis.factor %ti0 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %lhs0 = axis.product (%f3 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %f4 = axis.factor %to0 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %rhs0 = axis.product (%f4 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %f5 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs1 = axis.product (%f5 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f6 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs1 = axis.product (%f6 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<4xf32>
  %h = distributed.Collective %in : tensor<4xf32> on %mesh_in : !axis.factor_group<2> to tensor<4xf32> on %mesh_out : !axis.factor_group<2> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
}

// -----

// Mesh -> mesh: the two mesh atoms swap digits (collective permute).
// CHECK: mesh0.0: extent 2 stride 1 in mesh(mesh1.0) out mesh(mesh1.0) => permute
// CHECK-NEXT: mesh1.0: extent 2 stride 1 in mesh(mesh0.0) out mesh(mesh0.0) => permute
// CHECK-NEXT: in0.0: extent 4 pairs out0.0
// CHECK-NEXT: out0.0: extent 4 pairs in0.0
// CHECK-NEXT: payload bytes: 16
// RAW: mesh0.0: extent 2 stride 1 in mesh(mesh1.0) out mesh(mesh1.0) => permute
// RAW-NEXT: mesh1.0: extent 2 stride 1 in mesh(mesh0.0) out mesh(mesh0.0) => permute
// RAW-NEXT: in0.0: extent 4 pairs out0.0
// RAW-NEXT: out0.0: extent 4 pairs in0.0
// RAW-NEXT: payload bytes: 16
module @permute {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %ti0 = axis.getaxis tensor<4xf32> 0
  %to0 = axis.getaxis tensor<4xf32> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f4 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f4 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f5 = axis.factor %ti0 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %lhs0 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %f6 = axis.factor %to0 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %rhs0 = axis.product (%f6 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %lhs1 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f8 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs1 = axis.product (%f8 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f9 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs2 = axis.product (%f9 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f10 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %rhs2 = axis.product (%f10 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %map = axis.map %lhs0, %lhs1, %lhs2 to %rhs0, %rhs1, %rhs2 : [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<4xf32>
  %h = distributed.Collective %in : tensor<4xf32> on %mesh_in : !axis.factor_group<4> to tensor<4xf32> on %mesh_out : !axis.factor_group<4> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
}
