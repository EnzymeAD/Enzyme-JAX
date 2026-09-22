// RUN: enzymexlamlir-opt --split-input-file --distributed-atomize-collectives --distributed-lower-collectives %s | FileCheck %s

// distributed-lower-collectives replaces an atomic distributed.Collective/
// Await pair with the real StableHLO async op chain its decomposition plan
// (planCollective, CollectiveDecomposer.h) calls for: one
// stablehlo.async_start/async_done pair per PrimitiveStep, each tagged with
// a distributed.primitive_cost dictionary attribute carrying that step's own
// cost fields. Every RUN line atomizes the input first, since planCollective
// requires an atomic collective. CHECK lines below only pin the shape that
// matters (the op chain and its cost attribute); the atomization preamble
// above it is left unchecked.

// All-reduce: a single stablehlo.all_reduce, async-wrapped.
// CHECK-LABEL: module @all_reduce {
// CHECK: %[[IN:.+]] = tensor.empty() : tensor<4xf32>
// CHECK: %[[START:.+]] = "stablehlo.async_start"(%[[IN]]) (
// CHECK: ^bb0(%[[ARG:.+]]: tensor<4xf32>):
// CHECK:   "stablehlo.all_reduce"(%[[ARG]])
// CHECK-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<mesh = #stablehlo.mesh<axes=[<name = "axis_0", size = 2>]>, axes = [#stablehlo.axis_ref<name = "axis_0", sub_axis_info = (1)2>]>
// CHECK-SAME: use_global_device_ids
// CHECK:   stablehlo.add
// CHECK: }) {distributed.primitive_cost = {isolated_duration = {{.*}} : f64, kind = "all-reduce", latency = {{.*}} : f64, payload_in = 16 : i64, payload_out = 16 : i64, rho = [1.000000e+00], transfer_time = {{.*}} : f64, volume = [{{.*}}]}} : (tensor<4xf32>) -> !stablehlo.future<tensor<4xf32>>
// CHECK: %{{.+}} = "stablehlo.async_done"(%[[START]]) : (!stablehlo.future<tensor<4xf32>>) -> tensor<4xf32>
// CHECK-NOT: distributed.Collective
// CHECK-NOT: distributed.Await
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

// Reduce-scatter: a single stablehlo.reduce_scatter, async-wrapped.
// CHECK-LABEL: module @reduce_scatter {
// CHECK: %[[IN:.+]] = tensor.empty() : tensor<4xf32>
// CHECK: %[[START:.+]] = "stablehlo.async_start"(%[[IN]]) (
// CHECK: ^bb0(%[[ARG:.+]]: tensor<4xf32>):
// CHECK:   "stablehlo.reduce_scatter"(%[[ARG]])
// CHECK-SAME: scatter_dimension = 0
// CHECK-SAME: use_global_device_ids
// CHECK:   stablehlo.maximum
// CHECK: }) {distributed.primitive_cost = {isolated_duration = {{.*}} : f64, kind = "reduce-scatter", latency = {{.*}} : f64, payload_in = 16 : i64, payload_out = 8 : i64, rho = [1.000000e+00], transfer_time = {{.*}} : f64, volume = [{{.*}}]}} : (tensor<4xf32>) -> !stablehlo.future<tensor<2xf32>>
// CHECK: %{{.+}} = "stablehlo.async_done"(%[[START]]) : (!stablehlo.future<tensor<2xf32>>) -> tensor<2xf32>
// CHECK-NOT: distributed.Collective
// CHECK-NOT: distributed.Await
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

// All-gather: a single stablehlo.all_gather, async-wrapped, no reduction
// body.
// CHECK-LABEL: module @all_gather {
// CHECK: %[[IN:.+]] = tensor.empty() : tensor<2xf32>
// CHECK: %[[START:.+]] = "stablehlo.async_start"(%[[IN]]) (
// CHECK: ^bb0(%[[ARG:.+]]: tensor<2xf32>):
// CHECK:   "stablehlo.all_gather"(%[[ARG]])
// CHECK-SAME: all_gather_dim = 0
// CHECK-SAME: use_global_device_ids
// CHECK: }) {distributed.primitive_cost = {isolated_duration = {{.*}} : f64, kind = "all-gather", latency = {{.*}} : f64, payload_in = 8 : i64, payload_out = 16 : i64, rho = [1.000000e+00], transfer_time = {{.*}} : f64, volume = [{{.*}}]}} : (tensor<2xf32>) -> !stablehlo.future<tensor<4xf32>>
// CHECK: %{{.+}} = "stablehlo.async_done"(%[[START]]) : (!stablehlo.future<tensor<4xf32>>) -> tensor<4xf32>
// CHECK-NOT: distributed.Collective
// CHECK-NOT: distributed.Await
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

// All-to-all: split/concat need distinct dimensions, which a flat 1-D
// payload does not have, so a reshape to [1, N] and back brackets the
// async-wrapped stablehlo.all_to_all from outside the async region
// (async_start's body may only ever contain the one op it wraps).
// CHECK-LABEL: module @tile_to_tile {
// CHECK: %[[IN:.+]] = tensor.empty() : tensor<4x1xf32>
// CHECK: %[[FLAT:.+]] = stablehlo.reshape %[[IN]] : (tensor<4x1xf32>) -> tensor<4xf32>
// CHECK: %[[SPLIT:.+]] = stablehlo.reshape %[[FLAT]] : (tensor<4xf32>) -> tensor<1x4xf32>
// CHECK: %[[START:.+]] = "stablehlo.async_start"(%[[SPLIT]]) (
// CHECK: ^bb0(%[[ARG:.+]]: tensor<1x4xf32>):
// CHECK:   "stablehlo.all_to_all"(%[[ARG]])
// CHECK-SAME: concat_dimension = 0
// CHECK-SAME: split_count = 2
// CHECK-SAME: split_dimension = 1
// CHECK: }) {distributed.primitive_cost = {isolated_duration = {{.*}} : f64, kind = "all-to-all", latency = {{.*}} : f64, payload_in = 16 : i64, payload_out = 16 : i64, rho = [1.000000e+00], transfer_time = {{.*}} : f64, volume = [{{.*}}]}} : (tensor<1x4xf32>) -> !stablehlo.future<tensor<2x2xf32>>
// CHECK: %[[DONE:.+]] = "stablehlo.async_done"(%[[START]]) : (!stablehlo.future<tensor<2x2xf32>>) -> tensor<2x2xf32>
// CHECK: %[[FLAT2:.+]] = stablehlo.reshape %[[DONE]] : (tensor<2x2xf32>) -> tensor<4xf32>
// CHECK: stablehlo.reshape %[[FLAT2]] : (tensor<4xf32>) -> tensor<2x2xf32>
// CHECK-NOT: distributed.Collective
// CHECK-NOT: distributed.Await
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

// Collective permute: a single stablehlo.collective_permute, async-wrapped.
// The source/target pairing is computed statically from the two physical
// axes' extents/idStride (device 1 <-> device 2 swap their digits on the
// two extent-2 axes; devices 0 and 3, whose digits on both axes are equal,
// map to themselves).
// CHECK-LABEL: module @permute {
// CHECK: %[[IN:.+]] = tensor.empty() : tensor<4xf32>
// CHECK: %[[START:.+]] = "stablehlo.async_start"(%[[IN]]) (
// CHECK: ^bb0(%[[ARG:.+]]: tensor<4xf32>):
// CHECK:   "stablehlo.collective_permute"(%[[ARG]])
// CHECK-SAME: source_target_pairs = dense<{{\[}}[0, 0], [1, 2], [2, 1], [3, 3]]> : tensor<4x2xi64>
// CHECK: }) {distributed.primitive_cost = {isolated_duration = {{.*}} : f64, kind = "permute", latency = {{.*}} : f64, payload_in = 16 : i64, payload_out = 16 : i64, rho = [1.000000e+00, 1.000000e+00], transfer_time = {{.*}} : f64, volume = [{{.*}}, {{.*}}]}} : (tensor<4xf32>) -> !stablehlo.future<tensor<4xf32>>
// CHECK: %{{.+}} = "stablehlo.async_done"(%[[START]]) : (!stablehlo.future<tensor<4xf32>>) -> tensor<4xf32>
// CHECK-NOT: distributed.Collective
// CHECK-NOT: distributed.Await
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

// -----

// Local slice: no StableHLO collective. The per-device coordinate along the
// mesh axis is decoded from stablehlo.partition_id (id / idStride % extent)
// outside the async region (whose body may only ever contain the one op it
// wraps, here stablehlo.dynamic_slice), then handed in as a second async
// operand alongside the tensor being sliced.
// CHECK-LABEL: module @local_slice {
// CHECK: %[[IN:.+]] = tensor.empty() : tensor<4xf32>
// CHECK: %[[PID:.+]] = stablehlo.partition_id : tensor<ui32>
// CHECK: %[[PIDI:.+]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
// CHECK: %[[STRIDE:.+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK: %[[EXTENT:.+]] = stablehlo.constant dense<2> : tensor<i64>
// CHECK: %[[DIV:.+]] = stablehlo.divide %[[PIDI]], %[[STRIDE]] : tensor<i64>
// CHECK: %[[COORD:.+]] = stablehlo.remainder %[[DIV]], %[[EXTENT]] : tensor<i64>
// CHECK: %[[CHUNK:.+]] = stablehlo.constant dense<2> : tensor<i64>
// CHECK: %[[START_IDX:.+]] = stablehlo.multiply %[[COORD]], %[[CHUNK]] : tensor<i64>
// CHECK: %[[START:.+]] = "stablehlo.async_start"(%[[IN]], %[[START_IDX]]) (
// CHECK: ^bb0(%[[ARG:.+]]: tensor<4xf32>, %[[ARGIDX:.+]]: tensor<i64>):
// CHECK:   stablehlo.dynamic_slice %[[ARG]], %[[ARGIDX]], sizes = [2]
// CHECK: }) {distributed.primitive_cost = {isolated_duration = 0.000000e+00 : f64, kind = "local-slice", latency = 0.000000e+00 : f64, payload_in = 16 : i64, payload_out = 8 : i64, rho = [0.000000e+00], transfer_time = 0.000000e+00 : f64, volume = [0.000000e+00]}} : (tensor<4xf32>, tensor<i64>) -> !stablehlo.future<tensor<2xf32>>
// CHECK: %{{.+}} = "stablehlo.async_done"(%[[START]]) : (!stablehlo.future<tensor<2xf32>>) -> tensor<2xf32>
// CHECK-NOT: distributed.Collective
// CHECK-NOT: distributed.Await
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

// Multi-step chain: a peeled all-reduce (D17) decomposes into a
// reduce-scatter on the cheaper axis followed by an all-reduce on the
// shrunk payload (see decompose_collective_peel.mlir's identical
// peel_not_cheaper_uniform_bandwidth case for the cost derivation), so this
// atomic collective's chain has two steps. The second step's async_start
// consumes the first step's async_done result directly.
// CHECK-LABEL: module @peel_not_cheaper_uniform_bandwidth {
// CHECK: %[[IN:.+]] = tensor.empty() : tensor<64xi8>
// CHECK: %[[START0:.+]] = "stablehlo.async_start"(%[[IN]]) (
// CHECK: ^bb0(%{{.+}}: tensor<64xi8>):
// CHECK:   "stablehlo.reduce_scatter"
// CHECK: }) {distributed.primitive_cost = {isolated_duration = 4.812000e+01 : f64, kind = "reduce-scatter", latency = {{.*}} : f64, payload_in = 64 : i64, payload_out = 16 : i64
// CHECK: %[[DONE0:.+]] = "stablehlo.async_done"(%[[START0]]) : (!stablehlo.future<tensor<16xi8>>) -> tensor<16xi8>
// CHECK: %[[START1:.+]] = "stablehlo.async_start"(%[[DONE0]]) (
// CHECK: ^bb0(%{{.+}}: tensor<16xi8>):
// CHECK:   "stablehlo.all_reduce"
// CHECK: }) {distributed.primitive_cost = {isolated_duration = 2.414000e+01 : f64, kind = "all-reduce", latency = {{.*}} : f64, payload_in = 16 : i64, payload_out = 16 : i64
// CHECK: %{{.+}} = "stablehlo.async_done"(%[[START1]]) : (!stablehlo.future<tensor<16xi8>>) -> tensor<16xi8>
// CHECK-NOT: distributed.Collective
// CHECK-NOT: distributed.Await
module @peel_not_cheaper_uniform_bandwidth {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 4>, !distributed.physical_comm_axis<4, 1>]

  func.func @main() {
    return
  }

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 4>, !distributed.physical_comm_axis<4, 1>
  %rep4 = distributed.ReplicationAxis 4 : !distributed.replication_axis<4>
  %ti0 = axis.getaxis tensor<64xi8> 0
  %to0 = axis.getaxis tensor<16xi8> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %f4 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %mesh_out = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>, %f4 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f5 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %f6 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %red0 = axis.product (%f5 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>, %f6 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %f7 = axis.factor %ti0 : !axis.shape_axis<tensor<64xi8>, 0> <64, 1>
  %lhs0 = axis.product (%f7 : !axis.axis_factor<!axis.shape_axis<tensor<64xi8>, 0>, 64, 1>)
  %f8 = axis.factor %p1 : !distributed.physical_comm_axis<4, 1> <4, 1>
  %f9 = axis.factor %to0 : !axis.shape_axis<tensor<16xi8>, 0> <16, 1>
  %rhs0 = axis.product (%f8 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>, %f9 : !axis.axis_factor<!axis.shape_axis<tensor<16xi8>, 0>, 16, 1>)
  %f10 = axis.factor %rep4 : !distributed.replication_axis<4> <4, 1>
  %lhs1 = axis.product (%f10 : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
  %f11 = axis.factor %p0 : !distributed.physical_comm_axis<4, 4> <4, 1>
  %rhs1 = axis.product (%f11 : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 4, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<64>, !axis.factor_group<4>] [!axis.factor_group<64>, !axis.factor_group<4>]

  %in = tensor.empty() : tensor<64xi8>
  %h = distributed.Collective %in : tensor<64xi8> on %mesh_in : !axis.factor_group<16> to tensor<16xi8> on %mesh_out : !axis.factor_group<16> reduces (%red0 : !axis.factor_group<16>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<i8>, %rhs_v: tensor<i8>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<i8>
    stablehlo.return %r : tensor<i8>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<16xi8>> -> tensor<16xi8>
}

// -----

// Partial-axis all-reduce: a single physical axis of extent 4 split into two
// atoms (a "hi" atom, stride 2, left untouched by a self Mesh -> Mesh map,
// and a "lo" atom, stride 1, actually reduced), so the all-reduce's mesh
// operand spans a genuine sub-portion of one physical axis rather than the
// whole axis -- StepAtom's stride field (CollectiveCost.h) is what makes this
// positionable at all. sub_axis_info's pre_size 1 is the lo atom's own
// stride (1) times the axis's idStride (1, this mesh's only axis), grouping
// devices that agree on the untouched hi digit: {0,1} and {2,3}.
// CHECK-LABEL: module @partial_axis_all_reduce {
// CHECK: %[[IN:.+]] = tensor.empty() : tensor<8xi8>
// CHECK: %[[START:.+]] = "stablehlo.async_start"(%[[IN]]) (
// CHECK: ^bb0(%[[ARG:.+]]: tensor<8xi8>):
// CHECK:   "stablehlo.all_reduce"(%[[ARG]])
// CHECK-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<mesh = #stablehlo.mesh<axes=[<name = "axis_0", size = 4>]>, axes = [#stablehlo.axis_ref<name = "axis_0", sub_axis_info = (1)2>]>
// CHECK-SAME: use_global_device_ids
// CHECK:   stablehlo.add
// CHECK: }) {distributed.primitive_cost = {isolated_duration = {{.*}} : f64, kind = "all-reduce", latency = {{.*}} : f64, payload_in = 8 : i64, payload_out = 8 : i64, rho = [1.000000e+00], transfer_time = {{.*}} : f64, volume = [{{.*}}]}} : (tensor<8xi8>) -> !stablehlo.future<tensor<8xi8>>
// CHECK: %{{.+}} = "stablehlo.async_done"(%[[START]]) : (!stablehlo.future<tensor<8xi8>>) -> tensor<8xi8>
// CHECK-NOT: distributed.Collective
// CHECK-NOT: distributed.Await
module @partial_axis_all_reduce {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<8xi8> 0
  %to0 = axis.getaxis tensor<8xi8> 0

  // mesh_in/mesh_out: the whole axis, as two separate atoms (hi: stride 2,
  // lo: stride 1).
  %hi_in = axis.factor %p0 : !distributed.physical_comm_axis<4, 1> <2, 2>
  %lo_in = axis.factor %p0 : !distributed.physical_comm_axis<4, 1> <2, 1>
  %mesh_in = axis.product (%hi_in : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 2, 2>, %lo_in : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 2, 1>)
  %hi_out = axis.factor %p0 : !distributed.physical_comm_axis<4, 1> <2, 2>
  %lo_out = axis.factor %p0 : !distributed.physical_comm_axis<4, 1> <2, 1>
  %mesh_out = axis.product (%hi_out : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 2, 2>, %lo_out : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 2, 1>)

  // Only the "lo" atom is reduced.
  %lo_red = axis.factor %p0 : !distributed.physical_comm_axis<4, 1> <2, 1>
  %red0 = axis.product (%lo_red : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 2, 1>)

  // Tile self-map (identity relabeling, no mesh involvement).
  %f_ti = axis.factor %ti0 : !axis.shape_axis<tensor<8xi8>, 0> <8, 1>
  %lhs0 = axis.product (%f_ti : !axis.axis_factor<!axis.shape_axis<tensor<8xi8>, 0>, 8, 1>)
  %f_to = axis.factor %to0 : !axis.shape_axis<tensor<8xi8>, 0> <8, 1>
  %rhs0 = axis.product (%f_to : !axis.axis_factor<!axis.shape_axis<tensor<8xi8>, 0>, 8, 1>)

  // "lo" atom's out role: Replicate, via rep2 -> lo.
  %f_rep = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %lhs1 = axis.product (%f_rep : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %lo_out2 = axis.factor %p0 : !distributed.physical_comm_axis<4, 1> <2, 1>
  %rhs1 = axis.product (%lo_out2 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 2, 1>)

  // "hi" atom: self Mesh -> Mesh map (no-op, untouched).
  %hi_a = axis.factor %p0 : !distributed.physical_comm_axis<4, 1> <2, 2>
  %lhs2 = axis.product (%hi_a : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 2, 2>)
  %hi_b = axis.factor %p0 : !distributed.physical_comm_axis<4, 1> <2, 2>
  %rhs2 = axis.product (%hi_b : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 2, 2>)

  %map = axis.map %lhs0, %lhs1, %lhs2 to %rhs0, %rhs1, %rhs2 : [!axis.factor_group<8>, !axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<8>, !axis.factor_group<2>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<8xi8>
  %h = distributed.Collective %in : tensor<8xi8> on %mesh_in : !axis.factor_group<4> to tensor<8xi8> on %mesh_out : !axis.factor_group<4> reduces (%red0 : !axis.factor_group<2>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<i8>, %rhs_v: tensor<i8>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<i8>
    stablehlo.return %r : tensor<i8>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<8xi8>> -> tensor<8xi8>
}

// -----

// Multi-atom permute: a genuine 3-cycle across three physical axes (p0 -> p1
// -> p2 -> p0), not a single swapping pair, so buildPermuteStep has to
// re-derive which atom's digit feeds which (a flat StepAtom list alone
// doesn't say) via a second normalizeCollective call. Hand-derived expected
// pairs (device id = c0*4 + c1*2 + c2, target's (c0,c1,c2) = source's
// (c2,c0,c1)): 0->0, 1->4, 2->1, 3->5, 4->2, 5->6, 6->3, 7->7 -- two fixed
// points (all digits equal) and one genuine 3-cycle in the permutation
// (1 -> 4 -> 2 -> 1), confirming this isn't just three independent swaps.
// CHECK-LABEL: module @permute3 {
// CHECK: %[[IN:.+]] = tensor.empty() : tensor<8xf32>
// CHECK: %[[START:.+]] = "stablehlo.async_start"(%[[IN]]) (
// CHECK: ^bb0(%[[ARG:.+]]: tensor<8xf32>):
// CHECK:   "stablehlo.collective_permute"(%[[ARG]])
// CHECK-SAME: source_target_pairs = dense<{{\[}}[0, 0], [1, 4], [2, 1], [3, 5], [4, 2], [5, 6], [6, 3], [7, 7]]> : tensor<8x2xi64>
// CHECK: }) {distributed.primitive_cost = {isolated_duration = {{.*}} : f64, kind = "permute", latency = {{.*}} : f64, payload_in = 32 : i64, payload_out = 32 : i64, rho = [{{.*}}, {{.*}}, {{.*}}], transfer_time = {{.*}} : f64, volume = [{{.*}}, {{.*}}, {{.*}}]}} : (tensor<8xf32>) -> !stablehlo.future<tensor<8xf32>>
// CHECK: %{{.+}} = "stablehlo.async_done"(%[[START]]) : (!stablehlo.future<tensor<8xf32>>) -> tensor<8xf32>
// CHECK-NOT: distributed.Collective
// CHECK-NOT: distributed.Await
module @permute3 {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0, %p1, %p2 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %ti0 = axis.getaxis tensor<8xf32> 0
  %to0 = axis.getaxis tensor<8xf32> 0

  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %f2 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f3 = axis.factor %p2 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>, %f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f4 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %f5 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f6 = axis.factor %p2 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f4 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>, %f5 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f6 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)

  %f7 = axis.factor %ti0 : !axis.shape_axis<tensor<8xf32>, 0> <8, 1>
  %lhs0 = axis.product (%f7 : !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 8, 1>)
  %f8 = axis.factor %to0 : !axis.shape_axis<tensor<8xf32>, 0> <8, 1>
  %rhs0 = axis.product (%f8 : !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 8, 1>)

  // p0 -> p1, p1 -> p2, p2 -> p0: a genuine 3-cycle, not a pairwise swap.
  %f9 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %lhs1 = axis.product (%f9 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>)
  %f10 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %rhs1 = axis.product (%f10 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)

  %f11 = axis.factor %p1 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %lhs2 = axis.product (%f11 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %f12 = axis.factor %p2 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs2 = axis.product (%f12 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)

  %f13 = axis.factor %p2 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %lhs3 = axis.product (%f13 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f14 = axis.factor %p0 : !distributed.physical_comm_axis<2, 4> <2, 1>
  %rhs3 = axis.product (%f14 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>)

  %map = axis.map %lhs0, %lhs1, %lhs2, %lhs3 to %rhs0, %rhs1, %rhs2, %rhs3 : [!axis.factor_group<8>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<8>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>]

  %in = tensor.empty() : tensor<8xf32>
  %h = distributed.Collective %in : tensor<8xf32> on %mesh_in : !axis.factor_group<8> to tensor<8xf32> on %mesh_out : !axis.factor_group<8> reduces () maps %map : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<8xf32>> -> tensor<8xf32>
}
