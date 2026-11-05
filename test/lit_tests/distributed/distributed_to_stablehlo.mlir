// RUN: enzymexlamlir-opt --split-input-file --distributed-to-stablehlo %s | FileCheck %s --implicit-check-not=distributed --implicit-check-not=axis --implicit-check-not=sdy

// distributed-to-stablehlo turns a finalized schedule into a global-shaped
// public @main over the private per-device @main_per_device. Each argument's
// and result's declared sharding becomes its tiled HloSharding and the
// matching @Sharding/@SPMDFullToShardShape or @SPMDShardToFullShape custom
// calls, and its boundary casts are erased. No
// distributed, axis or sdy op or attribute is left in the output (the
// implicit CHECK-NOTs), including distributed.latency and
// distributed.primitive_cost.

// A finalized dot whose contraction dimension is split, with its all-reduce
// (shardy_baseline_dot_contraction on a 2x2 mesh). Device d is 2 * a + b for
// the coordinates a and b on the axes of id stride 2 and 1:
// - %x: columns over a, replicated over b: [1,2,2] in device order;
// - %w: rows over a, columns over b: [2,2];
// - result: columns over b, replicated over a, which needs the transpose.
// CHECK-LABEL: module attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32}
// CHECK-LABEL: func.func @main(
// CHECK-SAME: %[[X:.*]]: tensor<8x16xf32> {mhlo.sharding = "{devices=[1,2,2]<=[4] last_tile_dim_replicate}"}
// CHECK-SAME: %[[W:.*]]: tensor<16x8xf32> {mhlo.sharding = "{devices=[2,2]<=[4]}"}
// CHECK-SAME: -> (tensor<8x8xf32> {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"})
// CHECK: %[[X_ANNOTATED:.*]] = stablehlo.custom_call @Sharding(%[[X]]) {{.*}}mhlo.sharding = "{devices=[1,2,2]<=[4] last_tile_dim_replicate}"{{.*}} -> tensor<8x16xf32>
// CHECK: %[[X_LOCAL:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[X_ANNOTATED]]) {{.*}}mhlo.sharding = "{manual}"{{.*}} -> tensor<8x8xf32>
// CHECK: %[[W_ANNOTATED:.*]] = stablehlo.custom_call @Sharding(%[[W]]) {{.*}}mhlo.sharding = "{devices=[2,2]<=[4]}"{{.*}} -> tensor<16x8xf32>
// CHECK: %[[W_LOCAL:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[W_ANNOTATED]]) {{.*}}mhlo.sharding = "{manual}"{{.*}} -> tensor<8x4xf32>
// CHECK: %[[RESULT:.*]] = call @main_per_device(%[[X_LOCAL]], %[[W_LOCAL]])
// CHECK: %[[RESULT_MANUAL:.*]] = stablehlo.custom_call @Sharding(%[[RESULT]]) {{.*}}mhlo.sharding = "{manual}"{{.*}} -> tensor<8x4xf32>
// CHECK: %[[RESULT_GLOBAL:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[RESULT_MANUAL]]) {{.*}}mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"{{.*}} -> tensor<8x8xf32>
// CHECK: return %[[RESULT_GLOBAL]]
// CHECK-LABEL: func.func private @main_per_device(
// CHECK-SAME: %[[X_ARG:.*]]: tensor<8x8xf32>, %[[W_ARG:.*]]: tensor<8x4xf32>) -> tensor<8x4xf32>
// CHECK: stablehlo.dot_general %[[X_ARG]], %[[W_ARG]]
// CHECK: stablehlo.all_reduce
module attributes {distributed.shardy_baseline} {
  %0 = axis.factor %29#0 : !distributed.physical_comm_axis<2, 2><2, 1>
  %1 = axis.factor %29#1 : !distributed.physical_comm_axis<2, 1><2, 1>
  %2 = axis.product (%0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %3 = axis.product (%0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %4 = axis.factor %14 : !axis.shape_axis<tensor<8x4xf32>, 0><8, 1>
  %5 = axis.product (%4 : !axis.axis_factor<!axis.shape_axis<tensor<8x4xf32>, 0>, 8, 1>)
  %6 = axis.product (%1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %7 = axis.factor %15 : !axis.shape_axis<tensor<8x4xf32>, 1><4, 1>
  %8 = axis.product (%7 : !axis.axis_factor<!axis.shape_axis<tensor<8x4xf32>, 1>, 4, 1>)
  %9 = distributed.ReplicationAxis 2 : <2>
  %10 = axis.factor %9 : !distributed.replication_axis<2><2, 1>
  %11 = axis.product (%10 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %12 = axis.map %5, %6, %8, %11 to %5, %6, %8, %3 : [!axis.factor_group<8>, !axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<8>, !axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<2>]
  %13 = axis.product (%25 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %14 = axis.getaxis tensor<8x4xf32> 0
  %15 = axis.getaxis tensor<8x4xf32> 1
  %16 = axis.product ()
  %17 = axis.product (%19 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %18 = axis.product (%24 : !axis.axis_factor<!distributed.device_local_axis<8>, 8, 1>)
  %19 = axis.factor %29#0 : !distributed.physical_comm_axis<2, 2><2, 1>
  %20 = distributed.DeviceLocalAxis 8 : <8>
  %21 = axis.factor %20 : !distributed.device_local_axis<8><8, 1>
  %22 = axis.product (%21 : !axis.axis_factor<!distributed.device_local_axis<8>, 8, 1>)
  %23 = distributed.DeviceLocalAxis 8 : <8>
  %24 = axis.factor %23 : !distributed.device_local_axis<8><8, 1>
  %25 = axis.factor %29#1 : !distributed.physical_comm_axis<2, 1><2, 1>
  %26 = distributed.DeviceLocalAxis 4 : <4>
  %27 = axis.factor %26 : !distributed.device_local_axis<4><4, 1>
  %28 = axis.product (%27 : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>)
  %29:2 = distributed.GetPhysicalMeshAxes @m : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  "distributed.DistributedFunction"(%18, %17, %22, %13, %28) <{arg_attrs = [{sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}]>}, {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}], argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0], [1, 2]] : unreduced_axes = []>, <dim_partitioning_axes = [[1, 2], [3, 4]] : unreduced_axes = []>]>, function_type = (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>, output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0], [3, 4]] : unreduced_axes = []>]>, res_attrs = [{sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>}], sym_name = "main"}> ({
  ^bb0(%arg0: tensor<8x16xf32>, %arg1: tensor<16x8xf32>):
    %30 = distributed.CastGlobalToLocal %arg0 axes (%16 : !axis.factor_group<1>, %17 : !axis.factor_group<2>) : tensor<8x16xf32> -> tensor<8x8xf32>
    %31 = distributed.CastGlobalToLocal %arg1 axes (%17 : !axis.factor_group<2>, %13 : !axis.factor_group<2>) : tensor<16x8xf32> -> tensor<8x4xf32>
    %32 = stablehlo.dot_general %30, %31, contracting_dims = [1] x [0] {distributed.latency = 0.17066666666666666 : f64} : (tensor<8x8xf32>, tensor<8x4xf32>) -> tensor<8x4xf32>
    %33 = "stablehlo.all_reduce"(%32) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>, use_global_device_ids}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %35 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %35 : tensor<f32>
    }) {distributed.primitive_cost = {isolated_duration = 6.003000e+03 : f64, kind = "all-reduce", latency = 6.000000e+03 : f64, payload_in = 128 : i64, payload_out = 128 : i64, rho = [1.000000e+00, 0.000000e+00], transfer_time = 2.560000e+00 : f64, volume = [1.280000e+02, 0.000000e+00]}} : (tensor<8x4xf32>) -> tensor<8x4xf32>
    %34 = distributed.CastLocalToGlobal %33 axes (%16 : !axis.factor_group<1>, %13 : !axis.factor_group<2>) : tensor<8x4xf32> -> tensor<8x8xf32>
    distributed.DistributedYield (%34 : tensor<8x8xf32>)
  }) : (!axis.factor_group<8>, !axis.factor_group<2>, !axis.factor_group<8>, !axis.factor_group<2>, !axis.factor_group<4>) -> ()
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]
}

// -----

// A callee with its own boundary casts (shardy_baseline_call_boundary on a
// 2x4 mesh): it becomes a private func.func at its local types without a
// sharding, and the call a func.call.
// CHECK-LABEL: module attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32}
// CHECK-LABEL: func.func private @f(
// CHECK-SAME: %{{.*}}: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-LABEL: func.func @main(
// CHECK-SAME: tensor<8x16xf32> {mhlo.sharding = "{devices=[2,1,4]<=[8] last_tile_dim_replicate}"}
// CHECK-SAME: -> (tensor<8x16xf32> {mhlo.sharding = "{devices=[2,4]<=[8]}"})
// CHECK: call @main_per_device
// CHECK-LABEL: func.func private @main_per_device(
// CHECK-SAME: tensor<4x16xf32>) -> tensor<4x4xf32>
// CHECK: %[[CALLEE_RESULT:.*]] = call @f(%{{.*}}) : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: return %[[CALLEE_RESULT]]
module attributes {distributed.shardy_baseline} {
  %0 = axis.factor %37#0 : !distributed.physical_comm_axis<2, 4><2, 1>
  %1 = axis.factor %37#1 : !distributed.physical_comm_axis<4, 1><4, 1>
  %2 = axis.product (%0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>, %1 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %3 = axis.product (%0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>)
  %4 = axis.factor %20 : !axis.shape_axis<tensor<4x16xf32>, 0><4, 1>
  %5 = axis.factor %22 : !axis.shape_axis<tensor<4x4xf32>, 0><4, 1>
  %6 = axis.product (%4 : !axis.axis_factor<!axis.shape_axis<tensor<4x16xf32>, 0>, 4, 1>)
  %7 = axis.product (%5 : !axis.axis_factor<!axis.shape_axis<tensor<4x4xf32>, 0>, 4, 1>)
  %8 = axis.factor %21 : !axis.shape_axis<tensor<4x16xf32>, 1><4, 4>
  %9 = axis.product (%8 : !axis.axis_factor<!axis.shape_axis<tensor<4x16xf32>, 1>, 4, 4>)
  %10 = axis.product (%1 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %11 = axis.factor %21 : !axis.shape_axis<tensor<4x16xf32>, 1><4, 1>
  %12 = axis.factor %23 : !axis.shape_axis<tensor<4x4xf32>, 1><4, 1>
  %13 = axis.product (%11 : !axis.axis_factor<!axis.shape_axis<tensor<4x16xf32>, 1>, 4, 1>)
  %14 = axis.product (%12 : !axis.axis_factor<!axis.shape_axis<tensor<4x4xf32>, 1>, 4, 1>)
  %15 = distributed.ReplicationAxis 4 : <4>
  %16 = axis.factor %15 : !distributed.replication_axis<4><4, 1>
  %17 = axis.product (%16 : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
  %18 = axis.map %3, %6, %9, %13, %10 to %3, %7, %10, %14, %17 : [!axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<4>, !axis.factor_group<4>, !axis.factor_group<4>] [!axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<4>, !axis.factor_group<4>, !axis.factor_group<4>]
  %19 = axis.product (%36 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>)
  %20 = axis.getaxis tensor<4x16xf32> 0
  %21 = axis.getaxis tensor<4x16xf32> 1
  %22 = axis.getaxis tensor<4x4xf32> 0
  %23 = axis.getaxis tensor<4x4xf32> 1
  %24 = axis.product (%35 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %25 = axis.product ()
  %26 = axis.product (%28 : !axis.axis_factor<!distributed.device_local_axis<16>, 16, 1>)
  %27 = distributed.DeviceLocalAxis 16 : <16>
  %28 = axis.factor %27 : !distributed.device_local_axis<16><16, 1>
  %29 = distributed.DeviceLocalAxis 4 : <4>
  %30 = axis.factor %29 : !distributed.device_local_axis<4><4, 1>
  %31 = axis.product (%30 : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>)
  %32 = distributed.DeviceLocalAxis 4 : <4>
  %33 = axis.factor %32 : !distributed.device_local_axis<4><4, 1>
  %34 = axis.product (%33 : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>)
  %35 = axis.factor %37#1 : !distributed.physical_comm_axis<4, 1><4, 1>
  %36 = axis.factor %37#0 : !distributed.physical_comm_axis<2, 4><2, 1>
  %37:2 = distributed.GetPhysicalMeshAxes @m : !distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<4, 1>
  "distributed.DistributedFunction"(%19, %34, %24, %31) <{arg_attrs = [{sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"tp"}]>}], argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0, 1], [2, 3]] : unreduced_axes = []>]>, function_type = (tensor<8x16xf32>) -> tensor<8x16xf32>, output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0, 1], [2, 3]] : unreduced_axes = []>]>, res_attrs = [{sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"tp"}]>}], sym_name = "f", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<8x16xf32>):
    %38 = distributed.CastGlobalToLocal %arg0 axes (%19 : !axis.factor_group<2>, %24 : !axis.factor_group<4>) : tensor<8x16xf32> -> tensor<4x4xf32>
    %39 = stablehlo.negate %38 {distributed.latency = 0.042666666666666665 : f64} : tensor<4x4xf32>
    %40 = distributed.CastLocalToGlobal %39 axes (%19 : !axis.factor_group<2>, %24 : !axis.factor_group<4>) : tensor<4x4xf32> -> tensor<8x16xf32>
    distributed.DistributedYield (%40 : tensor<8x16xf32>)
  }) : (!axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<4>, !axis.factor_group<4>) -> ()
  "distributed.DistributedFunction"(%19, %34, %26, %24, %31) <{arg_attrs = [{sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {}]>}], argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0, 1], [2]] : unreduced_axes = []>]>, function_type = (tensor<8x16xf32>) -> tensor<8x16xf32>, output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0, 1], [3, 4]] : unreduced_axes = []>]>, res_attrs = [{sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"tp"}]>}], sym_name = "main"}> ({
  ^bb0(%arg0: tensor<8x16xf32>):
    %38 = distributed.CastGlobalToLocal %arg0 axes (%19 : !axis.factor_group<2>, %25 : !axis.factor_group<1>) : tensor<8x16xf32> -> tensor<4x16xf32>
    %39 = stablehlo.reshape %38 : (tensor<4x16xf32>) -> tensor<4x4x4xf32>
    %40 = stablehlo.partition_id : tensor<ui32>
    %41 = stablehlo.convert %40 : (tensor<ui32>) -> tensor<i64>
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_0 = stablehlo.constant dense<4> : tensor<i64>
    %42 = stablehlo.divide %41, %c : tensor<i64>
    %43 = stablehlo.remainder %42, %c_0 : tensor<i64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %44 = stablehlo.multiply %43, %c_1 : tensor<i64>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %45 = stablehlo.dynamic_slice %39, %c_2, %44, %c_2, sizes = [4, 1, 4] {distributed.primitive_cost = {isolated_duration = 0.000000e+00 : f64, kind = "local-slice", latency = 0.000000e+00 : f64, payload_in = 256 : i64, payload_out = 64 : i64, rho = [0.000000e+00, 0.000000e+00], transfer_time = 0.000000e+00 : f64, volume = [0.000000e+00, 0.000000e+00]}} : (tensor<4x4x4xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<4x1x4xf32>
    %46 = stablehlo.reshape %45 : (tensor<4x1x4xf32>) -> tensor<4x4xf32>
    %47 = distributed.DistributedCall @f (%46 : tensor<4x4xf32>) <[<dim_partitioning_axes = [[0, 1], [2, 3]] : unreduced_axes = []>]>
      -> (tensor<4x4xf32>) <[<dim_partitioning_axes = [[0, 1], [2, 3]] : unreduced_axes = []>]>
      axes (%19 : !axis.factor_group<2>, %25 : !axis.factor_group<1>, %24 : !axis.factor_group<4>, %25 : !axis.factor_group<1>)
    %48 = distributed.CastLocalToGlobal %47 axes (%19 : !axis.factor_group<2>, %24 : !axis.factor_group<4>) : tensor<4x4xf32> -> tensor<8x16xf32>
    distributed.DistributedYield (%48 : tensor<8x16xf32>)
  }) : (!axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<16>, !axis.factor_group<4>, !axis.factor_group<4>) -> ()
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<4, 1>]
}

// -----

// An argument and a result declared replicated keep their type and get
// {replicated}, beside a sharded argument and result.
// CHECK-LABEL: func.func @main(
// CHECK-SAME: tensor<8x4xf32> {mhlo.sharding = "{devices=[2,1,2]<=[4] last_tile_dim_replicate}"}
// CHECK-SAME: %[[BIAS:.*]]: tensor<4xf32> {mhlo.sharding = "{replicated}"}
// CHECK-SAME: -> (tensor<8x4xf32> {mhlo.sharding = "{devices=[2,1,2]<=[4] last_tile_dim_replicate}"}, tensor<4xf32> {mhlo.sharding = "{replicated}"})
// CHECK: %[[BIAS_ANNOTATED:.*]] = stablehlo.custom_call @Sharding(%[[BIAS]]) {{.*}}mhlo.sharding = "{replicated}"{{.*}} -> tensor<4xf32>
// CHECK: %[[BIAS_LOCAL:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[BIAS_ANNOTATED]]) {{.*}}mhlo.sharding = "{manual}"{{.*}} -> tensor<4xf32>
// CHECK: %[[RESULTS:.*]]:2 = call @main_per_device(%{{.*}}, %[[BIAS_LOCAL]])
// CHECK: %[[NEGATED_MANUAL:.*]] = stablehlo.custom_call @Sharding(%[[RESULTS]]#1) {{.*}}mhlo.sharding = "{manual}"
// CHECK: stablehlo.custom_call @SPMDShardToFullShape(%[[NEGATED_MANUAL]]) {{.*}}mhlo.sharding = "{replicated}"{{.*}} -> tensor<4xf32>
// CHECK-LABEL: func.func private @main_per_device(
// CHECK-SAME: tensor<4x4xf32>, %{{.*}}: tensor<4xf32>) -> (tensor<4x4xf32>, tensor<4xf32>)
module attributes {distributed.shardy_baseline} {
  %0 = axis.product ()
  %1 = axis.product (%5 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %2 = distributed.DeviceLocalAxis 4 : <4>
  %3 = axis.factor %2 : !distributed.device_local_axis<4><4, 1>
  %4 = axis.product (%3 : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>)
  %5 = axis.factor %9#0 : !distributed.physical_comm_axis<2, 2><2, 1>
  %6 = distributed.DeviceLocalAxis 4 : <4>
  %7 = axis.factor %6 : !distributed.device_local_axis<4><4, 1>
  %8 = axis.product (%7 : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>)
  %9:2 = distributed.GetPhysicalMeshAxes @m : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  "distributed.DistributedFunction"(%1, %8, %4) <{arg_attrs = [{sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}, {sdy.sharding = #sdy.sharding<@mesh, [{}]>}], argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0, 1], [2]] : unreduced_axes = []>, <dim_partitioning_axes = [[2]] : unreduced_axes = []>]>, function_type = (tensor<8x4xf32>, tensor<4xf32>) -> (tensor<8x4xf32>, tensor<4xf32>), output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0, 1], [2]] : unreduced_axes = []>, <dim_partitioning_axes = [[2]] : unreduced_axes = []>]>, res_attrs = [{sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}, {sdy.sharding = #sdy.sharding<@mesh, [{}]>}], sym_name = "main"}> ({
  ^bb0(%arg0: tensor<8x4xf32>, %arg1: tensor<4xf32>):
    %10 = distributed.CastGlobalToLocal %arg0 axes (%1 : !axis.factor_group<2>, %0 : !axis.factor_group<1>) : tensor<8x4xf32> -> tensor<4x4xf32>
    %11 = stablehlo.broadcast_in_dim %arg1, dims = [1] {distributed.latency = 0.026666666666666668 : f64} : (tensor<4xf32>) -> tensor<4x4xf32>
    %12 = stablehlo.add %10, %11 {distributed.latency = 6.400000e-02 : f64} : tensor<4x4xf32>
    %13 = stablehlo.negate %arg1 {distributed.latency = 0.010666666666666666 : f64} : tensor<4xf32>
    %14 = distributed.CastLocalToGlobal %12 axes (%1 : !axis.factor_group<2>, %0 : !axis.factor_group<1>) : tensor<4x4xf32> -> tensor<8x4xf32>
    distributed.DistributedYield (%14 : tensor<8x4xf32>, %13 : tensor<4xf32>)
  }) : (!axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<4>) -> ()
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]
}


// -----

// HloShardings on a 2-axis mesh of 16 devices, with id strides 4 and 1, so
// device d has coordinates d / 4 and d % 4:
// - %arg0 (llama's batch at 16 devices): dimension 0 over the stride-1 axis
//   entire (major), then the high digit of the stride-4 axis (minor), so the
//   tile is (d % 4) * 2 + d / 8 and the replica (d / 4) % 2;
// - %arg1: dimension 0 over the stride-1 axis only, so the tile is d % 4 and
//   the replica d / 4, which needs the transpose.
// The sdy.mesh is erased with the other metadata.
// CHECK-LABEL: module attributes {mhlo.num_partitions = 16 : i32, mhlo.num_replicas = 1 : i32}
// CHECK-LABEL: func.func @main(
// CHECK-SAME: tensor<8x288xf32> {mhlo.sharding = "{devices=[8,1,2]<=[2,2,4]T(2,0,1) last_tile_dim_replicate}"}
// CHECK-SAME: tensor<8x288xf32> {mhlo.sharding = "{devices=[4,1,4]<=[4,4]T(1,0) last_tile_dim_replicate}"}
// CHECK-LABEL: func.func private @main_per_device(
// CHECK-SAME: tensor<1x288xf32>, %{{.*}}: tensor<2x288xf32>) -> tensor<1x288xf32>
module {
  sdy.mesh @mesh = <["a"=4, "b"=4]>
  %axes:2 = distributed.GetPhysicalMeshAxes @mesh16 : !distributed.physical_comm_axis<4, 4>, !distributed.physical_comm_axis<4, 1>
  %minor = axis.factor %axes#1 : !distributed.physical_comm_axis<4, 1><4, 1>
  %major_high = axis.factor %axes#0 : !distributed.physical_comm_axis<4, 4><2, 2>
  %batch = axis.product (%minor : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>, %major_high : !axis.axis_factor<!distributed.physical_comm_axis<4, 4>, 2, 2>)
  %minor_only = axis.product (%minor : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %unsplit = axis.product ()
  "distributed.DistributedFunction"(%batch, %minor_only, %unsplit) <{argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0], [2]] : unreduced_axes = []>, <dim_partitioning_axes = [[1], [2]] : unreduced_axes = []>]>, function_type = (tensor<8x288xf32>, tensor<8x288xf32>) -> tensor<8x288xf32>, output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0], [2]] : unreduced_axes = []>]>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<8x288xf32>, %arg1: tensor<8x288xf32>):
    %0 = distributed.CastGlobalToLocal %arg0 axes (%batch : !axis.factor_group<8>, %unsplit : !axis.factor_group<1>) : tensor<8x288xf32> -> tensor<1x288xf32>
    %1 = distributed.CastGlobalToLocal %arg1 axes (%minor_only : !axis.factor_group<4>, %unsplit : !axis.factor_group<1>) : tensor<8x288xf32> -> tensor<2x288xf32>
    %2 = distributed.CastLocalToGlobal %0 axes (%batch : !axis.factor_group<8>, %unsplit : !axis.factor_group<1>) : tensor<1x288xf32> -> tensor<8x288xf32>
    distributed.DistributedYield (%2 : tensor<8x288xf32>)
  }) : (!axis.factor_group<8>, !axis.factor_group<4>, !axis.factor_group<1>) -> ()
  distributed.PhysicalMesh @mesh16 device_target "cpu" axes [!distributed.physical_comm_axis<4, 4>, !distributed.physical_comm_axis<4, 1>]
}

// -----

// An entry result unreduced only over an extent-1 group holds the whole sum,
// so it exports. Its jax.result_info is kept beside the sharding.
// CHECK-LABEL: func.func @main(
// CHECK-SAME: -> (tensor<4xf32> {jax.result_info = "result", mhlo.sharding = "{replicated}"})
module {
  %axes:2 = distributed.GetPhysicalMeshAxes @m : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %unsplit = axis.product ()
  "distributed.DistributedFunction"(%unsplit) <{argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>, function_type = (tensor<4xf32>) -> tensor<4xf32>, output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = [0]>]>, res_attrs = [{jax.result_info = "result"}], sym_name = "main"}> ({
  ^bb0(%arg0: tensor<4xf32>):
    distributed.DistributedYield (%arg0 : tensor<4xf32>)
  }) : (!axis.factor_group<1>) -> ()
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]
}

// -----

// A value that splits no dimension is {replicated}, not a tiling into ones.
// An extent-1 factor splits nothing, whatever its axis.
// CHECK-LABEL: func.func @main(%{{.*}}: tensor<4x4xf32> {mhlo.sharding = "{replicated}"})
module {
  %axes:2 = distributed.GetPhysicalMeshAxes @m : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %one = distributed.ReplicationAxis 1 : <1>
  %one_factor = axis.factor %one : !distributed.replication_axis<1><1, 1>
  %over_one = axis.product (%one_factor : !axis.axis_factor<!distributed.replication_axis<1>, 1, 1>)
  %unsplit = axis.product ()
  "distributed.DistributedFunction"(%unsplit, %over_one) <{argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0], [1]] : unreduced_axes = []>]>, function_type = (tensor<4x4xf32>) -> tensor<4x4xf32>, output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[], []] : unreduced_axes = []>]>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<4x4xf32>):
    %0 = distributed.CastGlobalToLocal %arg0 axes (%unsplit : !axis.factor_group<1>, %over_one : !axis.factor_group<1>) : tensor<4x4xf32> -> tensor<4x4xf32>
    distributed.DistributedYield (%0 : tensor<4x4xf32>)
  }) : (!axis.factor_group<1>, !axis.factor_group<1>) -> ()
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]
}

// -----

// A callee whose argument passes straight through, converted from
// `@f(%a [{"data"}, {"tp"}]) { return %a }` on a 2x4 mesh, has no boundary
// casts left after canonicalization. Its declared shardings still make it
// local, so it matches the call.
// CHECK-LABEL: func.func private @f(
// CHECK-SAME: %[[A:.*]]: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT: return %[[A]] : tensor<4x4xf32>
// CHECK-LABEL: func.func @main(
// CHECK-SAME: tensor<8x16xf32> {mhlo.sharding = "{devices=[2,4]<=[8]}"}
// CHECK-LABEL: func.func private @main_per_device(
// CHECK: call @f(%{{.*}}) : (tensor<4x4xf32>) -> tensor<4x4xf32>
module attributes {distributed.shardy_baseline} {
  %0 = axis.product (%7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 4>, 2, 1>)
  %1 = axis.product (%6 : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %2 = axis.product ()
  %3 = distributed.DeviceLocalAxis 4 : <4>
  %4 = axis.factor %3 : !distributed.device_local_axis<4><4, 1>
  %5 = axis.product (%4 : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>)
  %6 = axis.factor %11#1 : !distributed.physical_comm_axis<4, 1><4, 1>
  %7 = axis.factor %11#0 : !distributed.physical_comm_axis<2, 4><2, 1>
  %8 = distributed.DeviceLocalAxis 4 : <4>
  %9 = axis.factor %8 : !distributed.device_local_axis<4><4, 1>
  %10 = axis.product (%9 : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>)
  %11:2 = distributed.GetPhysicalMeshAxes @m : !distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<4, 1>
  "distributed.DistributedFunction"(%0, %5, %1, %10) <{arg_attrs = [{sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"tp"}]>}], argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0, 1], [2, 3]] : unreduced_axes = []>]>, function_type = (tensor<8x16xf32>) -> tensor<8x16xf32>, output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0, 1], [2, 3]] : unreduced_axes = []>]>, res_attrs = [{sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"tp"}]>}], sym_name = "f", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<8x16xf32>):
    distributed.DistributedYield (%arg0 : tensor<8x16xf32>)
  }) : (!axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<4>, !axis.factor_group<4>) -> ()
  "distributed.DistributedFunction"(%0, %5, %1, %10) <{arg_attrs = [{sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"tp"}]>}], argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0, 1], [2, 3]] : unreduced_axes = []>]>, function_type = (tensor<8x16xf32>) -> tensor<8x16xf32>, output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0, 1], [2, 3]] : unreduced_axes = []>]>, res_attrs = [{sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"tp"}]>}], sym_name = "main"}> ({
  ^bb0(%arg0: tensor<8x16xf32>):
    %12 = distributed.CastGlobalToLocal %arg0 axes (%0 : !axis.factor_group<2>, %1 : !axis.factor_group<4>) : tensor<8x16xf32> -> tensor<4x4xf32>
    %13 = distributed.DistributedCall @f (%12 : tensor<4x4xf32>) <[<dim_partitioning_axes = [[0, 1], [2, 3]] : unreduced_axes = []>]>
      -> (tensor<4x4xf32>) <[<dim_partitioning_axes = [[0, 1], [2, 3]] : unreduced_axes = []>]>
      axes (%0 : !axis.factor_group<2>, %2 : !axis.factor_group<1>, %1 : !axis.factor_group<4>, %2 : !axis.factor_group<1>)
    %14 = stablehlo.negate %13 {distributed.latency = 0.042666666666666665 : f64} : tensor<4x4xf32>
    %15 = distributed.CastLocalToGlobal %14 axes (%0 : !axis.factor_group<2>, %1 : !axis.factor_group<4>) : tensor<4x4xf32> -> tensor<8x16xf32>
    distributed.DistributedYield (%15 : tensor<8x16xf32>)
  }) : (!axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<4>, !axis.factor_group<4>) -> ()
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<4, 1>]
}


// -----

// A sharded argument also returned directly (`main(x [{"a"}, {}]) -> (x,
// negate(x))`, whose first result's casts canonicalization folded away): the
// result keeps the argument's local value and layout.
// CHECK-LABEL: func.func @main(
// CHECK-SAME: tensor<8x8xf32> {mhlo.sharding = "{devices=[2,1,2]<=[4] last_tile_dim_replicate}"}
// CHECK-SAME: -> (tensor<8x8xf32> {mhlo.sharding = "{devices=[2,1,2]<=[4] last_tile_dim_replicate}"}, tensor<8x8xf32> {mhlo.sharding = "{devices=[2,1,2]<=[4] last_tile_dim_replicate}"})
// CHECK-LABEL: func.func private @main_per_device(
// CHECK-SAME: %[[X:.*]]: tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK: %[[NEGATED:.*]] = stablehlo.negate %[[X]]
// CHECK: return %[[X]], %[[NEGATED]]
module attributes {distributed.shardy_baseline} {
  %0 = axis.product (%8 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %1 = axis.product ()
  %2 = distributed.DeviceLocalAxis 8 : <8>
  %3 = axis.factor %2 : !distributed.device_local_axis<8><8, 1>
  %4 = axis.product (%3 : !axis.axis_factor<!distributed.device_local_axis<8>, 8, 1>)
  %5 = distributed.DeviceLocalAxis 4 : <4>
  %6 = axis.factor %5 : !distributed.device_local_axis<4><4, 1>
  %7 = axis.product (%6 : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>)
  %8 = axis.factor %9#0 : !distributed.physical_comm_axis<2, 2><2, 1>
  %9:2 = distributed.GetPhysicalMeshAxes @m : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  "distributed.DistributedFunction"(%0, %7, %4) <{arg_attrs = [{sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}], argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0, 1], [2]] : unreduced_axes = []>]>, function_type = (tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>), output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0, 1], [2]] : unreduced_axes = []>, <dim_partitioning_axes = [[0, 1], [2]] : unreduced_axes = []>]>, res_attrs = [{sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}, {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}], sym_name = "main"}> ({
  ^bb0(%arg0: tensor<8x8xf32>):
    %10 = distributed.CastGlobalToLocal %arg0 axes (%0 : !axis.factor_group<2>, %1 : !axis.factor_group<1>) : tensor<8x8xf32> -> tensor<4x8xf32>
    %11 = stablehlo.negate %10 {distributed.latency = 0.08533333333333333 : f64} : tensor<4x8xf32>
    %12 = distributed.CastLocalToGlobal %11 axes (%0 : !axis.factor_group<2>, %1 : !axis.factor_group<1>) : tensor<4x8xf32> -> tensor<8x8xf32>
    distributed.DistributedYield (%arg0 : tensor<8x8xf32>, %12 : tensor<8x8xf32>)
  }) : (!axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<8>) -> ()
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]
}


// -----

// An unused sharded argument has no cast, but keeps its declared sharding and
// local type.
// CHECK-LABEL: func.func @main(
// CHECK-SAME: tensor<8x8xf32> {mhlo.sharding = "{devices=[2,1,2]<=[4] last_tile_dim_replicate}"}
// CHECK-SAME: %[[UNUSED:.*]]: tensor<8x8xf32> {mhlo.sharding = "{devices=[2,1,2]<=[4] last_tile_dim_replicate}"}
// CHECK: %[[UNUSED_ANNOTATED:.*]] = stablehlo.custom_call @Sharding(%[[UNUSED]])
// CHECK: stablehlo.custom_call @SPMDFullToShardShape(%[[UNUSED_ANNOTATED]]) {{.*}} -> tensor<4x8xf32>
// CHECK-LABEL: func.func private @main_per_device(
// CHECK-SAME: tensor<4x8xf32>, %{{.*}}: tensor<4x8xf32>) -> tensor<4x8xf32>
module attributes {distributed.shardy_baseline} {
  %0 = axis.product (%14 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %1 = axis.product ()
  %2 = distributed.DeviceLocalAxis 8 : <8>
  %3 = axis.factor %2 : !distributed.device_local_axis<8><8, 1>
  %4 = axis.product (%3 : !axis.axis_factor<!distributed.device_local_axis<8>, 8, 1>)
  %5 = distributed.DeviceLocalAxis 8 : <8>
  %6 = axis.factor %5 : !distributed.device_local_axis<8><8, 1>
  %7 = axis.product (%6 : !axis.axis_factor<!distributed.device_local_axis<8>, 8, 1>)
  %8 = distributed.DeviceLocalAxis 4 : <4>
  %9 = axis.factor %8 : !distributed.device_local_axis<4><4, 1>
  %10 = axis.product (%9 : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>)
  %11 = distributed.DeviceLocalAxis 4 : <4>
  %12 = axis.factor %11 : !distributed.device_local_axis<4><4, 1>
  %13 = axis.product (%12 : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>)
  %14 = axis.factor %15#0 : !distributed.physical_comm_axis<2, 2><2, 1>
  %15:2 = distributed.GetPhysicalMeshAxes @m : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  "distributed.DistributedFunction"(%0, %13, %4, %0, %10, %7) <{arg_attrs = [{sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}, {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}], argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0, 1], [2]] : unreduced_axes = []>, <dim_partitioning_axes = [[3, 4], [5]] : unreduced_axes = []>]>, function_type = (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>, output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0, 1], [2]] : unreduced_axes = []>]>, res_attrs = [{sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}], sym_name = "main"}> ({
  ^bb0(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>):
    %16 = distributed.CastGlobalToLocal %arg0 axes (%0 : !axis.factor_group<2>, %1 : !axis.factor_group<1>) : tensor<8x8xf32> -> tensor<4x8xf32>
    %17 = stablehlo.negate %16 {distributed.latency = 0.08533333333333333 : f64} : tensor<4x8xf32>
    %18 = distributed.CastLocalToGlobal %17 axes (%0 : !axis.factor_group<2>, %1 : !axis.factor_group<1>) : tensor<4x8xf32> -> tensor<8x8xf32>
    distributed.DistributedYield (%18 : tensor<8x8xf32>)
  }) : (!axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<8>, !axis.factor_group<2>, !axis.factor_group<4>, !axis.factor_group<8>) -> ()
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]
}

// -----

// Relayouts around a strided argument and result (declared as [p0][local 2][p1])
// move into @main, so the per-device function sees contiguous tiles. Result #0
// has no relayout.
// CHECK-LABEL: func.func @main(
// CHECK-SAME: %[[X:.*]]: tensor<8xf32> {mhlo.sharding = "{devices=[4]<=[4]}"}
// CHECK: %[[SPLIT:.*]] = stablehlo.reshape %[[X]] : (tensor<8xf32>) -> tensor<2x2x2x1xf32>
// CHECK: %[[CANONICAL:.*]] = stablehlo.transpose %[[SPLIT]], dims = [0, 2, 1, 3]
// CHECK: %[[MERGED:.*]] = stablehlo.reshape %[[CANONICAL]] : (tensor<2x2x2x1xf32>) -> tensor<8xf32>
// CHECK: %[[ANNOTATED:.*]] = stablehlo.custom_call @Sharding(%[[MERGED]]) {{.*}}mhlo.sharding = "{devices=[4]<=[4]}"
// CHECK: %[[TILE:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[ANNOTATED]]) {{.*}} -> tensor<2xf32>
// CHECK: call @main_per_device(%[[TILE]])
// CHECK: %[[CANONICAL_OUT:.*]] = stablehlo.custom_call @SPMDShardToFullShape({{.*}}) {{.*}} -> tensor<8xf32>
// CHECK: %[[FULL:.*]] = stablehlo.custom_call @SPMDShardToFullShape({{.*}}) {{.*}} -> tensor<8xf32>
// CHECK: %[[OUT_SPLIT:.*]] = stablehlo.reshape %[[FULL]] : (tensor<8xf32>) -> tensor<2x2x2x1xf32>
// CHECK: %[[OUT_ORIGINAL:.*]] = stablehlo.transpose %[[OUT_SPLIT]], dims = [0, 2, 1, 3]
// CHECK: %[[OUT:.*]] = stablehlo.reshape %[[OUT_ORIGINAL]] : (tensor<2x2x2x1xf32>) -> tensor<8xf32>
// CHECK: return %[[CANONICAL_OUT]], %[[OUT]]
// CHECK-LABEL: func.func private @main_per_device(
// CHECK-SAME: %[[LOCAL:.*]]: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)
// CHECK-NOT: stablehlo.transpose
// CHECK: %[[NEGATED:.*]] = stablehlo.negate %[[LOCAL]]
// CHECK-NOT: stablehlo.transpose
// CHECK: return %[[NEGATED]], %[[NEGATED]]
module {
  %axes:2 = distributed.GetPhysicalMeshAxes @m : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %p0 = axis.factor %axes#0 : !distributed.physical_comm_axis<2, 2><2, 1>
  %p1 = axis.factor %axes#1 : !distributed.physical_comm_axis<2, 1><2, 1>
  %over_p0 = axis.product (%p0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %over_p1 = axis.product (%p1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %over_both = axis.product (%p0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %p1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %local = distributed.DeviceLocalAxis 2 : <2>
  %local_factor = axis.factor %local : !distributed.device_local_axis<2><2, 1>
  %over_local = axis.product (%local_factor : !axis.axis_factor<!distributed.device_local_axis<2>, 2, 1>)
  "distributed.DistributedFunction"(%over_p0, %over_local, %over_p1, %over_both) <{argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0, 1, 2]] : unreduced_axes = []>]>, function_type = (tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>), output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[3]] : unreduced_axes = []>, <dim_partitioning_axes = [[0, 1, 2]] : unreduced_axes = []>]>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<8xf32>):
    %split = stablehlo.reshape %arg0 : (tensor<8xf32>) -> tensor<2x2x2x1xf32>
    %canonical = stablehlo.transpose %split, dims = [0, 2, 1, 3] : (tensor<2x2x2x1xf32>) -> tensor<2x2x2x1xf32>
    %merged = stablehlo.reshape %canonical : (tensor<2x2x2x1xf32>) -> tensor<8xf32>
    %tile = distributed.CastGlobalToLocal %merged axes (%over_both : !axis.factor_group<4>) : tensor<8xf32> -> tensor<2xf32>
    %negated = stablehlo.negate %tile : tensor<2xf32>
    %canonical_out = distributed.CastLocalToGlobal %negated axes (%over_both : !axis.factor_group<4>) : tensor<2xf32> -> tensor<8xf32>
    %strided_out = distributed.CastLocalToGlobal %negated axes (%over_both : !axis.factor_group<4>) : tensor<2xf32> -> tensor<8xf32>
    %out_split = stablehlo.reshape %strided_out : (tensor<8xf32>) -> tensor<2x2x2x1xf32>
    %out_original = stablehlo.transpose %out_split, dims = [0, 2, 1, 3] : (tensor<2x2x2x1xf32>) -> tensor<2x2x2x1xf32>
    %out_merged = stablehlo.reshape %out_original : (tensor<2x2x2x1xf32>) -> tensor<8xf32>
    distributed.DistributedYield (%canonical_out : tensor<8xf32>, %out_merged : tensor<8xf32>)
  }) : (!axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<4>) -> ()
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]
}

// -----

// Any global-scope computation feeding a cast moves into @main, here a negation.
// CHECK-LABEL: func.func @main(
// CHECK-SAME: %[[X:.*]]: tensor<4xf32> {mhlo.sharding = "{replicated}"}
// CHECK: %[[NEGATED:.*]] = stablehlo.negate %[[X]]
// CHECK: %[[ANNOTATED:.*]] = stablehlo.custom_call @Sharding(%[[NEGATED]]) {{.*}}mhlo.sharding = "{devices=[2,2]<=[4] last_tile_dim_replicate}"
// CHECK: stablehlo.custom_call @SPMDFullToShardShape(%[[ANNOTATED]]) {{.*}} -> tensor<2xf32>
// CHECK-LABEL: func.func private @main_per_device(
// CHECK-NOT: stablehlo.negate
// CHECK: return
module {
  %axes:2 = distributed.GetPhysicalMeshAxes @m : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %major = axis.factor %axes#0 : !distributed.physical_comm_axis<2, 2><2, 1>
  %over_major = axis.product (%major : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  "distributed.DistributedFunction"(%over_major) <{argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>, function_type = (tensor<4xf32>) -> tensor<2xf32>, output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<4xf32>):
    %0 = stablehlo.negate %arg0 : tensor<4xf32>
    %1 = distributed.CastGlobalToLocal %0 axes (%over_major : !axis.factor_group<2>) : tensor<4xf32> -> tensor<2xf32>
    distributed.DistributedYield (%1 : tensor<2xf32>)
  }) : (!axis.factor_group<2>) -> ()
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]
}

// -----

// A constant read by both the relayout and the per-device code is cloned into
// @main and stays in the per-device function.
// CHECK-LABEL: func.func @main(
// CHECK: %[[ONE:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<4xf32>
// CHECK: %[[SHIFTED:.*]] = stablehlo.add %{{.*}}, %[[ONE]] : tensor<4xf32>
// CHECK: stablehlo.custom_call @Sharding(%[[SHIFTED]])
// CHECK-LABEL: func.func private @main_per_device(
// CHECK-SAME: %[[TILE:.*]]: tensor<2xf32>)
// CHECK: %[[LOCAL_ONE:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<4xf32>
// CHECK: stablehlo.slice %[[LOCAL_ONE]]
module {
  %axes:2 = distributed.GetPhysicalMeshAxes @m : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %major = axis.factor %axes#0 : !distributed.physical_comm_axis<2, 2><2, 1>
  %over_major = axis.product (%major : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  "distributed.DistributedFunction"(%over_major) <{argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>, function_type = (tensor<4xf32>) -> tensor<2xf32>, output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<4xf32>):
    %one = stablehlo.constant dense<1.0> : tensor<4xf32>
    %0 = stablehlo.add %arg0, %one : tensor<4xf32>
    %1 = distributed.CastGlobalToLocal %0 axes (%over_major : !axis.factor_group<2>) : tensor<4xf32> -> tensor<2xf32>
    %2 = stablehlo.slice %one [0:2] : (tensor<4xf32>) -> tensor<2xf32>
    %3 = stablehlo.add %1, %2 : tensor<2xf32>
    distributed.DistributedYield (%3 : tensor<2xf32>)
  }) : (!axis.factor_group<2>) -> ()
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]
}
