// RUN: enzymexlamlir-opt --cluster-distributed-kernels --canonicalize --stabilize-axis-order %s -o - | FileCheck %s

// This is the canonicalized, CSE'd post-materialization form of a sharded
// reduction. Clustering must reuse the existing <8> logical axis.
module @axis_ssa_anchor {
  %0 = axis.product %9 : !axis.axis_factor<!distributed.logical_mesh_axis<8>, 8, 1>
  %1 = axis.getaxis tensor<1xf32> 0
  %2 = axis.factor %1 : !axis.shape_axis<tensor<1xf32>, 0><1, 1>
  %3 = axis.product %9, %2 : !axis.axis_factor<!distributed.logical_mesh_axis<8>, 8, 1>, !axis.axis_factor<!axis.shape_axis<tensor<1xf32>, 0>, 1, 1>
  %4 = axis.product %9, %6 : !axis.axis_factor<!distributed.logical_mesh_axis<8>, 8, 1>, !axis.axis_factor<!distributed.logical_mesh_axis<4>, 4, 1>
  %5 = distributed.LogicalMeshAxes [4] : !distributed.logical_mesh_axis<4>
  %6 = axis.factor %5 : !distributed.logical_mesh_axis<4><4, 1>
  %7 = axis.product %6 : !axis.axis_factor<!distributed.logical_mesh_axis<4>, 4, 1>
  %8 = distributed.LogicalMeshAxes [8] : !distributed.logical_mesh_axis<8>
  %9 = axis.factor %8 : !distributed.logical_mesh_axis<8><8, 1>
  "distributed.DistributedFunction"(%0, %7) <{argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0], [1]] : unreduced_axes = []>]>, function_type = (tensor<8x4xf32>) -> tensor<8xf32>, output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<8x4xf32>):
    %10 = sdy.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.reduce(%arg0 init: %10) applies stablehlo.add across dimensions = [1] {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [])->([i]) {i=8, j=4} reduction={j}>} : (tensor<8x4xf32>, tensor<f32>) -> tensor<8xf32>
    %12 = distributed.CastGlobalToLocal %11 axes %0 : !axis.factor_group<8> : tensor<8xf32> -> tensor<1xf32>
    %13 = axis.map %3 to %3 : [!axis.factor_group<8>] [!axis.factor_group<8>]
    %14 = distributed.Collective %12 : tensor<1xf32> on %4 : <32> to tensor<1xf32> on %0 : <8> reduces (%7 : !axis.factor_group<4>) maps %13 : !axis.map {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %18 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %18 : tensor<f32>
    }
    %15 = distributed.Await %14 : <tensor<1xf32>> -> tensor<1xf32>
    %16 = distributed.CastLocalToGlobal %15 axes %0 : !axis.factor_group<8> : tensor<1xf32> -> tensor<8xf32>
    %17 = distributed.CastLocalToGlobal %15 axes %0 : !axis.factor_group<8> : tensor<1xf32> -> tensor<8xf32>
    distributed.DistributedYield %17 tensor<8xf32>
  }) : (!axis.factor_group<8>, !axis.factor_group<4>) -> ()
}

// CHECK-LABEL: module @axis_ssa_anchor {
// CHECK: %[[LOGICAL_AXIS:.*]] = distributed.LogicalMeshAxes [8] : !distributed.logical_mesh_axis<8>
// CHECK-NOT: distributed.LogicalMeshAxes [8] : !distributed.logical_mesh_axis<8>
// CHECK: "distributed.DistributedFunction"
// CHECK: distributed.DistributedKernel
// CHECK: distributed.CastGlobalToLocal {{.*}} axes %{{.*}} : !axis.factor_group<8>
// CHECK: distributed.Collective
// CHECK: distributed.CastLocalToGlobal {{.*}} axes %{{.*}} : !axis.factor_group<8>
