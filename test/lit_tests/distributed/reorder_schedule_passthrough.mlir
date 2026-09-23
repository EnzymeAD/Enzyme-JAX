// RUN: enzymexlamlir-opt --distributed-reorder-schedule %s | FileCheck %s

// An op the schedule graph does not cost (here a distributed.CastGlobalToLocal
// left in main's body after kernel inlining) sits between a cheap producer
// (%t) and an expensive consumer (%d). The %t -> %local -> %d dependency must
// still order %d after %t. Otherwise %d looks independent, the simulator
// admits it first (it has the longer critical path), and reordering moves it
// ahead of %local, which travels with %t.
// CHECK-LABEL: module @passthrough_dependency
// CHECK: %[[T:.*]] = stablehlo.transpose %arg0
// CHECK: %[[LOCAL:.*]] = distributed.CastGlobalToLocal %[[T]]
// CHECK: stablehlo.dot_general %[[LOCAL]], %arg1
// CHECK: distributed.DistributedYield
module @passthrough_dependency {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %f0 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %g0 = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %g1 = "axis.product"() : () -> !axis.factor_group<1>

  "distributed.DistributedFunction"(%g0, %g1) <{
    argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[], []] : unreduced_axes = []>, <dim_partitioning_axes = [[], []] : unreduced_axes = []>]>,
    function_type = (tensor<64x64xf32>, tensor<64x512xf32>) -> tensor<32x512xf32>,
    output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[], []] : unreduced_axes = []>]>,
    sym_name = "main"
  }> ({
  ^bb0(%arg0: tensor<64x64xf32>, %arg1: tensor<64x512xf32>):
    %t = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %local = distributed.CastGlobalToLocal %t axes (%g0 : !axis.factor_group<2>, %g1 : !axis.factor_group<1>) : tensor<64x64xf32> -> tensor<32x64xf32>
    %d = stablehlo.dot_general %local, %arg1, contracting_dims = [1] x [0] : (tensor<32x64xf32>, tensor<64x512xf32>) -> tensor<32x512xf32>
    distributed.DistributedYield (%d : tensor<32x512xf32>)
  }) : (!axis.factor_group<2>, !axis.factor_group<1>) -> ()
}
