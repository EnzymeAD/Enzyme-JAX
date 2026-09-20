// RUN: enzymexlamlir-opt --canonicalize-sharded-factor-order %s | FileCheck %s

// The returned value's flat dimension is declared [device-local 8, physical x3,
// device-local 2]: the sharded factors sit between two device-local ones (the
// result of x[8,16].reshape(128) with the columns sharded). Reordering the
// return cast to put the sharded factors first changes which tensor the cast
// describes, so the pass pairs it with a relayout of the global result: split
// the dimension into its factors, transpose the sharded ones to the front's
// inverse, and merge back. The function's own signature is untouched.
// CHECK-LABEL: DistributedFunction
// CHECK: distributed.CastLocalToGlobal
// CHECK-NEXT: stablehlo.reshape {{.*}} : (tensor<128xf32>) -> tensor<2x2x2x8x2x1xf32>
// CHECK-NEXT: stablehlo.transpose {{.*}} dims = [3, 0, 1, 2, 4, 5]
// CHECK-NEXT: stablehlo.reshape {{.*}} : (tensor<8x2x2x2x2x1xf32>) -> tensor<128xf32>
// CHECK-NEXT: distributed.DistributedYield
module @jit__lambda attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  %0 = distributed.DeviceLocalAxis 2 : <2>
  %1 = distributed.DeviceLocalAxis 8 : <8>
  %2:3 = distributed.GetPhysicalMeshAxes @mesh448 : !distributed.physical_comm_axis<4, 32>, !distributed.physical_comm_axis<4, 8>, !distributed.physical_comm_axis<8, 1>
  %3 = axis.product (%4 : !axis.axis_factor<!distributed.device_local_axis<8>, 8, 1>)
  %4 = axis.factor %1 : !distributed.device_local_axis<8><8, 1>
  %5 = axis.product (%6 : !axis.axis_factor<!distributed.physical_comm_axis<4, 32>, 2, 2>, %7 : !axis.axis_factor<!distributed.physical_comm_axis<4, 8>, 2, 2>, %8 : !axis.axis_factor<!distributed.physical_comm_axis<4, 32>, 2, 1>, %9 : !axis.axis_factor<!distributed.device_local_axis<2>, 2, 1>)
  %6 = axis.factor %2#0 : !distributed.physical_comm_axis<4, 32><2, 2>
  %7 = axis.factor %2#1 : !distributed.physical_comm_axis<4, 8><2, 2>
  %8 = axis.factor %2#0 : !distributed.physical_comm_axis<4, 32><2, 1>
  %9 = axis.factor %0 : !distributed.device_local_axis<2><2, 1>
  %10 = axis.product (%4 : !axis.axis_factor<!distributed.device_local_axis<8>, 8, 1>, %6 : !axis.axis_factor<!distributed.physical_comm_axis<4, 32>, 2, 2>, %7 : !axis.axis_factor<!distributed.physical_comm_axis<4, 8>, 2, 2>, %8 : !axis.axis_factor<!distributed.physical_comm_axis<4, 32>, 2, 1>, %9 : !axis.axis_factor<!distributed.device_local_axis<2>, 2, 1>)
  "distributed.DistributedFunction"(%3, %5) <{argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0], [1]] : unreduced_axes = []>]>, function_type = (tensor<8x16xf32>) -> tensor<128xf32>, output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0, 1]] : unreduced_axes = []>]>, res_attrs = [{jax.result_info = "result"}], sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<8x16xf32>):
    %11 = distributed.CastGlobalToLocal %arg0 axes (%3 : !axis.factor_group<8>, %5 : !axis.factor_group<16>) : tensor<8x16xf32> -> tensor<1x1xf32>
    %12 = distributed.DistributedKernel (%11 : tensor<1x1xf32>) <[<dim_partitioning_axes = [[0], [1]] : unreduced_axes = []>]>
      -> (tensor<1xf32>) <[<dim_partitioning_axes = [[0, 1]] : unreduced_axes = []>]>
      axes (%3 : !axis.factor_group<8>, %5 : !axis.factor_group<16>) {
    ^bb0(%arg1: tensor<8x16xf32>):
      %14 = stablehlo.reshape %arg1 {distributed.argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0], [1]] : unreduced_axes = []>]>, distributed.output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0, 1]] : unreduced_axes = []>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([ij]) {i=8, j=16}>} : (tensor<8x16xf32>) -> tensor<128xf32>
      distributed.DistributedYield (%14 : tensor<128xf32>)
    }
    %13 = distributed.CastLocalToGlobal %12 axes (%10 : !axis.factor_group<128>) : tensor<1xf32> -> tensor<128xf32>
    distributed.DistributedYield (%13 : tensor<128xf32>)
  }) : (!axis.factor_group<8>, !axis.factor_group<16>) -> ()
  distributed.PhysicalMesh @mesh448 device_target "cpu" axes [!distributed.physical_comm_axis<4, 32>, !distributed.physical_comm_axis<4, 8>, !distributed.physical_comm_axis<8, 1>]
}

