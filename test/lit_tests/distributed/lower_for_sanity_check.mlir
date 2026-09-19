// RUN: enzymexlamlir-opt --split-input-file --distributed-lower-for-sanity-check %s | FileCheck %s

// Single physical axis, one trivial kernel, no collectives: expect one
// stablehlo.while, a reshape+broadcast_in_dim materializing the expanded
// input outside it, dynamic_slice/dynamic_update_slice inside, and a plain
// func.func (no distributed.*/axis.* residue) at the top level.
// CHECK-LABEL: func.func @main
// CHECK-NOT: distributed.
// CHECK-NOT: axis.
// CHECK: stablehlo.reshape
// CHECK: stablehlo.broadcast_in_dim
// CHECK: stablehlo.while
// CHECK: stablehlo.dynamic_slice
// CHECK: stablehlo.add
// CHECK: stablehlo.dynamic_update_slice
// CHECK: return
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %f0 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %axes_grp = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)

  "distributed.DistributedFunction"(%axes_grp) <{
    argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>,
    function_type = (tensor<4xf32>) -> tensor<4xf32>,
    output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>,
    sym_name = "main"
  }> ({
  ^bb0(%arg0: tensor<4xf32>):
    %local_in = distributed.CastGlobalToLocal %arg0 axes (%axes_grp : !axis.factor_group<2>) : tensor<4xf32> -> tensor<2xf32>
    %kres = distributed.DistributedKernel (%local_in : tensor<2xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
        -> (tensor<2xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
        axes () {
    ^bb1(%a: tensor<2xf32>):
      %c = stablehlo.add %a, %a : tensor<2xf32>
      distributed.DistributedYield (%c : tensor<2xf32>)
    }
    %global_out = distributed.CastLocalToGlobal %kres axes (%axes_grp : !axis.factor_group<2>) : tensor<2xf32> -> tensor<4xf32>
    distributed.DistributedYield (%global_out : tensor<4xf32>)
  }) : (!axis.factor_group<2>) -> ()
}

// -----

// Two physical axes sharding one kernel: expect two properly *nested*
// stablehlo.while ops (not a flat loop with decode math), each loop's own
// induction variable used directly as that axis's slice coordinate.
// CHECK-LABEL: func.func @main
// CHECK: stablehlo.while
// CHECK: stablehlo.while
// CHECK: stablehlo.dynamic_slice
// CHECK: stablehlo.dynamic_update_slice
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %f0 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f1 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %axes_grp = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)

  "distributed.DistributedFunction"(%axes_grp) <{
    argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>,
    function_type = (tensor<4xf32>) -> tensor<4xf32>,
    output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>,
    sym_name = "main"
  }> ({
  ^bb0(%arg0: tensor<4xf32>):
    %local_in = distributed.CastGlobalToLocal %arg0 axes (%axes_grp : !axis.factor_group<4>) : tensor<4xf32> -> tensor<1xf32>
    %kres = distributed.DistributedKernel (%local_in : tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
        -> (tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
        axes () {
    ^bb1(%a: tensor<1xf32>):
      %c = stablehlo.add %a, %a : tensor<1xf32>
      distributed.DistributedYield (%c : tensor<1xf32>)
    }
    %global_out = distributed.CastLocalToGlobal %kres axes (%axes_grp : !axis.factor_group<4>) : tensor<1xf32> -> tensor<4xf32>
    distributed.DistributedYield (%global_out : tensor<4xf32>)
  }) : (!axis.factor_group<4>) -> ()
}
