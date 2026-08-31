// RUN: enzymexlamlir-opt --split-input-file --verify-diagnostics %s

// Module-level axis and function wiring is valid without mesh wrapper.
module {
  %axis = axis.getaxis tensor<4xf32> 0
  %f0 = axis.factor %axis : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %ctx = axis.product %f0 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>
  "distributed.DistributedFunction"(%ctx) <{argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>, function_type = (tensor<4xf32>) -> tensor<4xf32>, output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<4xf32>):
    distributed.DistributedYield %arg0 tensor<4xf32>
  }) : (!axis.factor_group<4>) -> ()
}

// -----

// GetPhysicalMeshAxes verifier should reject unknown symbol references.
module {
  // expected-error @+1 {{references unknown physical mesh symbol @missing_mesh}}
  %a = distributed.GetPhysicalMeshAxes @missing_mesh : !distributed.physical_comm_axis<4, 1>
}
