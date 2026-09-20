// RUN: enzymexlamlir-opt --distributed-lower-for-sanity-check %s | FileCheck %s

// main calls the same callee twice: each call site is inlined in place (see
// inlineDistributedCalls in LowerForSanityCheck.cpp), so this produces
// exactly the two independent doubling while-loops a hand-unrolled version
// of the same program would, back to back, with no distributed.DistributedCall
// or distributed.DistributedFunction residue.
// CHECK-LABEL: func.func @main
// CHECK-NOT: distributed.
// CHECK-NOT: axis.
// CHECK: stablehlo.while
// CHECK: stablehlo.while
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
    sym_name = "double",
    sym_visibility = "private"
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

  "distributed.DistributedFunction"(%axes_grp) <{
    argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>,
    function_type = (tensor<4xf32>) -> tensor<4xf32>,
    output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>,
    sym_name = "main"
  }> ({
  ^bb0(%arg0: tensor<4xf32>):
    %local_in = distributed.CastGlobalToLocal %arg0 axes (%axes_grp : !axis.factor_group<2>) : tensor<4xf32> -> tensor<2xf32>
    %c1 = distributed.DistributedCall @double (%local_in : tensor<2xf32>) <[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>
        -> (tensor<2xf32>) <[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>
        axes (%axes_grp : !axis.factor_group<2>)
    %global_mid = distributed.CastLocalToGlobal %c1 axes (%axes_grp : !axis.factor_group<2>) : tensor<2xf32> -> tensor<4xf32>
    %local_mid = distributed.CastGlobalToLocal %global_mid axes (%axes_grp : !axis.factor_group<2>) : tensor<4xf32> -> tensor<2xf32>
    %c2 = distributed.DistributedCall @double (%local_mid : tensor<2xf32>) <[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>
        -> (tensor<2xf32>) <[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>
        axes (%axes_grp : !axis.factor_group<2>)
    %global_out = distributed.CastLocalToGlobal %c2 axes (%axes_grp : !axis.factor_group<2>) : tensor<2xf32> -> tensor<4xf32>
    distributed.DistributedYield (%global_out : tensor<4xf32>)
  }) : (!axis.factor_group<2>) -> ()
}
