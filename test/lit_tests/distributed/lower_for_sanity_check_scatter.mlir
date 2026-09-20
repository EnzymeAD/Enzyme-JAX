// RUN: enzymexlamlir-opt --distributed-lower-for-sanity-check %s | FileCheck %s

// A scatter-shaped collective: the mapping sends TensorDim(0) -> Physical
// (axis0), the reverse of the gather case in lower_for_sanity_check.mlir. The
// input never varied along the physical axis (its input_mesh is trivial), so
// the axis's index-0 representative is kept and the tensor dim's atoms become
// the mesh dim.
// CHECK-LABEL: func.func @main
// CHECK: stablehlo.while
// CHECK: stablehlo.slice
// CHECK: stablehlo.broadcast_in_dim {{.*}} dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
// CHECK: stablehlo.while
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %f0 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %real_grp = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %trivial_grp = axis.product ()

  %ta = axis.getaxis tensor<2xf32> 0
  %tf0 = axis.factor %ta : !axis.shape_axis<tensor<2xf32>, 0> <2, 1>

  %lhs = axis.product (%tf0 : !axis.axis_factor<!axis.shape_axis<tensor<2xf32>, 0>, 2, 1>)
  %rhs = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map = axis.map %lhs to %rhs : [!axis.factor_group<2>] [!axis.factor_group<2>]

  "distributed.DistributedFunction"(%trivial_grp) <{
    argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>,
    function_type = (tensor<2xf32>) -> (tensor<2xf32>),
    output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>,
    sym_name = "main"
  }> ({
  ^bb0(%arg0: tensor<2xf32>):
    %local_in = distributed.CastGlobalToLocal %arg0 axes (%trivial_grp : !axis.factor_group<1>) : tensor<2xf32> -> tensor<2xf32>
    %k1 = distributed.DistributedKernel (%local_in : tensor<2xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
        -> (tensor<2xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
        axes () {
    ^bb1(%a: tensor<2xf32>):
      %c = stablehlo.multiply %a, %a : tensor<2xf32>
      distributed.DistributedYield (%c : tensor<2xf32>)
    }
    %g1 = distributed.CastLocalToGlobal %k1 axes (%trivial_grp : !axis.factor_group<1>) : tensor<2xf32> -> tensor<2xf32>
    %l1 = distributed.CastGlobalToLocal %g1 axes (%trivial_grp : !axis.factor_group<1>) : tensor<2xf32> -> tensor<2xf32>
    %h = distributed.Collective %l1 : tensor<2xf32> on %trivial_grp : !axis.factor_group<1> to tensor<1xf32> on %real_grp : !axis.factor_group<2> reduces () maps %map : !axis.map
    %v = distributed.Await %h : !distributed.asynch_handle<tensor<1xf32>> -> tensor<1xf32>
    %k2 = distributed.DistributedKernel (%v : tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
        -> (tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
        axes () {
    ^bb3(%b: tensor<1xf32>):
      %d = stablehlo.add %b, %b : tensor<1xf32>
      distributed.DistributedYield (%d : tensor<1xf32>)
    }
    %global_out = distributed.CastLocalToGlobal %k2 axes (%real_grp : !axis.factor_group<2>) : tensor<1xf32> -> tensor<2xf32>
    distributed.DistributedYield (%global_out : tensor<2xf32>)
  }) : (!axis.factor_group<1>) -> ()
}
