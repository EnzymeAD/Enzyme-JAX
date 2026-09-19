// A non-trivial kernel (real sharding attributes still present -- the
// pipeline-ordering precondition this pass hard-fails against) is an
// expected-failure case: a clear diagnostic and non-zero exit, not a
// partial/best-effort lowering. Kept in its own file (rather than an
// additional --split-input-file chunk of lower_for_sanity_check.mlir) so a
// failing chunk can't interact with the exit status of that file's
// passing-case RUN line.
// RUN: not enzymexlamlir-opt --distributed-lower-for-sanity-check %s 2>&1 | FileCheck %s
// CHECK: requires every DistributedKernel to already be trivially local
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
    // A still-sharded (not trivially local) kernel: its own operand/result
    // (tensor<2xf32>) differ in shape from the region's block-arg/yield
    // (tensor<4xf32>) by the declared partitioning extent, i.e.
    // isTriviallyLocalKernel is false -- this is the pipeline-ordering
    // precondition violation this pass must fail loudly against.
    %kres = distributed.DistributedKernel (%local_in : tensor<2xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>
        -> (tensor<2xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>
        axes (%axes_grp : !axis.factor_group<2>) {
    ^bb1(%a: tensor<4xf32>):
      distributed.DistributedYield (%a : tensor<4xf32>)
    }
    %global_out = distributed.CastLocalToGlobal %kres axes (%axes_grp : !axis.factor_group<2>) : tensor<2xf32> -> tensor<4xf32>
    distributed.DistributedYield (%global_out : tensor<4xf32>)
  }) : (!axis.factor_group<2>) -> ()
}
