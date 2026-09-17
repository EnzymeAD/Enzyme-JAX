// RUN: enzymexlamlir-opt %s --drop-kernel-body-sharding-attrs | FileCheck %s

// distributed.argument_shardings/output_shardings on ops inside a kernel
// body are stale once distributed-lower-kernels has consumed them; this
// pass drops them unconditionally.
// CHECK-LABEL: func.func @drops_body_sharding_attrs
// CHECK: stablehlo.add %arg0, %arg0 : tensor<4xf32>
// CHECK-NOT: distributed.argument_shardings
// CHECK-NOT: distributed.output_shardings
func.func @drops_body_sharding_attrs() -> tensor<4xf32> {
  %in0 = tensor.empty() : tensor<4xf32>
  %r0 = distributed.DistributedKernel (%in0 : tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
  ^bb0(%arg0: tensor<4xf32>):
    %c = stablehlo.add %arg0, %arg0 {distributed.argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>, <dim_partitioning_axes = [[0]] : unreduced_axes = []>]>, distributed.output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>} : tensor<4xf32>
    distributed.DistributedYield (%c : tensor<4xf32>)
  }
  return %r0 : tensor<4xf32>
}
