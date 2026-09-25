// RUN: enzymexlamlir-opt %s --split-input-file --mlir-very-unsafe-disable-verifier-on-parsing --verify-each=0 --shardy-to-distributed-pipeline --verify-diagnostics -o /dev/null

// The frontend pipeline ends with the explicit kernel verification.

// Two slots covering the same digits of one axis.
module {
  func.func @main() { return }
  %dev = distributed.DeviceLocalAxis 4 : !distributed.device_local_axis<4>
  %whole = axis.factor %dev : !distributed.device_local_axis<4><4, 1>
  %half = axis.factor %dev : !distributed.device_local_axis<4><2, 2>
  %a = axis.product (%whole : !axis.axis_factor<!distributed.device_local_axis<4>, 4, 1>)
  %b = axis.product (%half : !axis.axis_factor<!distributed.device_local_axis<4>, 2, 2>)
  %in = stablehlo.constant dense<1.0> : tensor<1x1xf32>
  // expected-error@+1 {{requires partitioning_axes slots to be disjoint, but slot 0 and slot 1 both cover digits [2, 4) of the same '!distributed.device_local_axis<4>'}}
  %r = distributed.DistributedKernel (%in : tensor<1x1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0], [1]] : unreduced_axes = []>]>
    -> (tensor<1x1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0], [1]] : unreduced_axes = []>]>
    axes (%a : !axis.factor_group<4>, %b : !axis.factor_group<2>) {
  ^bb0(%arg0: tensor<4x2xf32>):
    distributed.DistributedYield (%arg0 : tensor<4x2xf32>)
  }
}
