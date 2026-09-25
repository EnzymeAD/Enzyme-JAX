// RUN: enzymexlamlir-opt %s --split-input-file --verify-diagnostics -o /dev/null

// One logical axis shards each dimension at most once: valid.
module {
  func.func @main() { return }
  %dev = distributed.DeviceLocalAxis 4 : !distributed.device_local_axis<4>
  %lo = axis.factor %dev : !distributed.device_local_axis<4><2, 1>
  %hi = axis.factor %dev : !distributed.device_local_axis<4><2, 2>
  %a = axis.product (%lo : !axis.axis_factor<!distributed.device_local_axis<4>, 2, 1>)
  %b = axis.product (%hi : !axis.axis_factor<!distributed.device_local_axis<4>, 2, 2>)
  %in = stablehlo.constant dense<1.0> : tensor<1x1xf32>
  %r = distributed.DistributedKernel (%in : tensor<1x1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0], [1]] : unreduced_axes = []>]>
    -> (tensor<1x1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0], [1]] : unreduced_axes = []>]>
    axes (%a : !axis.factor_group<2>, %b : !axis.factor_group<2>) {
  ^bb0(%arg0: tensor<2x2xf32>):
    distributed.DistributedYield (%arg0 : tensor<2x2xf32>)
  }
}

// -----

// Axis 0 shards both dimensions of one operand.
module {
  func.func @main() { return }
  %dev = distributed.DeviceLocalAxis 4 : !distributed.device_local_axis<4>
  %lo = axis.factor %dev : !distributed.device_local_axis<4><2, 1>
  %a = axis.product (%lo : !axis.axis_factor<!distributed.device_local_axis<4>, 2, 1>)
  %in = stablehlo.constant dense<1.0> : tensor<1x1xf32>
  // expected-error@+1 {{requires argument_shardings[0] to use partitioning axis 0 at most once, but it appears again in dim_partitioning_axes[1]}}
  %r = distributed.DistributedKernel (%in : tensor<1x1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0], [0]] : unreduced_axes = []>]>
    -> (tensor<1x1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[], []] : unreduced_axes = []>]>
    axes (%a : !axis.factor_group<2>) {
  ^bb0(%arg0: tensor<2x2xf32>):
    distributed.DistributedYield (%arg0 : tensor<2x2xf32>)
  }
}

// -----

// A result axis that shards a dimension is also listed unreduced.
module {
  func.func @main() { return }
  %dev = distributed.DeviceLocalAxis 4 : !distributed.device_local_axis<4>
  %lo = axis.factor %dev : !distributed.device_local_axis<4><2, 1>
  %a = axis.product (%lo : !axis.axis_factor<!distributed.device_local_axis<4>, 2, 1>)
  %in = stablehlo.constant dense<1.0> : tensor<1x1xf32>
  // expected-error@+1 {{requires output_shardings[0] to use partitioning axis 0 at most once, but it appears again in unreduced_axes}}
  %r = distributed.DistributedKernel (%in : tensor<1x1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[], []] : unreduced_axes = []>]>
    -> (tensor<1x1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0], []] : unreduced_axes = [0]>]>
    axes (%a : !axis.factor_group<2>) {
  ^bb0(%arg0: tensor<1x1xf32>):
    distributed.DistributedYield (%arg0 : tensor<1x1xf32>)
  }
}
