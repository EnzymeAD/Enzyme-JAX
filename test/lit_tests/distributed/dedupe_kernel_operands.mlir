// RUN: enzymexlamlir-opt %s --canonicalize -split-input-file | FileCheck %s

// The same value passed twice with equal sharding entries is one operand, and
// uses of the later block argument move to the first.
// CHECK-LABEL: func.func @merges_duplicates
// CHECK: distributed.DistributedKernel (%{{.*}} : tensor<8xf32>) <[<dim_partitioning_axes = {{\[}}[]] : unreduced_axes = []>]>
// CHECK-NEXT: -> (tensor<8xf32>)
// CHECK: ^bb0(%[[A:.*]]: tensor<8xf32>):
// CHECK-NEXT: stablehlo.subtract %[[A]], %[[A]]
func.func @merges_duplicates(%in: tensor<8xf32>) -> tensor<8xf32> {
  %r = distributed.DistributedKernel (%in : tensor<8xf32>, %in : tensor<8xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>, <dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<8xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
  ^bb0(%a: tensor<8xf32>, %b: tensor<8xf32>):
    %s = stablehlo.subtract %a, %b : tensor<8xf32>
    distributed.DistributedYield (%s : tensor<8xf32>)
  }
  return %r : tensor<8xf32>
}

// -----

// Distinct values stay separate operands.
// CHECK-LABEL: func.func @keeps_distinct_values
// CHECK: distributed.DistributedKernel (%{{.*}} : tensor<8xf32>, %{{.*}} : tensor<8xf32>)
func.func @keeps_distinct_values(%x: tensor<8xf32>, %y: tensor<8xf32>) -> tensor<8xf32> {
  %r = distributed.DistributedKernel (%x : tensor<8xf32>, %y : tensor<8xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>, <dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<8xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
  ^bb0(%a: tensor<8xf32>, %b: tensor<8xf32>):
    %s = stablehlo.subtract %a, %b : tensor<8xf32>
    distributed.DistributedYield (%s : tensor<8xf32>)
  }
  return %r : tensor<8xf32>
}

// -----

// The same value with different sharding entries is not merged.
// CHECK-LABEL: func.func @keeps_same_value_with_different_shardings
// CHECK: distributed.DistributedKernel (%{{.*}} : tensor<8xf32>, %{{.*}} : tensor<8xf32>)
func.func @keeps_same_value_with_different_shardings(%in: tensor<8xf32>, %f0: !axis.factor_group<1>) -> tensor<8xf32> {
  %r = distributed.DistributedKernel (%in : tensor<8xf32>, %in : tensor<8xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>, <dim_partitioning_axes = [[0]] : unreduced_axes = []>]>
      -> (tensor<8xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes (%f0 : !axis.factor_group<1>) {
  ^bb0(%a: tensor<8xf32>, %b: tensor<8xf32>):
    %s = stablehlo.subtract %a, %b : tensor<8xf32>
    distributed.DistributedYield (%s : tensor<8xf32>)
  }
  return %r : tensor<8xf32>
}
