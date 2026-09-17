// RUN: enzymexlamlir-opt %s --drop-trivial-kernel-sharding -split-input-file | FileCheck %s

// A trivial kernel (no real sharding: operand/result types already equal
// their body's block-arg/yield types) has its own argument_shardings/
// output_shardings blanked and partitioning_axes emptied, freeing whatever
// axis-algebra values used to feed those slots to become dead code.
// CHECK-LABEL: func.func @blanks_trivial_kernel
// CHECK: distributed.DistributedKernel (%{{.*}} : tensor<32x512xf32>, %{{.*}} : tensor<512x32xf32>) <[<dim_partitioning_axes = {{\[}}[], []] : unreduced_axes = []>, <dim_partitioning_axes = {{\[}}[], []] : unreduced_axes = []>]>
// CHECK-NEXT: -> (tensor<32x32xf32>) <[<dim_partitioning_axes = {{\[}}[], []] : unreduced_axes = []>]>
// CHECK-NEXT: axes () {
func.func @blanks_trivial_kernel(%lhs: tensor<32x512xf32>, %rhs: tensor<512x32xf32>, %f0: !axis.factor_group<1>, %f1: !axis.factor_group<1>) -> tensor<32x32xf32> {
  %r = distributed.DistributedKernel (%lhs : tensor<32x512xf32>, %rhs : tensor<512x32xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0], []] : unreduced_axes = []>, <dim_partitioning_axes = [[], [1]] : unreduced_axes = []>]>
      -> (tensor<32x32xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0], [1]] : unreduced_axes = []>]>
      axes (%f0 : !axis.factor_group<1>, %f1 : !axis.factor_group<1>) {
  ^bb0(%arg0: tensor<32x512xf32>, %arg1: tensor<512x32xf32>):
    %d = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<32x512xf32>, tensor<512x32xf32>) -> tensor<32x32xf32>
    distributed.DistributedYield (%d : tensor<32x32xf32>)
  }
  return %r : tensor<32x32xf32>
}

// -----

// A non-trivial kernel (operand type differs from its block-arg type) keeps
// its sharding metadata untouched.
// CHECK-LABEL: func.func @keeps_nontrivial_kernel
// CHECK: distributed.DistributedKernel (%{{.*}} : tensor<128x512xf32>) <[<dim_partitioning_axes = {{\[}}[0], []] : unreduced_axes = []>]>
func.func @keeps_nontrivial_kernel(%in: tensor<128x512xf32>, %f0: !axis.factor_group<4>) -> tensor<32x512xf32> {
  %r = distributed.DistributedKernel (%in : tensor<128x512xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0], []] : unreduced_axes = []>]>
      -> (tensor<32x512xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[], []] : unreduced_axes = []>]>
      axes (%f0 : !axis.factor_group<4>) {
  ^bb0(%arg0: tensor<32x512xf32>):
    distributed.DistributedYield (%arg0 : tensor<32x512xf32>)
  }
  return %r : tensor<32x512xf32>
}
