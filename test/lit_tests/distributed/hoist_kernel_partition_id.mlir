// RUN: enzymexlamlir-opt %s --hoist-kernel-partition-id -split-input-file | FileCheck %s

// stablehlo.partition_id sitting directly in a kernel body is hoisted out to
// one new stablehlo.partition_id in the kernel's parent scope, appended as a
// new trailing operand/block-argument pair with a dimensionless
// argument_shardings entry (the op is a rank-0 scalar, so there is no
// dimension to declare a partitioning axis over).
// CHECK-LABEL: func.func @hoists_single_partition_id
// CHECK: %[[HOISTED:.+]] = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT: distributed.DistributedKernel (%{{.*}} : tensor<32x512xf32>, %{{.*}} : tensor<512x32xf32>, %[[HOISTED]] : tensor<ui32>) <[<dim_partitioning_axes = {{\[}}[], []] : unreduced_axes = []>, <dim_partitioning_axes = {{\[}}[], []] : unreduced_axes = []>, <dim_partitioning_axes = [] : unreduced_axes = []>]>
// CHECK: ^bb0(%{{.*}}: tensor<32x512xf32>, %{{.*}}: tensor<512x32xf32>, %[[ARG:.+]]: tensor<ui32>):
// CHECK-NOT: stablehlo.partition_id
// CHECK: stablehlo.compare EQ, %[[ARG]], %{{.*}}
// CHECK-NOT: stablehlo.partition_id
// CHECK: distributed.DistributedYield
func.func @hoists_single_partition_id(%lhs: tensor<32x512xf32>, %rhs: tensor<512x32xf32>, %f0: !axis.factor_group<1>, %f1: !axis.factor_group<1>) -> tensor<32x32xf32> {
  %r = distributed.DistributedKernel (%lhs : tensor<32x512xf32>, %rhs : tensor<512x32xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[], []] : unreduced_axes = []>, <dim_partitioning_axes = [[], []] : unreduced_axes = []>]>
      -> (tensor<32x32xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[], []] : unreduced_axes = []>]>
      axes (%f0 : !axis.factor_group<1>, %f1 : !axis.factor_group<1>) {
  ^bb0(%arg0: tensor<32x512xf32>, %arg1: tensor<512x32xf32>):
    %pid = stablehlo.partition_id : tensor<ui32>
    %zero = stablehlo.constant dense<0> : tensor<ui32>
    %cmp = stablehlo.compare EQ, %pid, %zero : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
    %d = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<32x512xf32>, tensor<512x32xf32>) -> tensor<32x32xf32>
    distributed.DistributedYield (%d : tensor<32x32xf32>)
  }
  return %r : tensor<32x32xf32>
}

// -----

// Multiple occurrences in one body (deterministic and identical, since
// stablehlo.partition_id takes no operands) collapse onto the single new
// hoisted value, not one per occurrence.
// CHECK-LABEL: func.func @collapses_multiple_partition_ids
// CHECK: %[[HOISTED:.+]] = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT: distributed.DistributedKernel (%{{.*}} : tensor<32x512xf32>, %{{.*}} : tensor<512x32xf32>, %[[HOISTED]] : tensor<ui32>)
// CHECK: ^bb0(%{{.*}}: tensor<32x512xf32>, %{{.*}}: tensor<512x32xf32>, %[[ARG:.+]]: tensor<ui32>):
// CHECK-NOT: stablehlo.partition_id
// CHECK: stablehlo.compare EQ, %[[ARG]], %{{.*}}
// CHECK: stablehlo.compare NE, %[[ARG]], %{{.*}}
// CHECK-NOT: stablehlo.partition_id
// CHECK: distributed.DistributedYield
func.func @collapses_multiple_partition_ids(%lhs: tensor<32x512xf32>, %rhs: tensor<512x32xf32>, %f0: !axis.factor_group<1>, %f1: !axis.factor_group<1>) -> tensor<32x32xf32> {
  %r = distributed.DistributedKernel (%lhs : tensor<32x512xf32>, %rhs : tensor<512x32xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[], []] : unreduced_axes = []>, <dim_partitioning_axes = [[], []] : unreduced_axes = []>]>
      -> (tensor<32x32xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[], []] : unreduced_axes = []>]>
      axes (%f0 : !axis.factor_group<1>, %f1 : !axis.factor_group<1>) {
  ^bb0(%arg0: tensor<32x512xf32>, %arg1: tensor<512x32xf32>):
    %pid0 = stablehlo.partition_id : tensor<ui32>
    %pid1 = stablehlo.partition_id : tensor<ui32>
    %zero = stablehlo.constant dense<0> : tensor<ui32>
    %cmp0 = stablehlo.compare EQ, %pid0, %zero : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
    %cmp1 = stablehlo.compare NE, %pid1, %zero : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
    %d = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<32x512xf32>, tensor<512x32xf32>) -> tensor<32x32xf32>
    distributed.DistributedYield (%d : tensor<32x32xf32>)
  }
  return %r : tensor<32x32xf32>
}

// -----

// A kernel with no stablehlo.partition_id anywhere in its body is left
// untouched: same operand count, same argument_shardings.
// CHECK-LABEL: func.func @leaves_kernel_without_partition_id_untouched
// CHECK-NOT: stablehlo.partition_id
// CHECK: distributed.DistributedKernel (%{{.*}} : tensor<32x512xf32>, %{{.*}} : tensor<512x32xf32>) <[<dim_partitioning_axes = {{\[}}[], []] : unreduced_axes = []>, <dim_partitioning_axes = {{\[}}[], []] : unreduced_axes = []>]>
func.func @leaves_kernel_without_partition_id_untouched(%lhs: tensor<32x512xf32>, %rhs: tensor<512x32xf32>, %f0: !axis.factor_group<1>, %f1: !axis.factor_group<1>) -> tensor<32x32xf32> {
  %r = distributed.DistributedKernel (%lhs : tensor<32x512xf32>, %rhs : tensor<512x32xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[], []] : unreduced_axes = []>, <dim_partitioning_axes = [[], []] : unreduced_axes = []>]>
      -> (tensor<32x32xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[], []] : unreduced_axes = []>]>
      axes (%f0 : !axis.factor_group<1>, %f1 : !axis.factor_group<1>) {
  ^bb0(%arg0: tensor<32x512xf32>, %arg1: tensor<512x32xf32>):
    %d = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<32x512xf32>, tensor<512x32xf32>) -> tensor<32x32xf32>
    distributed.DistributedYield (%d : tensor<32x32xf32>)
  }
  return %r : tensor<32x32xf32>
}
