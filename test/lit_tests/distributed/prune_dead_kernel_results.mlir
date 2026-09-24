// RUN: enzymexlamlir-opt %s --canonicalize -split-input-file | FileCheck %s

// A kernel result with no uses is removed from the op's result list, its
// body's yield, and output_shardings, and the body compute that only fed it
// is dropped. Remaining results keep their relative order.
// CHECK-LABEL: func.func @prunes_dead_results
// CHECK: %[[R:.*]] = distributed.DistributedKernel (%{{.*}} : tensor<8xf32>) <[<dim_partitioning_axes = {{\[}}[]] : unreduced_axes = []>]>
// CHECK-NEXT: -> (tensor<8xf32>) <[<dim_partitioning_axes = {{\[}}[]] : unreduced_axes = []>]>
// CHECK-NEXT: axes () {
// CHECK-NEXT: ^bb0(%[[A:.*]]: tensor<8xf32>):
// CHECK-NEXT: %[[E:.*]] = stablehlo.exponential %[[A]]
// CHECK-NEXT: distributed.DistributedYield (%[[E]] : tensor<8xf32>)
// CHECK: return %[[R]]
func.func @prunes_dead_results(%in: tensor<8xf32>) -> tensor<8xf32> {
  %r:3 = distributed.DistributedKernel (%in : tensor<8xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<8xf32>, tensor<8xf32>, tensor<f32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>, <dim_partitioning_axes = [[]] : unreduced_axes = []>, <dim_partitioning_axes = [] : unreduced_axes = []>]>
      axes () {
  ^bb0(%arg0: tensor<8xf32>):
    %n = stablehlo.negate %arg0 : tensor<8xf32>
    %e = stablehlo.exponential %arg0 : tensor<8xf32>
    %c = stablehlo.constant dense<0.0> : tensor<f32>
    distributed.DistributedYield (%n : tensor<8xf32>, %e : tensor<8xf32>, %c : tensor<f32>)
  }
  return %r#1 : tensor<8xf32>
}

// -----

// Every result live: the kernel is untouched.
// CHECK-LABEL: func.func @keeps_live_results
// CHECK: distributed.DistributedKernel
// CHECK-NEXT: -> (tensor<8xf32>, tensor<8xf32>)
// CHECK: distributed.DistributedYield (%{{.*}} : tensor<8xf32>, %{{.*}} : tensor<8xf32>)
func.func @keeps_live_results(%in: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  %r:2 = distributed.DistributedKernel (%in : tensor<8xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<8xf32>, tensor<8xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>, <dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
  ^bb0(%arg0: tensor<8xf32>):
    %n = stablehlo.negate %arg0 : tensor<8xf32>
    %e = stablehlo.exponential %arg0 : tensor<8xf32>
    distributed.DistributedYield (%n : tensor<8xf32>, %e : tensor<8xf32>)
  }
  return %r#0, %r#1 : tensor<8xf32>, tensor<8xf32>
}

// -----

// A kernel with no live results is not removed or emptied.
// CHECK-LABEL: func.func @keeps_all_dead_kernel
// CHECK: distributed.DistributedKernel
// CHECK-NEXT: -> (tensor<8xf32>)
func.func @keeps_all_dead_kernel(%in: tensor<8xf32>) {
  %r = distributed.DistributedKernel (%in : tensor<8xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<8xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
  ^bb0(%arg0: tensor<8xf32>):
    %n = stablehlo.negate %arg0 : tensor<8xf32>
    distributed.DistributedYield (%n : tensor<8xf32>)
  }
  return
}
