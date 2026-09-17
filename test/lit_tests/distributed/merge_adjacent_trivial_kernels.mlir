// RUN: enzymexlamlir-opt %s --merge-adjacent-trivial-kernels -split-input-file | FileCheck %s

// Two adjacent trivial kernels (operand/result types already equal their
// body's block-arg/yield types, so no real sharding survives) with no SSA
// relationship between them merge into one kernel that returns both of
// their original results.
// CHECK-LABEL: func.func @independent_no_dependency
// CHECK: %[[IN0:.*]] = tensor.empty
// CHECK: %[[IN1:.*]] = tensor.empty
// CHECK: %[[R:.*]]:2 = distributed.DistributedKernel (%[[IN0]] : tensor<4xf32>, %[[IN1]] : tensor<4xf32>) {{.*}}
// CHECK-NEXT: -> (tensor<4xf32>, tensor<4xf32>) {{.*}}
// CHECK-NEXT: axes () {
// CHECK-NEXT: ^bb0(%[[A0:.*]]: tensor<4xf32>, %[[A1:.*]]: tensor<4xf32>):
// CHECK-NEXT: %[[C:.*]] = stablehlo.add %[[A0]], %[[A0]]
// CHECK-NEXT: %[[D:.*]] = stablehlo.multiply %[[A1]], %[[A1]]
// CHECK-NEXT: distributed.DistributedYield (%[[C]] : tensor<4xf32>, %[[D]] : tensor<4xf32>)
// CHECK: return %[[R]]#0, %[[R]]#1
func.func @independent_no_dependency() -> (tensor<4xf32>, tensor<4xf32>) {
  %in0 = tensor.empty() : tensor<4xf32>
  %in1 = tensor.empty() : tensor<4xf32>
  %r0 = distributed.DistributedKernel (%in0 : tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
  ^bb0(%arg0: tensor<4xf32>):
    %c = stablehlo.add %arg0, %arg0 : tensor<4xf32>
    distributed.DistributedYield (%c : tensor<4xf32>)
  }
  %r1 = distributed.DistributedKernel (%in1 : tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
  ^bb0(%arg1: tensor<4xf32>):
    %d = stablehlo.multiply %arg1, %arg1 : tensor<4xf32>
    distributed.DistributedYield (%d : tensor<4xf32>)
  }
  return %r0, %r1 : tensor<4xf32>, tensor<4xf32>
}

// -----

// The second kernel directly consumes the first kernel's result. The
// dependency is rewired to the shared internal value instead of round
// tripping through the (now removed) external result -- but since %r0 is
// also used outside the pair (returned directly), it must still appear
// among the merged kernel's own results.
// CHECK-LABEL: func.func @direct_dependency
// CHECK: %[[IN0:.*]] = tensor.empty
// CHECK: %[[R:.*]]:2 = distributed.DistributedKernel (%[[IN0]] : tensor<4xf32>) {{.*}}
// CHECK-NEXT: -> (tensor<4xf32>, tensor<4xf32>) {{.*}}
// CHECK-NEXT: axes () {
// CHECK-NEXT: ^bb0(%[[A0:.*]]: tensor<4xf32>):
// CHECK-NEXT: %[[C:.*]] = stablehlo.add %[[A0]], %[[A0]]
// CHECK-NEXT: %[[D:.*]] = stablehlo.multiply %[[C]], %[[C]]
// CHECK-NEXT: distributed.DistributedYield (%[[C]] : tensor<4xf32>, %[[D]] : tensor<4xf32>)
// CHECK: return %[[R]]#0, %[[R]]#1
func.func @direct_dependency() -> (tensor<4xf32>, tensor<4xf32>) {
  %in0 = tensor.empty() : tensor<4xf32>
  %r0 = distributed.DistributedKernel (%in0 : tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
  ^bb0(%arg0: tensor<4xf32>):
    %c = stablehlo.add %arg0, %arg0 : tensor<4xf32>
    distributed.DistributedYield (%c : tensor<4xf32>)
  }
  %r1 = distributed.DistributedKernel (%r0 : tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
  ^bb0(%arg1: tensor<4xf32>):
    %d = stablehlo.multiply %arg1, %arg1 : tensor<4xf32>
    distributed.DistributedYield (%d : tensor<4xf32>)
  }
  return %r0, %r1 : tensor<4xf32>, tensor<4xf32>
}

// -----

// Three adjacent trivial kernels collapse to a single kernel in one pass
// invocation: the first merge produces a new kernel that is itself
// revisited by the greedy driver, which then merges it with the third.
// CHECK-LABEL: func.func @three_in_a_row
// CHECK: %[[IN0:.*]] = tensor.empty
// CHECK: %[[R:.*]]:3 = distributed.DistributedKernel (%[[IN0]] : tensor<4xf32>) {{.*}}
// CHECK-NEXT: -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {{.*}}
// CHECK-NEXT: axes () {
// CHECK-NEXT: ^bb0(%[[A0:.*]]: tensor<4xf32>):
// CHECK-NEXT: %[[C:.*]] = stablehlo.add %[[A0]], %[[A0]]
// CHECK-NEXT: %[[D:.*]] = stablehlo.multiply %[[C]], %[[C]]
// CHECK-NEXT: %[[E:.*]] = stablehlo.subtract %[[D]], %[[D]]
// CHECK-NEXT: distributed.DistributedYield (%[[C]] : tensor<4xf32>, %[[D]] : tensor<4xf32>, %[[E]] : tensor<4xf32>)
// CHECK: return %[[R]]#0, %[[R]]#1, %[[R]]#2
func.func @three_in_a_row() -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
  %in0 = tensor.empty() : tensor<4xf32>
  %r0 = distributed.DistributedKernel (%in0 : tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
  ^bb0(%arg0: tensor<4xf32>):
    %c = stablehlo.add %arg0, %arg0 : tensor<4xf32>
    distributed.DistributedYield (%c : tensor<4xf32>)
  }
  %r1 = distributed.DistributedKernel (%r0 : tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
  ^bb0(%arg1: tensor<4xf32>):
    %d = stablehlo.multiply %arg1, %arg1 : tensor<4xf32>
    distributed.DistributedYield (%d : tensor<4xf32>)
  }
  %r2 = distributed.DistributedKernel (%r1 : tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
  ^bb0(%arg2: tensor<4xf32>):
    %e = stablehlo.subtract %arg2, %arg2 : tensor<4xf32>
    distributed.DistributedYield (%e : tensor<4xf32>)
  }
  return %r0, %r1, %r2 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
}

// -----

// A trivial kernel adjacent to a still-genuinely-sharded kernel (its
// operand/result type differs from its body's block-arg/yield type, so
// real partitioning survives) must not be merged.
// CHECK-LABEL: module @trivial_next_to_sharded
module @trivial_next_to_sharded {
  func.func @main() {
    return
  }
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 1>]
  %phys = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 1>
  %pf = axis.factor %phys : !distributed.physical_comm_axis<4, 1><4, 1>
  %pg = axis.product (%pf : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 4, 1>)
  %in0 = tensor.empty() : tensor<4xf32>
  // CHECK: distributed.DistributedKernel
  %r0 = distributed.DistributedKernel (%in0 : tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
  ^bb0(%arg0: tensor<4xf32>):
    %c = stablehlo.add %arg0, %arg0 : tensor<4xf32>
    distributed.DistributedYield (%c : tensor<4xf32>)
  }
  %in1 = tensor.empty() : tensor<1xf32>
  // CHECK: distributed.DistributedKernel
  %r1 = distributed.DistributedKernel (%in1 : tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>
      -> (tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>
      axes (%pg : !axis.factor_group<4>) {
  ^bb0(%arg1: tensor<4xf32>):
    distributed.DistributedYield (%arg1 : tensor<4xf32>)
  }
}

// -----

// Two trivial kernels separated by an unrelated op are not merged: this
// pass only ever looks at the immediate next sibling in the block, never
// hops over intervening ops (any op breaks adjacency, not just
// communication ops).
// CHECK-LABEL: func.func @not_physically_adjacent
func.func @not_physically_adjacent() -> (tensor<4xf32>, tensor<4xf32>) {
  %in0 = tensor.empty() : tensor<4xf32>
  // CHECK: distributed.DistributedKernel
  %r0 = distributed.DistributedKernel (%in0 : tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
  ^bb0(%arg0: tensor<4xf32>):
    %c = stablehlo.add %arg0, %arg0 : tensor<4xf32>
    distributed.DistributedYield (%c : tensor<4xf32>)
  }
  %unrelated = tensor.empty() : tensor<1xf32>
  %in1 = tensor.empty() : tensor<4xf32>
  // CHECK: distributed.DistributedKernel
  %r1 = distributed.DistributedKernel (%in1 : tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
  ^bb0(%arg1: tensor<4xf32>):
    %d = stablehlo.multiply %arg1, %arg1 : tensor<4xf32>
    distributed.DistributedYield (%d : tensor<4xf32>)
  }
  return %r0, %r1 : tensor<4xf32>, tensor<4xf32>
}
