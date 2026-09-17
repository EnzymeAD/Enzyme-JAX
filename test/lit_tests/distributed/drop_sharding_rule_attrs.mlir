// RUN: enzymexlamlir-opt %s --drop-sharding-rule-attrs | FileCheck %s

// sdy.sharding_rule attrs on ops nested inside a distributed.DistributedKernel
// body must be droppable even though the enclosing function is a
// distributed.DistributedFunction, not a func.func -- Shardy's own
// sdy-drop-sharding-rules pass is anchored on func::FuncOp and would never
// find these.
// CHECK-LABEL: func.func @drops_sharding_rule
// CHECK: stablehlo.dot_general
// CHECK-NOT: sdy.sharding_rule
func.func @drops_sharding_rule(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %r = distributed.DistributedKernel (%a : tensor<4x4xf32>, %b : tensor<4x4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[], []] : unreduced_axes = []>, <dim_partitioning_axes = [[], []] : unreduced_axes = []>]>
      -> (tensor<4x4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[], []] : unreduced_axes = []>]>
      axes () {
  ^bb0(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>):
    %d = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] {sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=4, j=4, k=4} reduction={k}>} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    distributed.DistributedYield (%d : tensor<4x4xf32>)
  }
  return %r : tensor<4x4xf32>
}
