// RUN: enzymexlamlir-opt %s --sink-kernel-boundary-values --canonicalize -split-input-file | FileCheck %s

// A constant materialized inside one kernel and yielded to a second kernel is
// rebuilt inside the second kernel's own body instead, so the second kernel
// no longer needs the first's result as an ordinary operand. The pruning
// canonicalizations then drop the now-dead result/operand pair: the first
// kernel keeps only its genuinely-consumed result, and the second keeps none
// of its own operands.
// CHECK-LABEL: func.func @sinks_constant
// CHECK: %[[CST:.*]] = stablehlo.constant
// CHECK: %[[K1:.*]] = distributed.DistributedKernel (%{{.*}} : tensor<8xf32>) <
// CHECK-NEXT: -> (tensor<8xf32>)
// CHECK: %[[K2:.*]] = distributed.DistributedKernel () <
// CHECK-NEXT: -> (tensor<8xf32>)
// CHECK-NEXT: axes ()
// CHECK-NEXT: stablehlo.multiply %[[CST]], %[[CST]]
// CHECK: stablehlo.add %[[K1]], %[[K2]]
func.func @sinks_constant(%in: tensor<8xf32>) -> tensor<8xf32> {
  %k1:2 = distributed.DistributedKernel (%in : tensor<8xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<8xf32>, tensor<8xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>, <dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
  ^bb0(%a: tensor<8xf32>):
    %cst = stablehlo.constant dense<0.0> : tensor<8xf32>
    %s = stablehlo.add %a, %cst : tensor<8xf32>
    distributed.DistributedYield (%s : tensor<8xf32>, %cst : tensor<8xf32>)
  }
  %k2 = distributed.DistributedKernel (%k1#1 : tensor<8xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<8xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
  ^bb0(%b: tensor<8xf32>):
    %m = stablehlo.multiply %b, %b : tensor<8xf32>
    distributed.DistributedYield (%m : tensor<8xf32>)
  }
  %sum = stablehlo.add %k1#0, %k2 : tensor<8xf32>
  return %sum : tensor<8xf32>
}

// -----

// A pure op (here, a self-multiply) reachable only from a kernel's own
// argument, standing in for a transpose of a weight argument, is rebuilt
// inside the consuming kernel's body from that kernel's own copy of the
// same argument, since it is already, by SSA identity, one of the
// consumer's own operands: nothing new needs to cross the kernel boundary.
// CHECK-LABEL: func.func @sinks_arg_only_pure_chain
// CHECK: %[[K1:.*]] = distributed.DistributedKernel (%[[W:.*]] : tensor<8xf32>) <
// CHECK-NEXT: -> (tensor<8xf32>)
// CHECK: stablehlo.negate
// CHECK: %[[K2:.*]] = distributed.DistributedKernel (%[[W]] : tensor<8xf32>) <
// CHECK-NEXT: -> (tensor<8xf32>)
// CHECK: ^bb0(%[[WB:.*]]: tensor<8xf32>):
// CHECK-NEXT: %[[SQ:.*]] = stablehlo.multiply %[[WB]], %[[WB]]
// CHECK-NEXT: stablehlo.subtract %[[SQ]], %[[WB]]
// CHECK: stablehlo.add %[[K1]], %[[K2]]
func.func @sinks_arg_only_pure_chain(%w: tensor<8xf32>) -> tensor<8xf32> {
  %k1:2 = distributed.DistributedKernel (%w : tensor<8xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<8xf32>, tensor<8xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>, <dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
  ^bb0(%wa: tensor<8xf32>):
    %neg = stablehlo.negate %wa : tensor<8xf32>
    %sq = stablehlo.multiply %wa, %wa : tensor<8xf32>
    distributed.DistributedYield (%neg : tensor<8xf32>, %sq : tensor<8xf32>)
  }
  %k2 = distributed.DistributedKernel (%w : tensor<8xf32>, %k1#1 : tensor<8xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>, <dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<8xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
  ^bb0(%wb: tensor<8xf32>, %sqb: tensor<8xf32>):
    %r = stablehlo.subtract %sqb, %wb : tensor<8xf32>
    distributed.DistributedYield (%r : tensor<8xf32>)
  }
  %out = stablehlo.add %k1#0, %k2 : tensor<8xf32>
  return %out : tensor<8xf32>
}

// -----

// A genuinely data-dependent cross-kernel value (here, the first kernel's
// own computed result, not reachable from anything the second kernel already
// has) must stay a real operand: sinking it would either duplicate real
// cross-kernel work or, worse, require pulling in an argument the second
// kernel never depended on. Neither kernel's operand list changes.
// CHECK-LABEL: func.func @keeps_real_cross_kernel_dependency
// CHECK: %[[K1:.*]] = distributed.DistributedKernel (%{{.*}} : tensor<8xf32>)
// CHECK: distributed.DistributedKernel (%{{.*}} : tensor<8xf32>, %[[K1]] : tensor<8xf32>)
func.func @keeps_real_cross_kernel_dependency(%a: tensor<8xf32>, %b: tensor<8xf32>) -> tensor<8xf32> {
  %k1 = distributed.DistributedKernel (%a : tensor<8xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<8xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
  ^bb0(%aa: tensor<8xf32>):
    %s = stablehlo.multiply %aa, %aa : tensor<8xf32>
    distributed.DistributedYield (%s : tensor<8xf32>)
  }
  %k2 = distributed.DistributedKernel (%b : tensor<8xf32>, %k1 : tensor<8xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>, <dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<8xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
  ^bb0(%bb: tensor<8xf32>, %sb: tensor<8xf32>):
    %r = stablehlo.add %bb, %sb : tensor<8xf32>
    distributed.DistributedYield (%r : tensor<8xf32>)
  }
  return %k2 : tensor<8xf32>
}
