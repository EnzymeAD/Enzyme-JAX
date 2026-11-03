// RUN: enzymexlamlir-opt --split-input-file --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline %s | FileCheck %s --implicit-check-not="logical_mesh_axis<144>"

// Shardy has no rule for stablehlo.iota, so the compiler synthesizes one: the
// iota dimension needs replication and every other dimension is pointwise. In
// every case below the iota dimension is the only 144-wide dimension, so no
// shardable 144-wide logical axis may appear.

// The iota dimension, hinted sharded on the other operand, becomes device-local.
// CHECK-LABEL: module @iota_dimension_sharded_operand
// CHECK: distributed.DeviceLocalAxis 144
// CHECK: distributed.DistributedKernel
// CHECK: stablehlo.iota dim = 1
// CHECK: stablehlo.multiply
module @iota_dimension_sharded_operand {
  sdy.mesh @mesh = <["tp"=4]>
  func.func public @main(%x: tensor<8x144xf32>) -> tensor<8x144xf32> {
    %0 = sdy.sharding_constraint %x <@mesh, [{}, {"tp"}]> : tensor<8x144xf32>
    %1 = stablehlo.iota dim = 1 : tensor<8x144xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<8x144xf32>
    return %2 : tensor<8x144xf32>
  }
}

// -----

// The iota's pointwise dimension 0 stays a shardable logical axis.
// CHECK-LABEL: module @iota_pointwise_dimension_sharded_operand
// CHECK-NOT: device_local_axis<8>
// CHECK: distributed.LogicalMeshAxes 8
// CHECK-NOT: device_local_axis<8>
// CHECK: distributed.DistributedKernel
// CHECK: stablehlo.iota dim = 1
module @iota_pointwise_dimension_sharded_operand {
  sdy.mesh @mesh = <["tp"=4]>
  func.func public @main(%x: tensor<8x144xf32>) -> tensor<8x144xf32> {
    %0 = sdy.sharding_constraint %x <@mesh, [{"tp"}, {}]> : tensor<8x144xf32>
    %1 = stablehlo.iota dim = 1 : tensor<8x144xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<8x144xf32>
    return %2 : tensor<8x144xf32>
  }
}

// -----

// A 1-D iota broadcast into a sharded consumer dimension becomes device-local.
// CHECK-LABEL: module @iota_broadcast_into_sharded_consumer
// CHECK: distributed.DeviceLocalAxis 144
// CHECK: stablehlo.iota dim = 0
// CHECK: stablehlo.broadcast_in_dim
// CHECK: stablehlo.multiply
module @iota_broadcast_into_sharded_consumer {
  sdy.mesh @mesh = <["tp"=4]>
  func.func public @main(%x: tensor<8x144xf32>) -> tensor<8x144xf32> {
    %0 = sdy.sharding_constraint %x <@mesh, [{}, {"tp"}]> : tensor<8x144xf32>
    %1 = stablehlo.iota dim = 0 : tensor<144xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<144xf32>) -> tensor<8x144xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<8x144xf32>
    return %3 : tensor<8x144xf32>
  }
}
