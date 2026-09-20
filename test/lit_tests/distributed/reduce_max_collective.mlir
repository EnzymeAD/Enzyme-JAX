// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --shardy-to-distributed-pipeline %s | FileCheck %s

// A max-reduction over a sharded dimension: each device reduces its shard
// with max, so the collective combining the partial results across devices
// must also take the max. A sum there gives values larger than any input.
// CHECK: distributed.Collective
// CHECK: stablehlo.maximum
// CHECK-NOT: stablehlo.add
module @jit__lambda attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["tp"=4]>
  func.func public @main(%arg0: tensor<8x16xf32>) -> (tensor<8xf32> {jax.result_info = "result"}) {
    %0 = sdy.sharding_constraint %arg0 <@mesh, [{}, {"tp"}]> : tensor<8x16xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.maximum across dimensions = [1] : (tensor<8x16xf32>, tensor<f32>) -> tensor<8xf32>
    return %1 : tensor<8xf32>
  }
}
