// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{reorder_commutative_associative=1})" %s | FileCheck %s

module @reactant_fn attributes {mhlo.num_partitions = 8 : i64, mhlo.num_replicas = 1 : i64} {
  sdy.mesh @mesh = <["x"=4, "y"=2]>
  func.func @main(%arg0: tensor<16x16x16xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {"y"}]>}, %arg1: tensor<16x16x16xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {"y"}]>}, %arg2: tensor<16x16x16xf64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x", "y"}, {}]>}) -> (tensor<16x16x16xf64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x", "y"}, {}]>}) {
    // CHECK: %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {"y"}]>]>} : tensor<16x16x16xf64>
    %0 = stablehlo.add %arg2, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x", "y"}, {}]>]>} : tensor<16x16x16xf64>
    // CHECK-NEXT: %1 = stablehlo.add %arg2, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x", "y"}, {}]>]>} : tensor<16x16x16xf64>
    %1 = stablehlo.add %0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x", "y"}, {}]>]>} : tensor<16x16x16xf64>
    return %1 : tensor<16x16x16xf64>
  }
}
