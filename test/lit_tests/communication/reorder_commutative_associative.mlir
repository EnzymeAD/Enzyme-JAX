// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{reorder_associative=1})" %s | FileCheck %s

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

module @reactant_fn2 attributes {mhlo.num_partitions = 8 : i64, mhlo.num_replicas = 1 : i64} {
  sdy.mesh @mesh = <["x"=4, "y"=2]>
  func.func @main(%arg0: tensor<16x16x16xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {"y"}]>}, %arg1: tensor<16x16x16xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {"y"}]>}, %arg2: tensor<16x16x16xf64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x", "y"}, {}]>}) -> (tensor<16x16x16xf64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x", "y"}, {}]>}) {
    // CHECK: %0 = stablehlo.add %arg1, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {"y"}]>]>} : tensor<16x16x16xf64>
    %0 = stablehlo.add %arg2, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x", "y"}, {}]>]>} : tensor<16x16x16xf64>
    // CHECK-NEXT: %1 = stablehlo.add %0, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x", "y"}, {}]>]>} : tensor<16x16x16xf64>
    %1 = stablehlo.add %arg1, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x", "y"}, {}]>]>} : tensor<16x16x16xf64>
    return %1 : tensor<16x16x16xf64>
  }
}

module @reactant_fn3 attributes {mhlo.num_partitions = 8 : i64, mhlo.num_replicas = 1 : i64} {
  sdy.mesh @mesh = <["x"=4, "y"=2]>
  func.func @main(%arg0: tensor<16x16x16xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {"y"}]>}, %arg1: tensor<16x16x16xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {"y"}]>}, %arg2: tensor<16x16x16xf64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x", "y"}, {}]>}, %arg3: tensor<16x16x16xf64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x", "y"}, {}]>}) -> (tensor<16x16x16xf64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x", "y"}, {}]>}) {
    // CHECK: %0 = stablehlo.add %arg2, %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x", "y"}, {}]>]>} : tensor<16x16x16xf64>
    // CHECK-NEXT: %1 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {"y"}]>]>} : tensor<16x16x16xf64>
    // CHECK-NEXT: %2 = stablehlo.add %0, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x", "y"}, {}]>]>} : tensor<16x16x16xf64>
    %0 = stablehlo.add %arg2, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x", "y"}, {}]>]>} : tensor<16x16x16xf64>
    %1 = stablehlo.add %arg3, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x", "y"}, {}]>]>} : tensor<16x16x16xf64>
    %2 = stablehlo.add %0, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x", "y"}, {}]>]>} : tensor<16x16x16xf64>
    return %2 : tensor<16x16x16xf64>
  }
}
