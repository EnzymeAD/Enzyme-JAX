// RUN: enzymexlamlir-opt --split-input-file --sdy-propagation-pipeline --sdy-export-pipeline='enable-insert-explicit-collectives=true' --xla-sdy-stablehlo-export-pipeline %s | FileCheck %s

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func.func @bcast_constant_2x1x1x1_to_2x4x16x16
func.func @bcast_constant_2x1x1x1_to_2x4x16x16(
    %arg0: tensor<2x4x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}, {}]>})
    -> (tensor<2x4x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}, {}]>}) {
  // CHECK-NOT: all_gather
  // CHECK-NOT: collective_permute
  // CHECK-NOT: all_to_all
  // CHECK-NOT: mhlo.copy
  // CHECK: %[[CST:.*]] = stablehlo.constant {mhlo.sharding = "{devices=[2,1,1,1,2]<=[4] last_tile_dim_replicate}"} dense<{{\[\[}}{{\[\[}}1.000000e+00{{\]\]\], \[\[\[}}2.000000e+00{{\]\]\]\]}}> : tensor<2x1x1x1xf32>
  // CHECK-NEXT: %[[SQ:.*]] = stablehlo.reshape %[[CST]] {mhlo.sharding = "{devices=[2,2]<=[4] last_tile_dim_replicate}"} : (tensor<2x1x1x1xf32>) -> tensor<2xf32>
  // CHECK-NEXT: %[[BC:.*]] = stablehlo.broadcast_in_dim %[[SQ]], dims = [0] {mhlo.sharding = "{devices=[2,2,1,1]<=[4]}"} : (tensor<2xf32>) -> tensor<2x4x16x16xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %[[BC]] {mhlo.sharding = "{devices=[2,2,1,1]<=[4]}"} : tensor<2x4x16x16xf32>
  // CHECK-NEXT: return %[[ADD]] : tensor<2x4x16x16xf32>
  %0 = stablehlo.constant dense<[[[[1.0]]], [[[2.0]]]]> : tensor<2x1x1x1xf32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3] : (tensor<2x1x1x1xf32>) -> tensor<2x4x16x16xf32>
  %2 = stablehlo.add %arg0, %1 : tensor<2x4x16x16xf32>
  return %2 : tensor<2x4x16x16xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func.func @bcast_constant_1x4x16x16_to_2x4x16x16
func.func @bcast_constant_1x4x16x16_to_2x4x16x16(
    %arg0: tensor<2x4x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}, {}]>})
    -> (tensor<2x4x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}, {}]>}) {
  // CHECK-NOT: all_gather
  // CHECK-NOT: collective_permute
  // CHECK-NOT: all_to_all
  // CHECK-NOT: mhlo.copy
  // CHECK: %[[CST:.*]] = stablehlo.constant {mhlo.sharding = "{devices=[1,2,1,1,2]<=[2,2]T(1,0) last_tile_dim_replicate}"} dense<1.000000e+00> : tensor<1x4x16x16xf32>
  // CHECK-NEXT: %[[SQ:.*]] = stablehlo.reshape %[[CST]] {mhlo.sharding = "{devices=[2,1,1,2]<=[2,2]T(1,0) last_tile_dim_replicate}"} : (tensor<1x4x16x16xf32>) -> tensor<4x16x16xf32>
  // CHECK-NEXT: %[[BC:.*]] = stablehlo.broadcast_in_dim %[[SQ]], dims = [1, 2, 3] {mhlo.sharding = "{devices=[2,2,1,1]<=[4]}"} : (tensor<4x16x16xf32>) -> tensor<2x4x16x16xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %[[BC]] {mhlo.sharding = "{devices=[2,2,1,1]<=[4]}"} : tensor<2x4x16x16xf32>
  // CHECK-NEXT: return %[[ADD]] : tensor<2x4x16x16xf32>
  %0 = stablehlo.constant dense<1.0> : tensor<1x4x16x16xf32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3] : (tensor<1x4x16x16xf32>) -> tensor<2x4x16x16xf32>
  %2 = stablehlo.add %arg0, %1 : tensor<2x4x16x16xf32>
  return %2 : tensor<2x4x16x16xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func.func @bcast_constant_1x1x1x1_to_2x4x16x16
func.func @bcast_constant_1x1x1x1_to_2x4x16x16(
    %arg0: tensor<2x4x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}, {}]>})
    -> (tensor<2x4x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}, {}]>}) {
  // CHECK-NOT: all_gather
  // CHECK-NOT: collective_permute
  // CHECK-NOT: all_to_all
  // CHECK-NOT: mhlo.copy
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<3.000000e+00> : tensor<1x1x1x1xf32>
  // CHECK-NEXT: %[[SQ:.*]] = stablehlo.reshape %[[CST]] {mhlo.sharding = "{replicated}"} : (tensor<1x1x1x1xf32>) -> tensor<f32>
  // CHECK-NEXT: %[[BC:.*]] = stablehlo.broadcast_in_dim %[[SQ]], dims = [] {mhlo.sharding = "{devices=[2,2,1,1]<=[4]}"} : (tensor<f32>) -> tensor<2x4x16x16xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %[[BC]] {mhlo.sharding = "{devices=[2,2,1,1]<=[4]}"} : tensor<2x4x16x16xf32>
  // CHECK-NEXT: return %[[ADD]] : tensor<2x4x16x16xf32>
  %0 = stablehlo.constant dense<3.0> : tensor<1x1x1x1xf32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xf32>) -> tensor<2x4x16x16xf32>
  %2 = stablehlo.add %arg0, %1 : tensor<2x4x16x16xf32>
  return %2 : tensor<2x4x16x16xf32>
}
