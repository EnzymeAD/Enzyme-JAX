// RUN: enzymexlamlir-opt --sdy-insert-explicit-reshards='enable-full-version=true' %s | FileCheck %s

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: func.func @reshape_insert_unsharded_singleton_single_reshard
func.func @reshape_insert_unsharded_singleton_single_reshard(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
    -> (tensor<1x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}]>}) {
  // CHECK-NEXT: %[[RESHAPE:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"a"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<1x8x8xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[RESHAPE]] <@mesh, [{}, {}, {"a"}]> : tensor<1x8x8xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<1x8x8xf32>
  %0 = stablehlo.reshape %arg0 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {"a"}]>]>
  } : (tensor<8x8xf32>) -> tensor<1x8x8xf32>
  return %0 : tensor<1x8x8xf32>
}

// CHECK-LABEL: func.func @reshape_remove_unsharded_singleton_single_reshard
func.func @reshape_remove_unsharded_singleton_single_reshard(
    %arg0: tensor<1x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}, {}]>})
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}]>}) {
  // CHECK-NEXT: %[[RESHAPE:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<1x8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[RESHAPE]] <@mesh, [{}, {"a"}]> : tensor<8x8xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x8xf32>
  %0 = stablehlo.reshape %arg0 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"a"}]>]>
  } : (tensor<1x8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
