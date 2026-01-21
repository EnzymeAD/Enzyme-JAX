// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{periodic_concat=0 concat_to_pad_comm=0 concat_to_dus=1 dus_to_pad_comm=0})" %s | FileCheck %s

sdy.mesh @mesh = <["z"=1, "x"=4, "y"=4]>
func.func @main1(%1589: tensor<4x1519x3056xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<4x1519x3057xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {

    %1594 = stablehlo.slice %1589 [0:4, 0:1519, 2:3056] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<4x1519x3056xf32>) -> tensor<4x1519x3054xf32>

    %1592 = stablehlo.slice %1589 [0:4, 0:1519, 3053:3056] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<4x1519x3056xf32>) -> tensor<4x1519x3xf32>

    %3394 = stablehlo.concatenate %1592, %1594, dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<4x1519x3xf32>, tensor<4x1519x3054xf32>) -> tensor<4x1519x3057xf32>
    return %3394 : tensor<4x1519x3057xf32>
}

// CHECK:  func.func @main1(%arg0: tensor<4x1519x3056xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<4x1519x3057xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:4, 0:1519, 3053:3056] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<4x1519x3056xf32>) -> tensor<4x1519x3xf32>
// CHECK-NEXT:    %1 = stablehlo.pad %arg0, %cst, low = [0, 0, 1], high = [0, 0, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<4x1519x3056xf32>, tensor<f32>) -> tensor<4x1519x3057xf32>
// CHECK-NEXT:    %2 = stablehlo.dynamic_update_slice %1, %0, %c, %c, %c {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<4x1519x3057xf32>, tensor<4x1519x3xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x1519x3057xf32>
// CHECK-NEXT:    return %2 : tensor<4x1519x3057xf32>
// CHECK-NEXT:  }
