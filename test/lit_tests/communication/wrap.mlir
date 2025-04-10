// RUN: enzymexlamlir-opt --optimize-communication %s | FileCheck %s

sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=4]>
func.func @main(%arg0: tensor<1x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x8x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg0 [0:1, 0:8, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x24x96xf64>) -> tensor<1x8x80xf64>
    %1 = "enzymexla.wrap"(%0) <{dimension = 2 : i64, lhs = 8 : i64, rhs = 8 : i64}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x8x80xf64>) -> tensor<1x8x96xf64>
    return %1 : tensor<1x8x96xf64>
}

// CHECK-LABEL:   func.func @main
// CHECK: sdy.manual_computation
// CHECK: "stablehlo.if"
// CHECK: "stablehlo.if"
// CHECK{LITERAL}: %15 = "stablehlo.collective_permute"(%14) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[0, 4], [1, 5], [2, 6], [3, 7], [12, 8], [13, 9], [14, 10], [15, 11]]> : tensor<8x2xi64>}> : (tensor<1x2x4xf64>) -> tensor<1x2x4xf64>
// CHECK: stablehlo.dynamic_slice
// CHECK: }, {
// CHECK{LITERAL}: %15 = "stablehlo.collective_permute"(%14) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[0, 4], [1, 5], [2, 6], [3, 7], [12, 8], [13, 9], [14, 10], [15, 11]]> : tensor<8x2xi64>}> : (tensor<1x2x4xf64>) -> tensor<1x2x4xf64>
// CHECK: stablehlo.dynamic_slice
// CHECK: }, {
// CHECK{LITERAL}: %13 = "stablehlo.collective_permute"(%12) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[0, 12], [1, 13], [2, 14], [3, 15]]> : tensor<4x2xi64>}> : (tensor<1x2x8xf64>) -> tensor<1x2x8xf64>
// CHECK{LITERAL}: %15 = "stablehlo.collective_permute"(%14) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[12, 0], [13, 1], [14, 2], [15, 3]]> : tensor<4x2xi64>}> : (tensor<1x2x8xf64>) -> tensor<1x2x8xf64>
// CHECK: "stablehlo.if"
// CHECK: stablehlo.slice
// CHECK-NEXT: stablehlo.concatenate
// CHECK-NEXT: stablehlo.return
// CHECK-NEXT: }, {
// CHECK-NEXT: stablehlo.slice
// CHECK-NEXT: stablehlo.concatenate
// CHECK-NEXT: stablehlo.return
