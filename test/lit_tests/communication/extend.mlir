// RUN: enzymexlamlir-opt --optimize-communication %s | FileCheck %s

sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=4]>
func.func @main(%arg0: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x10x82xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg0 [11:12, 7:17, 0:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>) -> tensor<1x10x80xf64>
    %1 = "enzymexla.extend"(%0) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 1 : i64}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x80xf64>) -> tensor<1x10x82xf64>
    return %1 : tensor<1x10x82xf64>
}

// CHECK-LABEL:   func.func @main
// CHECK: sdy.manual_computation
// CHECK: "stablehlo.if"
// CHECK: "stablehlo.if"
// CHECK{LITERAL}: %16 = "stablehlo.collective_permute"(%15) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[0, 4], [1, 5], [2, 6], [3, 7], [12, 8], [13, 9], [14, 10], [15, 11]]> : tensor<8x2xi64>}> : (tensor<1x2x1xf64>) -> tensor<1x2x1xf64>
// CHECK: stablehlo.dynamic_slice
// CHECK: }, {
// CHECK{LITERAL}: %16 = "stablehlo.collective_permute"(%15) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[0, 4], [1, 5], [2, 6], [3, 7], [12, 8], [13, 9], [14, 10], [15, 11]]> : tensor<8x2xi64>}> : (tensor<1x2x1xf64>) -> tensor<1x2x1xf64>
// CHECK: stablehlo.dynamic_slice
// CHECK: "stablehlo.if"
// CHECK: stablehlo.slice
// CHECK-NEXT: stablehlo.slice
// CHECK-NEXT: stablehlo.concatenate
// CHECK-NEXT: stablehlo.return
// CHECK-NEXT: }, {
// CHECK-NEXT: stablehlo.slice
// CHECK-NEXT: stablehlo.slice
// CHECK-NEXT: stablehlo.concatenate
// CHECK-NEXT: stablehlo.return
