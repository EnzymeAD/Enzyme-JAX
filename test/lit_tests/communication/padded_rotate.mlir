// RUN: enzymexlamlir-opt --optimize-communication %s | FileCheck %s

// XFAIL: *

sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=4]>

func.func @main2(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<4x8x82xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg0 [8:12, 8:16, 8:90] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x96xf64>) -> tensor<4x8x82xf64>
    %1 = "enzymexla.rotate"(%0) <{amount = 2 : si32, dimension = 2 : si32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x8x82xf64>) -> tensor<4x8x82xf64>
    return %1 : tensor<4x8x82xf64>
}

// CHECK:   func.func @main2(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<4x8x82xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.slice %arg0 [8:12, 8:16, 8:90] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x96xf64>) -> tensor<4x8x82xf64>
// CHECK-NEXT:     %1 = stablehlo.pad %0, %cst, low = [0, 0, 0], high = [0, 0, 2], interior = [0, 0, 0] : (tensor<4x8x82xf64>, tensor<f64>) -> tensor<4x8x84xf64>
// CHECK-NEXT:     %2 = sdy.manual_computation(%1) in_shardings=[<@mesh1, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh1, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "x", "y"} (%arg1: tensor<4x2x21xf64>) {
// CHECK-NEXT:       %4 = stablehlo.slice %arg1 [0:4, 0:2, 0:4] : (tensor<4x2x21xf64>) -> tensor<4x2x4xf64>
// CHECK-NEXT{LITERAL}:       %5 = "stablehlo.collective_permute"(%4) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[4, 0], [5, 1], [6, 2], [7, 3], [8, 4], [9, 5], [10, 6], [11, 7], [12, 8], [13, 9], [14, 10], [15, 11], [0, 12], [1, 13], [2, 14], [3, 15]]> : tensor<16x2xi64>}> : (tensor<4x2x4xf64>) -> tensor<4x2x4xf64>
// CHECK-NEXT:       %6 = stablehlo.slice %arg1 [0:4, 0:2, 4:21] : (tensor<4x2x21xf64>) -> tensor<4x2x17xf64>
// CHECK-NEXT:       %7 = stablehlo.concatenate %6, %5, dim = 2 : (tensor<4x2x17xf64>, tensor<4x2x4xf64>) -> tensor<4x2x21xf64>
// CHECK-NEXT:       sdy.return %7 : tensor<4x2x21xf64>
// CHECK-NEXT:     } : (tensor<4x8x84xf64>) -> tensor<4x8x84xf64>
// CHECK-NEXT:     %3 = stablehlo.slice %2 [0:4, 0:8, 0:82] : (tensor<4x8x84xf64>) -> tensor<4x8x82xf64>
// CHECK-NEXT:     return %3 : tensor<4x8x82xf64>
// CHECK-NEXT:   }
