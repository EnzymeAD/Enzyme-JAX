// RUN: enzymexlamlir-opt --optimize-communication %s | FileCheck %s

sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=4]>
func.func @main1(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<4x8x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg0 [8:12, 8:16, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x96xf64>) -> tensor<4x8x80xf64>
    %1 = "enzymexla.rotate"(%0) <{amount = 2 : si32, dimension = 2 : si32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x8x80xf64>) -> tensor<4x8x80xf64>
    return %1 : tensor<4x8x80xf64>
}

func.func @main3(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<4x8x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg0 [8:12, 8:16, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x96xf64>) -> tensor<4x8x80xf64>
    %1 = "enzymexla.rotate"(%0) <{amount = 2 : si32, dimension = 1 : si32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x8x80xf64>) -> tensor<4x8x80xf64>
    return %1 : tensor<4x8x80xf64>
}

// CHECK: module {
// CHECK-NEXT:   sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=4]>
// CHECK-NEXT:   func.func @main1(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<4x8x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:     %0 = stablehlo.slice %arg0 [8:12, 8:16, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x96xf64>) -> tensor<4x8x80xf64>
// CHECK-NEXT:     %1 = sdy.manual_computation(%0) in_shardings=[<@mesh1, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh1, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "x", "y"} (%arg1: tensor<4x2x20xf64>) {
// CHECK-NEXT:       %2 = stablehlo.slice %arg1 [0:4, 0:2, 0:2] : (tensor<4x2x20xf64>) -> tensor<4x2x2xf64>
// CHECK-NEXT{LITERAL}:       %3 = "stablehlo.collective_permute"(%2) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[4, 0], [5, 1], [6, 2], [7, 3], [8, 4], [9, 5], [10, 6], [11, 7], [12, 8], [13, 9], [14, 10], [15, 11], [0, 12], [1, 13], [2, 14], [3, 15]]> : tensor<16x2xi64>}> : (tensor<4x2x2xf64>) -> tensor<4x2x2xf64>
// CHECK-NEXT:       %4 = stablehlo.slice %arg1 [0:4, 0:2, 2:20] : (tensor<4x2x20xf64>) -> tensor<4x2x18xf64>
// CHECK-NEXT:       %5 = stablehlo.concatenate %4, %3, dim = 2 : (tensor<4x2x18xf64>, tensor<4x2x2xf64>) -> tensor<4x2x20xf64>
// CHECK-NEXT:       sdy.return %5 : tensor<4x2x20xf64>
// CHECK-NEXT:     } : (tensor<4x8x80xf64>) -> tensor<4x8x80xf64>
// CHECK-NEXT:     return %1 : tensor<4x8x80xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @main3(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<4x8x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:      %0 = stablehlo.slice %arg0 [8:12, 8:16, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x96xf64>) -> tensor<4x8x80xf64>
// CHECK-NEXT:      %1 = sdy.manual_computation(%0) in_shardings=[<@mesh1, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh1, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "x", "y"} (%arg1: tensor<4x2x20xf64>) {
// CHECK-NEXT:        %2 = stablehlo.slice %arg1 [0:4, 0:2, 0:20] : (tensor<4x2x20xf64>) -> tensor<4x2x20xf64>
// CHECK-NEXT{LITERAL}:        %3 = "stablehlo.collective_permute"(%2) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[1, 0], [2, 1], [3, 2], [0, 3], [5, 4], [6, 5], [7, 6], [4, 7], [9, 8], [10, 9], [11, 10], [8, 11], [13, 12], [14, 13], [15, 14], [12, 15]]> : tensor<16x2xi64>}> : (tensor<4x2x20xf64>) -> tensor<4x2x20xf64>
// CHECK-NEXT:        %4 = stablehlo.slice %arg1 [0:4, 2:2, 0:20] : (tensor<4x2x20xf64>) -> tensor<4x0x20xf64>
// CHECK-NEXT:        %5 = stablehlo.concatenate %4, %3, dim = 1 : (tensor<4x0x20xf64>, tensor<4x2x20xf64>) -> tensor<4x2x20xf64>
// CHECK-NEXT:        sdy.return %5 : tensor<4x2x20xf64>
// CHECK-NEXT:      } : (tensor<4x8x80xf64>) -> tensor<4x8x80xf64>
// CHECK-NEXT:      return %1 : tensor<4x8x80xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
