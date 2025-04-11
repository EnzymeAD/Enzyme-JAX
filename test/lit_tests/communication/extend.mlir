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

sdy.mesh @mesh2 = <["z"=1, "x"=2, "y"=4]>
func.func @main2(%arg0: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh2, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x10x82xf64> {sdy.sharding = #sdy.sharding<@mesh2, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg0 [11:12, 7:17, 0:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>) -> tensor<1x10x80xf64>
    %1 = "enzymexla.extend"(%0) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 1 : i64}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x80xf64>) -> tensor<1x10x82xf64>
    return %1 : tensor<1x10x82xf64>
}

// CHECK: sdy.mesh @mesh2 = <["z"=1, "x"=2, "y"=4]>
// CHECK-NEXT:  func.func @main2(%arg0: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh2, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x10x82xf64> {sdy.sharding = #sdy.sharding<@mesh2, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [11:12, 7:17, 0:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>) -> tensor<1x10x80xf64>
// CHECK-NEXT:    %1 = sdy.manual_computation(%0) in_shardings=[<@mesh2, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh2, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "x", "y"} (%arg1: tensor<1x2x40xf64>) {
// CHECK-NEXT:      %c = stablehlo.constant dense<2> : tensor<ui32>
// CHECK-NEXT:      %c_0 = stablehlo.constant dense<0> : tensor<ui32>
// CHECK-NEXT:      %2 = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:      %3 = stablehlo.remainder %2, %c : tensor<ui32>
// CHECK-NEXT:      %4 = stablehlo.compare  EQ, %3, %c_0 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %5 = "stablehlo.if"(%4) ({
// CHECK-NEXT:        %6 = stablehlo.slice %arg1 [0:1, 0:2, 0:1] : (tensor<1x2x40xf64>) -> tensor<1x2x1xf64>
// CHECK-NEXT:        %7 = stablehlo.concatenate %6, %arg1, dim = 2 : (tensor<1x2x1xf64>, tensor<1x2x40xf64>) -> tensor<1x2x41xf64>
// CHECK-NEXT:        stablehlo.return %7 : tensor<1x2x41xf64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        %6 = stablehlo.slice %arg1 [0:1, 0:2, 39:40] : (tensor<1x2x40xf64>) -> tensor<1x2x1xf64>
// CHECK-NEXT:        %7 = stablehlo.concatenate %arg1, %6, dim = 2 : (tensor<1x2x40xf64>, tensor<1x2x1xf64>) -> tensor<1x2x41xf64>
// CHECK-NEXT:        stablehlo.return %7 : tensor<1x2x41xf64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<1x2x41xf64>
// CHECK-NEXT:      sdy.return %5 : tensor<1x2x41xf64>
// CHECK-NEXT:    } : (tensor<1x10x80xf64>) -> tensor<1x10x82xf64>
// CHECK-NEXT:    return %1 : tensor<1x10x82xf64>
// CHECK-NEXT:  }
