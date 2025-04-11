// RUN: enzymexlamlir-opt --optimize-communication %s | FileCheck %s

sdy.mesh @mesh = <["x"=2, "y"=1, "z"=1]>
func.func @wrap(%7175 : tensor<3x10x80xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> tensor<3x10x82xf64> {
    %7542 = "enzymexla.wrap"(%7175) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 1 : i64}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<3x10x80xf64>) -> tensor<3x10x82xf64>
    stablehlo.return %7542 : tensor<3x10x82xf64>
}

// CHECK:  sdy.mesh @mesh = <["x"=2, "y"=1, "z"=1]>
// CHECK-NEXT:  func.func @wrap(%arg0: tensor<3x10x80xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> tensor<3x10x82xf64> {
// CHECK-NEXT:    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh, [{"z"}, {"y"}, {"x"}]>] manual_axes={"x", "y", "z"} (%arg1: tensor<3x10x40xf64>) {
// CHECK-NEXT:      %c = stablehlo.constant dense<2> : tensor<ui32>
// CHECK-NEXT:      %c_0 = stablehlo.constant dense<0> : tensor<ui32>
// CHECK-NEXT:      %1 = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:      %2 = stablehlo.remainder %1, %c : tensor<ui32>
// CHECK-NEXT:      %3 = stablehlo.compare  EQ, %2, %c_0 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %4 = "stablehlo.if"(%3) ({
// CHECK-NEXT:        %7 = stablehlo.slice %arg1 [0:3, 0:10, 0:1] : (tensor<3x10x40xf64>) -> tensor<3x10x1xf64>
// CHECK-NEXT:        stablehlo.return %7 : tensor<3x10x1xf64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        %7 = stablehlo.slice %arg1 [0:3, 0:10, 39:40] : (tensor<3x10x40xf64>) -> tensor<3x10x1xf64>
// CHECK-NEXT:        stablehlo.return %7 : tensor<3x10x1xf64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<3x10x1xf64>
// CHECK-NEXT{LITERAL}:      %5 = "stablehlo.collective_permute"(%4) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[0, 1], [1, 0]]> : tensor<2x2xi64>}> : (tensor<3x10x1xf64>) -> tensor<3x10x1xf64>
// CHECK-NEXT:      %6 = "stablehlo.if"(%3) ({
// CHECK-NEXT:        %7 = stablehlo.concatenate %5, %arg1, dim = 2 : (tensor<3x10x1xf64>, tensor<3x10x40xf64>) -> tensor<3x10x41xf64>
// CHECK-NEXT:        stablehlo.return %7 : tensor<3x10x41xf64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        %7 = stablehlo.concatenate %arg1, %5, dim = 2 : (tensor<3x10x40xf64>, tensor<3x10x1xf64>) -> tensor<3x10x41xf64>
// CHECK-NEXT:        stablehlo.return %7 : tensor<3x10x41xf64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<3x10x41xf64>
// CHECK-NEXT:      sdy.return %6 : tensor<3x10x41xf64>
// CHECK-NEXT:    } : (tensor<3x10x80xf64>) -> tensor<3x10x82xf64>
// CHECK-NEXT:    stablehlo.return %0 : tensor<3x10x82xf64>
// CHECK-NEXT:  }
