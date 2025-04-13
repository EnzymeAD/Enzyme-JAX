// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{extend_comm=1 extend_to_pad_comm=0})" %s | FileCheck %s

sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=6]>
func.func @main(%arg0: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x10x82xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg0 [11:12, 7:17, 0:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>) -> tensor<1x10x80xf64>
    %1 = "enzymexla.extend"(%0) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 1 : i64}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x80xf64>) -> tensor<1x10x82xf64>
    return %1 : tensor<1x10x82xf64>
}

// CHECK:  func.func @main(%arg0: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x10x82xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [11:12, 7:17, 0:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>) -> tensor<1x10x80xf64>
// CHECK-NEXT:    %1 = stablehlo.pad %0, %cst, low = [0, 0, 0], high = [0, 2, 0], interior = [0, 0, 0] : (tensor<1x10x80xf64>, tensor<f64>) -> tensor<1x12x80xf64>
// CHECK-NEXT:    %2 = sdy.manual_computation(%1) in_shardings=[<@mesh1, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh1, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "x", "y"} (%arg1: tensor<1x2x20xf64>) {
// CHECK-NEXT:      %c = stablehlo.constant dense<2> : tensor<ui32>
// CHECK-NEXT:      %c_0 = stablehlo.constant dense<1> : tensor<ui32>
// CHECK-NEXT:      %c_1 = stablehlo.constant dense<4> : tensor<ui32>
// CHECK-NEXT:      %c_2 = stablehlo.constant dense<0> : tensor<ui32>
// CHECK-NEXT:      %4 = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:      %5 = stablehlo.remainder %4, %c_1 : tensor<ui32>
// CHECK-NEXT:      %6 = stablehlo.compare  EQ, %5, %c_2 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %7 = stablehlo.add %4, %c_0 : tensor<ui32>
// CHECK-NEXT:      %8 = stablehlo.remainder %7, %c_1 : tensor<ui32>
// CHECK-NEXT:      %9 = stablehlo.compare  EQ, %8, %c_2 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %10 = stablehlo.not %6 : tensor<i1>
// CHECK-NEXT:      %11 = stablehlo.not %9 : tensor<i1>
// CHECK-NEXT:      %12 = stablehlo.and %10, %11 : tensor<i1>
// CHECK-NEXT:      %13 = "stablehlo.if"(%12) ({
// CHECK-NEXT:        %14 = stablehlo.remainder %4, %c_1 : tensor<ui32>
// CHECK-NEXT:        %15 = stablehlo.compare  LT, %5, %c : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %16 = "stablehlo.if"(%15) ({
// CHECK-NEXT:          %19 = stablehlo.slice %arg1 [0:1, 0:2, 18:20] : (tensor<1x2x20xf64>) -> tensor<1x2x2xf64>
// CHECK-NEXT:          stablehlo.return %19 : tensor<1x2x2xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %19 = stablehlo.slice %arg1 [0:1, 0:2, 0:2] : (tensor<1x2x20xf64>) -> tensor<1x2x2xf64>
// CHECK-NEXT:          stablehlo.return %19 : tensor<1x2x2xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<1x2x2xf64>
// CHECK-NEXT{LITERAL}:        %17 = "stablehlo.collective_permute"(%16) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11], [18, 12], [19, 13], [20, 14], [21, 15], [22, 16], [23, 17]]> : tensor<12x2xi64>}> : (tensor<1x2x2xf64>) -> tensor<1x2x2xf64>
// CHECK-NEXT:        %18 = "stablehlo.if"(%15) ({
// CHECK-NEXT:          %19 = stablehlo.concatenate %17, %arg1, dim = 2 : (tensor<1x2x2xf64>, tensor<1x2x20xf64>) -> tensor<1x2x22xf64>
// CHECK-NEXT:          %20 = stablehlo.multiply %14, %c_0 : tensor<ui32>
// CHECK-NEXT:          %21 = stablehlo.dynamic_slice %19, %c_2, %c_2, %20, sizes = [1, 2, 21] : (tensor<1x2x22xf64>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<1x2x21xf64>
// CHECK-NEXT:          stablehlo.return %21 : tensor<1x2x21xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %19 = stablehlo.concatenate %arg1, %17, dim = 2 : (tensor<1x2x20xf64>, tensor<1x2x2xf64>) -> tensor<1x2x22xf64>
// CHECK-NEXT:          %20 = stablehlo.multiply %14, %c_0 : tensor<ui32>
// CHECK-NEXT:          %21 = stablehlo.subtract %c_0, %20 : tensor<ui32>
// CHECK-NEXT:          %22 = stablehlo.dynamic_slice %19, %c_2, %c_2, %21, sizes = [1, 2, 21] : (tensor<1x2x22xf64>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<1x2x21xf64>
// CHECK-NEXT:          stablehlo.return %22 : tensor<1x2x21xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<1x2x21xf64>
// CHECK-NEXT:        stablehlo.return %18 : tensor<1x2x21xf64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        %14 = "stablehlo.if"(%6) ({
// CHECK-NEXT:          %15 = stablehlo.slice %arg1 [0:1, 0:2, 0:2] : (tensor<1x2x20xf64>) -> tensor<1x2x2xf64>
// CHECK-NEXT:          %16 = stablehlo.slice %arg1 [0:1, 0:2, 0:19] : (tensor<1x2x20xf64>) -> tensor<1x2x19xf64>
// CHECK-NEXT:          %17 = stablehlo.concatenate %15, %16, dim = 2 : (tensor<1x2x2xf64>, tensor<1x2x19xf64>) -> tensor<1x2x21xf64>
// CHECK-NEXT:          stablehlo.return %17 : tensor<1x2x21xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %15 = stablehlo.slice %arg1 [0:1, 0:2, 18:20] : (tensor<1x2x20xf64>) -> tensor<1x2x2xf64>
// CHECK-NEXT:          %16 = stablehlo.slice %arg1 [0:1, 0:2, 1:20] : (tensor<1x2x20xf64>) -> tensor<1x2x19xf64>
// CHECK-NEXT:          %17 = stablehlo.concatenate %16, %15, dim = 2 : (tensor<1x2x19xf64>, tensor<1x2x2xf64>) -> tensor<1x2x21xf64>
// CHECK-NEXT:          stablehlo.return %17 : tensor<1x2x21xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<1x2x21xf64>
// CHECK-NEXT:        stablehlo.return %14 : tensor<1x2x21xf64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<1x2x21xf64>
// CHECK-NEXT:      sdy.return %13 : tensor<1x2x21xf64>
// CHECK-NEXT:    } : (tensor<1x12x80xf64>) -> tensor<1x12x84xf64>
// CHECK-NEXT:    %3 = stablehlo.slice %2 [0:1, 0:10, 1:83] : (tensor<1x12x84xf64>) -> tensor<1x10x82xf64>
// CHECK-NEXT:    return %3 : tensor<1x10x82xf64>
// CHECK-NEXT:  }