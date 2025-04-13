// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{wrap_comm=1 wrap_to_pad_comm=0})" %s | FileCheck %s

sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=5]>
func.func @main(%arg0: tensor<1x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x8x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg0 [0:1, 0:8, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x24x96xf64>) -> tensor<1x8x80xf64>
    %1 = "enzymexla.wrap"(%0) <{dimension = 2 : i64, lhs = 8 : i64, rhs = 8 : i64}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x8x80xf64>) -> tensor<1x8x96xf64>
    return %1 : tensor<1x8x96xf64>
}

// CHECK:  func.func @main(%arg0: tensor<1x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x8x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 0:8, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x24x96xf64>) -> tensor<1x8x80xf64>
// CHECK-NEXT:    %1 = stablehlo.pad %0, %cst, low = [0, 0, 0], high = [0, 2, 0], interior = [0, 0, 0] : (tensor<1x8x80xf64>, tensor<f64>) -> tensor<1x10x80xf64>
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
// CHECK-NEXT:          %19 = stablehlo.slice %arg1 [0:1, 0:2, 12:20] : (tensor<1x2x20xf64>) -> tensor<1x2x8xf64>
// CHECK-NEXT:          stablehlo.return %19 : tensor<1x2x8xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %19 = stablehlo.slice %arg1 [0:1, 0:2, 0:8] : (tensor<1x2x20xf64>) -> tensor<1x2x8xf64>
// CHECK-NEXT:          stablehlo.return %19 : tensor<1x2x8xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<1x2x8xf64>
// CHECK-NEXT{LITERAL}:        %17 = "stablehlo.collective_permute"(%16) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[0, 5], [1, 6], [2, 7], [3, 8], [4, 9], [15, 10], [16, 11], [17, 12], [18, 13], [19, 14]]> : tensor<10x2xi64>}> : (tensor<1x2x8xf64>) -> tensor<1x2x8xf64>
// CHECK-NEXT:        %18 = "stablehlo.if"(%15) ({
// CHECK-NEXT:          %19 = stablehlo.concatenate %17, %arg1, dim = 2 : (tensor<1x2x8xf64>, tensor<1x2x20xf64>) -> tensor<1x2x28xf64>
// CHECK-NEXT:          %20 = stablehlo.multiply %14, %c_1 : tensor<ui32>
// CHECK-NEXT:          %21 = stablehlo.dynamic_slice %19, %c_2, %c_2, %20, sizes = [1, 2, 24] : (tensor<1x2x28xf64>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<1x2x24xf64>
// CHECK-NEXT:          stablehlo.return %21 : tensor<1x2x24xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %19 = stablehlo.concatenate %arg1, %17, dim = 2 : (tensor<1x2x20xf64>, tensor<1x2x8xf64>) -> tensor<1x2x28xf64>
// CHECK-NEXT:          %20 = stablehlo.multiply %14, %c_1 : tensor<ui32>
// CHECK-NEXT:          %21 = stablehlo.subtract %c_1, %20 : tensor<ui32>
// CHECK-NEXT:          %22 = stablehlo.dynamic_slice %19, %c_2, %c_2, %21, sizes = [1, 2, 24] : (tensor<1x2x28xf64>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<1x2x24xf64>
// CHECK-NEXT:          stablehlo.return %22 : tensor<1x2x24xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<1x2x24xf64>
// CHECK-NEXT:        stablehlo.return %18 : tensor<1x2x24xf64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        %14 = "stablehlo.if"(%6) ({
// CHECK-NEXT:          %17 = stablehlo.slice %arg1 [0:1, 0:2, 0:8] : (tensor<1x2x20xf64>) -> tensor<1x2x8xf64>
// CHECK-NEXT:          stablehlo.return %17 : tensor<1x2x8xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %17 = stablehlo.slice %arg1 [0:1, 0:2, 12:20] : (tensor<1x2x20xf64>) -> tensor<1x2x8xf64>
// CHECK-NEXT:          stablehlo.return %17 : tensor<1x2x8xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<1x2x8xf64>
// CHECK-NEXT{LITERAL}:        %15 = "stablehlo.collective_permute"(%14) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 0>, source_target_pairs = dense<[[0, 15], [1, 16], [2, 17], [3, 18], [4, 19], [15, 0], [16, 1], [17, 2], [18, 3], [19, 4]]> : tensor<10x2xi64>}> : (tensor<1x2x8xf64>) -> tensor<1x2x8xf64>
// CHECK-NEXT:        %16 = "stablehlo.if"(%6) ({
// CHECK-NEXT:          %17 = stablehlo.slice %arg1 [0:1, 0:2, 0:16] : (tensor<1x2x20xf64>) -> tensor<1x2x16xf64>
// CHECK-NEXT:          %18 = stablehlo.concatenate %15, %17, dim = 2 : (tensor<1x2x8xf64>, tensor<1x2x16xf64>) -> tensor<1x2x24xf64>
// CHECK-NEXT:          stablehlo.return %18 : tensor<1x2x24xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %17 = stablehlo.slice %arg1 [0:1, 0:2, 4:20] : (tensor<1x2x20xf64>) -> tensor<1x2x16xf64>
// CHECK-NEXT:          %18 = stablehlo.concatenate %17, %15, dim = 2 : (tensor<1x2x16xf64>, tensor<1x2x8xf64>) -> tensor<1x2x24xf64>
// CHECK-NEXT:          stablehlo.return %18 : tensor<1x2x24xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<1x2x24xf64>
// CHECK-NEXT:        stablehlo.return %16 : tensor<1x2x24xf64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<1x2x24xf64>
// CHECK-NEXT:      sdy.return %13 : tensor<1x2x24xf64>
// CHECK-NEXT:    } : (tensor<1x10x80xf64>) -> tensor<1x10x96xf64>
// CHECK-NEXT:    %3 = stablehlo.slice %2 [0:1, 0:8, 0:96] : (tensor<1x10x96xf64>) -> tensor<1x8x96xf64>
// CHECK-NEXT:    return %3 : tensor<1x8x96xf64>
// CHECK-NEXT:  }
