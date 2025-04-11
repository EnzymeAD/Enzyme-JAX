// RUN: enzymexlamlir-opt --optimize-communication %s | FileCheck %s

sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=4]>
func.func @main1(%arg0: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x83xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg1 [0:20, 0:24, 0:3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>) -> tensor<20x24x3xf64>
    %1 = stablehlo.concatenate %arg0, %0, dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>, tensor<20x24x3xf64>) -> tensor<20x24x83xf64>
    return %1 : tensor<20x24x83xf64>
}

func.func @main2(%arg0: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x81xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg1 [0:20, 0:24, 0:1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>) -> tensor<20x24x1xf64>
    %1 = stablehlo.concatenate %0, %arg0, dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x1xf64>, tensor<20x24x80xf64>) -> tensor<20x24x81xf64>
    return %1 : tensor<20x24x81xf64>
}

// CHECK: sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=4]>
// CHECK-NEXT:  func.func @main1(%arg0: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x83xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [0:20, 0:24, 0:3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>) -> tensor<20x24x3xf64>
// CHECK-NEXT:    %1 = stablehlo.pad %0, %cst, low = [0, 0, 12], high = [0, 0, 1], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x3xf64>, tensor<f64>) -> tensor<20x24x16xf64>
// CHECK-NEXT:    %2 = sdy.manual_computation(%1, %arg0) in_shardings=[<@mesh1, [{"z"}, {"y"}, {"x"}]>, <@mesh1, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh1, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "x", "y"} (%arg2: tensor<20x6x4xf64>, %arg3: tensor<20x6x20xf64>) {
// CHECK-NEXT:      %c = stablehlo.constant dense<4> : tensor<ui32>
// CHECK-NEXT:      %c_0 = stablehlo.constant dense<1> : tensor<ui32>
// CHECK-NEXT:      %c_1 = stablehlo.constant dense<0> : tensor<ui32>
// CHECK-NEXT:      %4 = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:      %5 = stablehlo.add %4, %c_0 : tensor<ui32>
// CHECK-NEXT:      %6 = stablehlo.remainder %5, %c : tensor<ui32>
// CHECK-NEXT:      %7 = stablehlo.compare  EQ, %6, %c_1 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %8 = stablehlo.slice %arg3 [0:20, 0:6, 0:4] : (tensor<20x6x20xf64>) -> tensor<20x6x4xf64>
// CHECK-NEXT{LITERAL}:      %9 = "stablehlo.collective_permute"(%8) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[4, 0], [5, 1], [6, 2], [7, 3], [8, 4], [9, 5], [10, 6], [11, 7], [12, 8], [13, 9], [14, 10], [15, 11], [0, 12], [1, 13], [2, 14], [3, 15]]> : tensor<16x2xi64>}> : (tensor<20x6x4xf64>) -> tensor<20x6x4xf64>
// CHECK-NEXT:      %10 = "stablehlo.if"(%7) ({
// CHECK-NEXT:        stablehlo.return %arg2 : tensor<20x6x4xf64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        stablehlo.return %9 : tensor<20x6x4xf64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<20x6x4xf64>
// CHECK-NEXT:      %11 = stablehlo.concatenate %arg3, %10, dim = 2 : (tensor<20x6x20xf64>, tensor<20x6x4xf64>) -> tensor<20x6x24xf64>
// CHECK-NEXT:      %12 = stablehlo.add %4, %c_0 : tensor<ui32>
// CHECK-NEXT:      %13 = stablehlo.multiply %12, %c_0 : tensor<ui32>
// CHECK-NEXT:      %14 = stablehlo.subtract %c, %13 : tensor<ui32>
// CHECK-NEXT:      %15 = stablehlo.dynamic_slice %11, %c_1, %c_1, %14, sizes = [20, 6, 21] : (tensor<20x6x24xf64>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<20x6x21xf64>
// CHECK-NEXT:      sdy.return %15 : tensor<20x6x21xf64>
// CHECK-NEXT:    } : (tensor<20x24x16xf64>, tensor<20x24x80xf64>) -> tensor<20x24x84xf64>
// CHECK-NEXT:    %3 = stablehlo.slice %2 [0:20, 0:24, 0:83] : (tensor<20x24x84xf64>) -> tensor<20x24x83xf64>
// CHECK-NEXT:    return %3 : tensor<20x24x83xf64>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @main2(%arg0: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x81xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [0:20, 0:24, 0:1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>) -> tensor<20x24x1xf64>
// CHECK-NEXT:    %1 = stablehlo.pad %0, %cst, low = [0, 0, 3], high = [0, 0, 12], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x1xf64>, tensor<f64>) -> tensor<20x24x16xf64>
// CHECK-NEXT:    %2 = sdy.manual_computation(%1, %arg0) in_shardings=[<@mesh1, [{"z"}, {"y"}, {"x"}]>, <@mesh1, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh1, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "x", "y"} (%arg2: tensor<20x6x4xf64>, %arg3: tensor<20x6x20xf64>) {
// CHECK-NEXT:      %c = stablehlo.constant dense<4> : tensor<ui32>
// CHECK-NEXT:      %c_0 = stablehlo.constant dense<1> : tensor<ui32>
// CHECK-NEXT:      %c_1 = stablehlo.constant dense<0> : tensor<ui32>
// CHECK-NEXT:      %4 = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:      %5 = stablehlo.remainder %4, %c : tensor<ui32>
// CHECK-NEXT:      %6 = stablehlo.compare  EQ, %5, %c_1 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %7 = stablehlo.slice %arg3 [0:20, 0:6, 16:20] : (tensor<20x6x20xf64>) -> tensor<20x6x4xf64>
// CHECK-NEXT{LITERAL}:      %8 = "stablehlo.collective_permute"(%7) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[12, 0], [13, 1], [14, 2], [15, 3], [0, 4], [1, 5], [2, 6], [3, 7], [4, 8], [5, 9], [6, 10], [7, 11], [8, 12], [9, 13], [10, 14], [11, 15]]> : tensor<16x2xi64>}> : (tensor<20x6x4xf64>) -> tensor<20x6x4xf64>
// CHECK-NEXT:      %9 = "stablehlo.if"(%6) ({
// CHECK-NEXT:        stablehlo.return %arg2 : tensor<20x6x4xf64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        stablehlo.return %8 : tensor<20x6x4xf64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<20x6x4xf64>
// CHECK-NEXT:      %10 = stablehlo.concatenate %9, %arg3, dim = 2 : (tensor<20x6x4xf64>, tensor<20x6x20xf64>) -> tensor<20x6x24xf64>
// CHECK-NEXT:      %11 = stablehlo.subtract %4, %c_0 : tensor<ui32>
// CHECK-NEXT:      %12 = stablehlo.multiply %11, %c_0 : tensor<ui32>
// CHECK-NEXT:      %13 = stablehlo.dynamic_slice %10, %c_1, %c_1, %12, sizes = [20, 6, 21] : (tensor<20x6x24xf64>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<20x6x21xf64>
// CHECK-NEXT:      sdy.return %13 : tensor<20x6x21xf64>
// CHECK-NEXT:    } : (tensor<20x24x16xf64>, tensor<20x24x80xf64>) -> tensor<20x24x84xf64>
// CHECK-NEXT:    %3 = stablehlo.slice %2 [0:20, 0:24, 3:84] : (tensor<20x24x84xf64>) -> tensor<20x24x81xf64>
// CHECK-NEXT:    return %3 : tensor<20x24x81xf64>
// CHECK-NEXT:  }

