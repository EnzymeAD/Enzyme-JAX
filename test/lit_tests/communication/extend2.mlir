// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{extend_dus_like=1 extend_comm=0 extend_to_pad_comm=0})" %s | FileCheck %s 

sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=4]>
func.func @main(%arg0: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x10x81xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg0 [11:12, 7:17, 0:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>) -> tensor<1x10x80xf64>
    %1 = "enzymexla.extend"(%0) <{dimension = 2 : i64, lhs = 0 : i64, rhs = 1 : i64}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x80xf64>) -> tensor<1x10x81xf64>
    return %1 : tensor<1x10x81xf64>
}


func.func @main2(%arg0: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x10x81xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg0 [11:12, 7:17, 0:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>) -> tensor<1x10x80xf64>
    %1 = "enzymexla.extend"(%0) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 0 : i64}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x80xf64>) -> tensor<1x10x81xf64>
    return %1 : tensor<1x10x81xf64>
}

// CHECK:  func.func @main(%arg0: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x10x81xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [11:12, 7:17, 0:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>) -> tensor<1x10x80xf64>
// CHECK-NEXT:    %1 = stablehlo.pad %0, %cst, low = [0, 0, 0], high = [0, 2, 4], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x80xf64>, tensor<f64>) -> tensor<1x12x84xf64>
// CHECK-NEXT:    %2 = stablehlo.pad %0, %cst, low = [0, 0, 1], high = [0, 2, 3], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x80xf64>, tensor<f64>) -> tensor<1x12x84xf64>
// CHECK-NEXT:    %3 = sdy.manual_computation(%1, %2) in_shardings=[<@mesh1, [{"z"}, {"y"}, {"x"}]>, <@mesh1, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh1, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "y", "x"} (%arg1: tensor<1x3x21xf64>, %arg2: tensor<1x3x21xf64>) {
// CHECK-NEXT:      %c = stablehlo.constant dense<17> : tensor<1x3x21xui32>
// CHECK-NEXT:      %c_0 = stablehlo.constant dense<3> : tensor<ui32>
// CHECK-NEXT:      %c_1 = stablehlo.constant dense<4> : tensor<ui32>
// CHECK-NEXT:      %5 = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:      %6 = stablehlo.divide %5, %c_1 : tensor<ui32>
// CHECK-NEXT:      %7 = stablehlo.compare  LE, %6, %c_0 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %8 = "stablehlo.if"(%7) ({
// CHECK-NEXT:        %9 = stablehlo.iota dim = 2 : tensor<1x3x21xui32>
// CHECK-NEXT:        %10 = stablehlo.compare  LT, %9, %c : (tensor<1x3x21xui32>, tensor<1x3x21xui32>) -> tensor<1x3x21xi1>
// CHECK-NEXT:        %11 = stablehlo.compare  LT, %6, %c_0 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<i1>) -> tensor<1x3x21xi1>
// CHECK-NEXT:        %13 = stablehlo.or %10, %12 : tensor<1x3x21xi1>
// CHECK-NEXT:        %14 = stablehlo.select %13, %arg1, %arg2 : tensor<1x3x21xi1>, tensor<1x3x21xf64>
// CHECK-NEXT:        stablehlo.return %14 : tensor<1x3x21xf64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        stablehlo.return %arg2 : tensor<1x3x21xf64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<1x3x21xf64>
// CHECK-NEXT:      sdy.return %8 : tensor<1x3x21xf64>
// CHECK-NEXT:    } : (tensor<1x12x84xf64>, tensor<1x12x84xf64>) -> tensor<1x12x84xf64>
// CHECK-NEXT:    %4 = stablehlo.slice %3 [0:1, 0:10, 0:81] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x12x84xf64>) -> tensor<1x10x81xf64>
// CHECK-NEXT:    return %4 : tensor<1x10x81xf64>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @main2(%arg0: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x10x81xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [11:12, 7:17, 0:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>) -> tensor<1x10x80xf64>
// CHECK-NEXT:    %1 = stablehlo.pad %0, %cst, low = [0, 0, 0], high = [0, 2, 4], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x80xf64>, tensor<f64>) -> tensor<1x12x84xf64>
// CHECK-NEXT:    %2 = stablehlo.pad %0, %cst, low = [0, 0, 1], high = [0, 2, 3], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x80xf64>, tensor<f64>) -> tensor<1x12x84xf64>
// CHECK-NEXT:    %3 = sdy.manual_computation(%1, %2) in_shardings=[<@mesh1, [{"z"}, {"y"}, {"x"}]>, <@mesh1, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh1, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "y", "x"} (%arg1: tensor<1x3x21xf64>, %arg2: tensor<1x3x21xf64>) {
// CHECK-NEXT:      %c = stablehlo.constant dense<1> : tensor<1x3x21xui32>
// CHECK-NEXT:      %c_0 = stablehlo.constant dense<4> : tensor<ui32>
// CHECK-NEXT:      %c_1 = stablehlo.constant dense<0> : tensor<ui32>
// CHECK-NEXT:      %5 = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:      %6 = stablehlo.divide %5, %c_0 : tensor<ui32>
// CHECK-NEXT:      %7 = stablehlo.compare  LE, %6, %c_1 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %8 = "stablehlo.if"(%7) ({
// CHECK-NEXT:        %9 = stablehlo.iota dim = 2 : tensor<1x3x21xui32>
// CHECK-NEXT:        %10 = stablehlo.compare  LT, %9, %c : (tensor<1x3x21xui32>, tensor<1x3x21xui32>) -> tensor<1x3x21xi1>
// CHECK-NEXT:        %11 = stablehlo.select %10, %arg1, %arg2 : tensor<1x3x21xi1>, tensor<1x3x21xf64>
// CHECK-NEXT:        stablehlo.return %11 : tensor<1x3x21xf64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        stablehlo.return %arg2 : tensor<1x3x21xf64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<1x3x21xf64>
// CHECK-NEXT:      sdy.return %8 : tensor<1x3x21xf64>
// CHECK-NEXT:    } : (tensor<1x12x84xf64>, tensor<1x12x84xf64>) -> tensor<1x12x84xf64>
// CHECK-NEXT:    %4 = stablehlo.slice %3 [0:1, 0:10, 0:81] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x12x84xf64>) -> tensor<1x10x81xf64>
// CHECK-NEXT:    return %4 : tensor<1x10x81xf64>
// CHECK-NEXT:  }
