// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{wrap_comm=1 wrap_to_pad_comm=0})" %s | FileCheck %s
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{wrap_to_pad_comm=1})" %s | FileCheck %s --check-prefix=PAD

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

// PAD: sdy.mesh @mesh = <["x"=2, "y"=1, "z"=1]>
// PAD-NEXT:  func.func @wrap(%arg0: tensor<3x10x80xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> tensor<3x10x82xf64> {
// PAD-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// PAD-NEXT:    %0 = stablehlo.slice %arg0 [0:3, 0:10, 0:1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<3x10x80xf64>) -> tensor<3x10x1xf64>
// PAD-NEXT:    %1 = stablehlo.slice %arg0 [0:3, 0:10, 79:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<3x10x80xf64>) -> tensor<3x10x1xf64>
// PAD-NEXT:    %2 = stablehlo.pad %0, %cst, low = [0, 0, 81], high = [0, 0, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<3x10x1xf64>, tensor<f64>) -> tensor<3x10x82xf64>
// PAD-NEXT:    %3 = stablehlo.pad %1, %cst, low = [0, 0, 0], high = [0, 0, 81], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<3x10x1xf64>, tensor<f64>) -> tensor<3x10x82xf64>
// PAD-NEXT:    %4 = stablehlo.pad %arg0, %cst, low = [0, 0, 1], high = [0, 0, 1], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<3x10x80xf64>, tensor<f64>) -> tensor<3x10x82xf64>
// PAD-NEXT:    %5 = stablehlo.add %3, %4 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : tensor<3x10x82xf64>
// PAD-NEXT:    %6 = stablehlo.add %5, %2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : tensor<3x10x82xf64>
// PAD-NEXT:    stablehlo.return %6 : tensor<3x10x82xf64>
// PAD-NEXT:  }
