// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{wrap_comm=1 wrap_to_pad_comm=0})" %s | FileCheck %s
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{wrap_to_pad_comm=1})" %s | FileCheck %s --check-prefix=PAD

sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=4]>
func.func @main(%arg0: tensor<1x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x8x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg0 [0:1, 0:8, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x24x96xf64>) -> tensor<1x8x80xf64>
    %1 = "enzymexla.wrap"(%0) <{dimension = 2 : i64, lhs = 8 : i64, rhs = 8 : i64}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x8x80xf64>) -> tensor<1x8x96xf64>
    return %1 : tensor<1x8x96xf64>
}

// CHECK: sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=4]>
// CHECK-NEXT: func.func @main(%arg0: tensor<1x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x8x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:     %0 = stablehlo.slice %arg0 [0:1, 0:8, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x24x96xf64>) -> tensor<1x8x80xf64>
// CHECK-NEXT:     %1 = sdy.manual_computation(%0) in_shardings=[<@mesh1, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh1, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "x", "y"} (%arg1: tensor<1x2x20xf64>) {
// CHECK-NEXT:       %c = stablehlo.constant dense<2> : tensor<ui32>
// CHECK-NEXT:       %c_0 = stablehlo.constant dense<1> : tensor<ui32>
// CHECK-NEXT:       %c_1 = stablehlo.constant dense<4> : tensor<ui32>
// CHECK-NEXT:       %c_2 = stablehlo.constant dense<0> : tensor<ui32>
// CHECK-NEXT:       %2 = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:       %3 = stablehlo.remainder %2, %c_1 : tensor<ui32>
// CHECK-NEXT:       %4 = stablehlo.compare  EQ, %3, %c_2 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:       %5 = stablehlo.add %2, %c_0 : tensor<ui32>
// CHECK-NEXT:       %6 = stablehlo.remainder %5, %c_1 : tensor<ui32>
// CHECK-NEXT:       %7 = stablehlo.compare  EQ, %6, %c_2 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:       %8 = stablehlo.not %4 : tensor<i1>
// CHECK-NEXT:       %9 = stablehlo.not %7 : tensor<i1>
// CHECK-NEXT:       %10 = stablehlo.and %8, %9 : tensor<i1>
// CHECK-NEXT:       %11 = "stablehlo.if"(%10) ({
// CHECK-NEXT:         %12 = stablehlo.remainder %2, %c_1 : tensor<ui32>
// CHECK-NEXT:         %13 = stablehlo.compare  LT, %3, %c : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:         %14 = "stablehlo.if"(%13) ({
// CHECK-NEXT:           %17 = stablehlo.slice %arg1 [0:1, 0:2, 12:20] : (tensor<1x2x20xf64>) -> tensor<1x2x8xf64>
// CHECK-NEXT:           stablehlo.return %17 : tensor<1x2x8xf64>
// CHECK-NEXT:         }, {
// CHECK-NEXT:           %17 = stablehlo.slice %arg1 [0:1, 0:2, 0:8] : (tensor<1x2x20xf64>) -> tensor<1x2x8xf64>
// CHECK-NEXT:           stablehlo.return %17 : tensor<1x2x8xf64>
// CHECK-NEXT:         }) : (tensor<i1>) -> tensor<1x2x8xf64>
// CHECK-NEXT{LITERAL}:         %15 = "stablehlo.collective_permute"(%14) <{channel_handle = #stablehlo.channel_handle<handle = 3, type = 0>, source_target_pairs = dense<[[0, 4], [1, 5], [2, 6], [3, 7], [12, 8], [13, 9], [14, 10], [15, 11]]> : tensor<8x2xi64>}> : (tensor<1x2x8xf64>) -> tensor<1x2x8xf64>
// CHECK-NEXT:         %16 = "stablehlo.if"(%13) ({
// CHECK-NEXT:           %17 = stablehlo.concatenate %15, %arg1, dim = 2 : (tensor<1x2x8xf64>, tensor<1x2x20xf64>) -> tensor<1x2x28xf64>
// CHECK-NEXT:           %18 = stablehlo.multiply %12, %c_1 : tensor<ui32>
// CHECK-NEXT:           %19 = stablehlo.dynamic_slice %17, %c_2, %c_2, %18, sizes = [1, 2, 24] : (tensor<1x2x28xf64>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<1x2x24xf64>
// CHECK-NEXT:           stablehlo.return %19 : tensor<1x2x24xf64>
// CHECK-NEXT:         }, {
// CHECK-NEXT:           %17 = stablehlo.concatenate %arg1, %15, dim = 2 : (tensor<1x2x20xf64>, tensor<1x2x8xf64>) -> tensor<1x2x28xf64>
// CHECK-NEXT:           %18 = stablehlo.multiply %12, %c_1 : tensor<ui32>
// CHECK-NEXT:           %19 = stablehlo.subtract %c_1, %18 : tensor<ui32>
// CHECK-NEXT:           %20 = stablehlo.dynamic_slice %17, %c_2, %c_2, %19, sizes = [1, 2, 24] : (tensor<1x2x28xf64>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<1x2x24xf64>
// CHECK-NEXT:           stablehlo.return %20 : tensor<1x2x24xf64>
// CHECK-NEXT:         }) : (tensor<i1>) -> tensor<1x2x24xf64>
// CHECK-NEXT:         stablehlo.return %16 : tensor<1x2x24xf64>
// CHECK-NEXT:       }, {
// CHECK-NEXT:         %12 = "stablehlo.if"(%4) ({
// CHECK-NEXT:           %15 = stablehlo.slice %arg1 [0:1, 0:2, 0:8] : (tensor<1x2x20xf64>) -> tensor<1x2x8xf64>
// CHECK-NEXT:           stablehlo.return %15 : tensor<1x2x8xf64>
// CHECK-NEXT:         }, {
// CHECK-NEXT:           %15 = stablehlo.slice %arg1 [0:1, 0:2, 12:20] : (tensor<1x2x20xf64>) -> tensor<1x2x8xf64>
// CHECK-NEXT:           stablehlo.return %15 : tensor<1x2x8xf64>
// CHECK-NEXT:         }) : (tensor<i1>) -> tensor<1x2x8xf64>
// CHECK-NEXT{LITERAL}:         %13 = "stablehlo.collective_permute"(%12) <{channel_handle = #stablehlo.channel_handle<handle = 4, type = 0>, source_target_pairs = dense<[[0, 12], [1, 13], [2, 14], [3, 15], [12, 0], [13, 1], [14, 2], [15, 3]]> : tensor<8x2xi64>}> : (tensor<1x2x8xf64>) -> tensor<1x2x8xf64>
// CHECK-NEXT:         %14 = "stablehlo.if"(%4) ({
// CHECK-NEXT:           %15 = stablehlo.slice %arg1 [0:1, 0:2, 0:16] : (tensor<1x2x20xf64>) -> tensor<1x2x16xf64>
// CHECK-NEXT:           %16 = stablehlo.concatenate %13, %15, dim = 2 : (tensor<1x2x8xf64>, tensor<1x2x16xf64>) -> tensor<1x2x24xf64>
// CHECK-NEXT:           stablehlo.return %16 : tensor<1x2x24xf64>
// CHECK-NEXT:         }, {
// CHECK-NEXT:           %15 = stablehlo.slice %arg1 [0:1, 0:2, 4:20] : (tensor<1x2x20xf64>) -> tensor<1x2x16xf64>
// CHECK-NEXT:           %16 = stablehlo.concatenate %15, %13, dim = 2 : (tensor<1x2x16xf64>, tensor<1x2x8xf64>) -> tensor<1x2x24xf64>
// CHECK-NEXT:           stablehlo.return %16 : tensor<1x2x24xf64>
// CHECK-NEXT:         }) : (tensor<i1>) -> tensor<1x2x24xf64>
// CHECK-NEXT:         stablehlo.return %14 : tensor<1x2x24xf64>
// CHECK-NEXT:       }) : (tensor<i1>) -> tensor<1x2x24xf64>
// CHECK-NEXT:       sdy.return %11 : tensor<1x2x24xf64>
// CHECK-NEXT:     } : (tensor<1x8x80xf64>) -> tensor<1x8x96xf64>
// CHECK-NEXT:     return %1 : tensor<1x8x96xf64>
// CHECK-NEXT: }

// PAD:  sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=4]>
// PAD-NEXT:  func.func @main(%arg0: tensor<1x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x8x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
// PAD-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// PAD-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 0:8, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x24x96xf64>) -> tensor<1x8x80xf64>
// PAD-NEXT:    %1 = stablehlo.slice %0 [0:1, 0:8, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x8x80xf64>) -> tensor<1x8x8xf64>
// PAD-NEXT:    %2 = stablehlo.slice %0 [0:1, 0:8, 72:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x8x80xf64>) -> tensor<1x8x8xf64>
// PAD-NEXT:    %3 = stablehlo.pad %1, %cst, low = [0, 0, 88], high = [0, 0, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x8x8xf64>, tensor<f64>) -> tensor<1x8x96xf64>
// PAD-NEXT:    %4 = stablehlo.pad %2, %cst, low = [0, 0, 0], high = [0, 0, 88], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x8x8xf64>, tensor<f64>) -> tensor<1x8x96xf64>
// PAD-NEXT:    %5 = stablehlo.pad %0, %cst, low = [0, 0, 8], high = [0, 0, 8], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x8x80xf64>, tensor<f64>) -> tensor<1x8x96xf64>
// PAD-NEXT:    %6 = stablehlo.add %4, %5 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : tensor<1x8x96xf64>
// PAD-NEXT:    %7 = stablehlo.add %6, %3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : tensor<1x8x96xf64>
// PAD-NEXT:    return %7 : tensor<1x8x96xf64>
// PAD-NEXT:  }


func.func @main2(%arg0: tensor<1x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x8x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg0 [0:1, 0:8, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x24x96xf64>) -> tensor<1x8x80xf64>
    %1 = "enzymexla.wrap"(%0) <{dimension = 2 : i64, lhs = 10 : i64, rhs = 6 : i64}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x8x80xf64>) -> tensor<1x8x96xf64>
    return %1 : tensor<1x8x96xf64>
}

// PAD:  func.func @main2(%arg0: tensor<1x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x8x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
// PAD-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// PAD-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 0:8, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x24x96xf64>) -> tensor<1x8x80xf64>
// PAD-NEXT:    %1 = stablehlo.slice %0 [0:1, 0:8, 0:6] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x8x80xf64>) -> tensor<1x8x6xf64>
// PAD-NEXT:    %2 = stablehlo.slice %0 [0:1, 0:8, 70:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x8x80xf64>) -> tensor<1x8x10xf64>
// PAD-NEXT:    %3 = stablehlo.pad %1, %cst, low = [0, 0, 90], high = [0, 0, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x8x6xf64>, tensor<f64>) -> tensor<1x8x96xf64>
// PAD-NEXT:    %4 = stablehlo.pad %2, %cst, low = [0, 0, 0], high = [0, 0, 86], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x8x10xf64>, tensor<f64>) -> tensor<1x8x96xf64>
// PAD-NEXT:    %5 = stablehlo.pad %0, %cst, low = [0, 0, 10], high = [0, 0, 6], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x8x80xf64>, tensor<f64>) -> tensor<1x8x96xf64>
// PAD-NEXT:    %6 = stablehlo.add %4, %5 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : tensor<1x8x96xf64>
// PAD-NEXT:    %7 = stablehlo.add %6, %3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : tensor<1x8x96xf64>
// PAD-NEXT:    return %7 : tensor<1x8x96xf64>
// PAD-NEXT:  }
