// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{extend_comm=1 extend_to_pad_comm=0 extend_to_pad_comm2=0})" %s | FileCheck %s
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{extend_comm=0 extend_to_pad_comm=1 extend_to_pad_comm2=0})" %s | FileCheck %s --check-prefix=PAD
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{extend_comm=0 extend_to_pad_comm=0 extend_to_pad_comm2=1},cse)" %s | FileCheck %s --check-prefix=SEL

sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=5]>
func.func @main(%arg0: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x10x82xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg0 [11:12, 7:17, 0:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>) -> tensor<1x10x80xf64>
    %1 = "enzymexla.extend"(%0) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 1 : i64}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x80xf64>) -> tensor<1x10x82xf64>
    return %1 : tensor<1x10x82xf64>
}

// CHECK: sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=5]>
// CHECK-NEXT: func.func @main(%arg0: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x10x82xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:     %0 = stablehlo.slice %arg0 [11:12, 7:17, 0:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>) -> tensor<1x10x80xf64>
// CHECK-NEXT:     %1 = sdy.manual_computation(%0) in_shardings=[<@mesh1, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh1, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "x", "y"} (%arg1: tensor<1x2x20xf64>) {
// CHECK-NEXT:       %c = stablehlo.constant dense<2> : tensor<ui32>
// CHECK-NEXT:       %c_0 = stablehlo.constant dense<1> : tensor<ui32>
// CHECK-NEXT:       %c_1 = stablehlo.constant dense<4> : tensor<ui32>
// CHECK-NEXT:       %c_2 = stablehlo.constant dense<0> : tensor<ui32>
// CHECK-NEXT:       %3 = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:       %4 = stablehlo.remainder %3, %c_1 : tensor<ui32>
// CHECK-NEXT:       %5 = stablehlo.compare  EQ, %4, %c_2 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:       %6 = stablehlo.add %3, %c_0 : tensor<ui32>
// CHECK-NEXT:       %7 = stablehlo.remainder %6, %c_1 : tensor<ui32>
// CHECK-NEXT:       %8 = stablehlo.compare  EQ, %7, %c_2 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:       %9 = stablehlo.not %5 : tensor<i1>
// CHECK-NEXT:       %10 = stablehlo.not %8 : tensor<i1>
// CHECK-NEXT:       %11 = stablehlo.and %9, %10 : tensor<i1>
// CHECK-NEXT:       %12 = "stablehlo.if"(%11) ({
// CHECK-NEXT:         %13 = stablehlo.remainder %3, %c_1 : tensor<ui32>
// CHECK-NEXT:         %14 = stablehlo.compare  LT, %4, %c : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:         %15 = "stablehlo.if"(%14) ({
// CHECK-NEXT:           %18 = stablehlo.slice %arg1 [0:1, 0:2, 18:20] : (tensor<1x2x20xf64>) -> tensor<1x2x2xf64>
// CHECK-NEXT:           stablehlo.return %18 : tensor<1x2x2xf64>
// CHECK-NEXT:         }, {
// CHECK-NEXT:           %18 = stablehlo.slice %arg1 [0:1, 0:2, 0:2] : (tensor<1x2x20xf64>) -> tensor<1x2x2xf64>
// CHECK-NEXT:           stablehlo.return %18 : tensor<1x2x2xf64>
// CHECK-NEXT:         }) : (tensor<i1>) -> tensor<1x2x2xf64>
// CHECK-NEXT{LITERAL}:         %16 = "stablehlo.collective_permute"(%15) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[0, 5], [1, 6], [2, 7], [3, 8], [4, 9], [15, 10], [16, 11], [17, 12], [18, 13], [19, 14]]> : tensor<10x2xi64>}> : (tensor<1x2x2xf64>) -> tensor<1x2x2xf64>
// CHECK-NEXT:         %17 = "stablehlo.if"(%14) ({
// CHECK-NEXT:           %18 = stablehlo.concatenate %16, %arg1, dim = 2 : (tensor<1x2x2xf64>, tensor<1x2x20xf64>) -> tensor<1x2x22xf64>
// CHECK-NEXT:           %19 = stablehlo.multiply %13, %c_0 : tensor<ui32>
// CHECK-NEXT:           %20 = stablehlo.dynamic_slice %18, %c_2, %c_2, %19, sizes = [1, 2, 21] : (tensor<1x2x22xf64>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<1x2x21xf64>
// CHECK-NEXT:           stablehlo.return %20 : tensor<1x2x21xf64>
// CHECK-NEXT:         }, {
// CHECK-NEXT:           %18 = stablehlo.concatenate %arg1, %16, dim = 2 : (tensor<1x2x20xf64>, tensor<1x2x2xf64>) -> tensor<1x2x22xf64>
// CHECK-NEXT:           %19 = stablehlo.multiply %13, %c_0 : tensor<ui32>
// CHECK-NEXT:           %20 = stablehlo.subtract %c_0, %19 : tensor<ui32>
// CHECK-NEXT:           %21 = stablehlo.dynamic_slice %18, %c_2, %c_2, %20, sizes = [1, 2, 21] : (tensor<1x2x22xf64>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<1x2x21xf64>
// CHECK-NEXT:           stablehlo.return %21 : tensor<1x2x21xf64>
// CHECK-NEXT:         }) : (tensor<i1>) -> tensor<1x2x21xf64>
// CHECK-NEXT:         stablehlo.return %17 : tensor<1x2x21xf64>
// CHECK-NEXT:       }, {
// CHECK-NEXT:         %13 = "stablehlo.if"(%5) ({
// CHECK-NEXT:           %14 = stablehlo.slice %arg1 [0:1, 0:2, 0:2] : (tensor<1x2x20xf64>) -> tensor<1x2x2xf64>
// CHECK-NEXT:           %15 = stablehlo.slice %arg1 [0:1, 0:2, 0:19] : (tensor<1x2x20xf64>) -> tensor<1x2x19xf64>
// CHECK-NEXT:           %16 = stablehlo.concatenate %14, %15, dim = 2 : (tensor<1x2x2xf64>, tensor<1x2x19xf64>) -> tensor<1x2x21xf64>
// CHECK-NEXT:           stablehlo.return %16 : tensor<1x2x21xf64>
// CHECK-NEXT:         }, {
// CHECK-NEXT:           %14 = stablehlo.slice %arg1 [0:1, 0:2, 18:20] : (tensor<1x2x20xf64>) -> tensor<1x2x2xf64>
// CHECK-NEXT:           %15 = stablehlo.slice %arg1 [0:1, 0:2, 1:20] : (tensor<1x2x20xf64>) -> tensor<1x2x19xf64>
// CHECK-NEXT:           %16 = stablehlo.concatenate %15, %14, dim = 2 : (tensor<1x2x19xf64>, tensor<1x2x2xf64>) -> tensor<1x2x21xf64>
// CHECK-NEXT:           stablehlo.return %16 : tensor<1x2x21xf64>
// CHECK-NEXT:         }) : (tensor<i1>) -> tensor<1x2x21xf64>
// CHECK-NEXT:         stablehlo.return %13 : tensor<1x2x21xf64>
// CHECK-NEXT:       }) : (tensor<i1>) -> tensor<1x2x21xf64>
// CHECK-NEXT:       sdy.return %12 : tensor<1x2x21xf64>
// CHECK-NEXT:     } : (tensor<1x10x80xf64>) -> tensor<1x10x84xf64>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:1, 0:10, 1:83] : (tensor<1x10x84xf64>) -> tensor<1x10x82xf64>
// CHECK-NEXT:     return %2 : tensor<1x10x82xf64>
// CHECK-NEXT: }

// PAD:  sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=5]>
// PAD-NEXT:  func.func @main(%arg0: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x10x82xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
// PAD-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// PAD-NEXT:    %0 = stablehlo.slice %arg0 [11:12, 7:17, 0:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>) -> tensor<1x10x80xf64>
// PAD-NEXT:    %1 = stablehlo.slice %0 [0:1, 0:10, 0:1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x80xf64>) -> tensor<1x10x1xf64>
// PAD-NEXT:    %2 = stablehlo.slice %0 [0:1, 0:10, 79:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x80xf64>) -> tensor<1x10x1xf64>
// PAD-NEXT:    %3 = stablehlo.pad %1, %cst, low = [0, 0, 0], high = [0, 0, 81], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x1xf64>, tensor<f64>) -> tensor<1x10x82xf64>
// PAD-NEXT:    %4 = stablehlo.pad %2, %cst, low = [0, 0, 81], high = [0, 0, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x1xf64>, tensor<f64>) -> tensor<1x10x82xf64>
// PAD-NEXT:    %5 = stablehlo.pad %0, %cst, low = [0, 0, 1], high = [0, 0, 1], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x80xf64>, tensor<f64>) -> tensor<1x10x82xf64>
// PAD-NEXT:    %6 = stablehlo.add %3, %5 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : tensor<1x10x82xf64>
// PAD-NEXT:    %7 = stablehlo.add %6, %4 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : tensor<1x10x82xf64>
// PAD-NEXT:    return %7 : tensor<1x10x82xf64>
// PAD-NEXT:  }

// SEL:  func.func @main(%arg0: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x10x82xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
// SEL-NEXT:    %c = stablehlo.constant dense<81> : tensor<1x10x82xi32>
// SEL-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<1x10x82xi32>
// SEL-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// SEL-NEXT:    %0 = stablehlo.slice %arg0 [11:12, 7:17, 0:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>) -> tensor<1x10x80xf64>
// SEL-NEXT:    %1 = stablehlo.pad %0, %cst, low = [0, 0, 1], high = [0, 0, 1], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x80xf64>, tensor<f64>) -> tensor<1x10x82xf64>
// SEL-NEXT:    %2 = stablehlo.iota dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : tensor<1x10x82xi32>
// SEL-NEXT:    %3 = stablehlo.pad %0, %cst, low = [0, 0, 0], high = [0, 0, 2], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x80xf64>, tensor<f64>) -> tensor<1x10x82xf64>
// SEL-NEXT:    %4 = stablehlo.compare  LT, %2, %c_0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x82xi32>, tensor<1x10x82xi32>) -> tensor<1x10x82xi1>
// SEL-NEXT:    %5 = stablehlo.select %4, %3, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : tensor<1x10x82xi1>, tensor<1x10x82xf64>
// SEL-NEXT:    %6 = stablehlo.pad %0, %cst, low = [0, 0, 2], high = [0, 0, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x80xf64>, tensor<f64>) -> tensor<1x10x82xf64>
// SEL-NEXT:    %7 = stablehlo.compare  LT, %2, %c {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x82xi32>, tensor<1x10x82xi32>) -> tensor<1x10x82xi1>
// SEL-NEXT:    %8 = stablehlo.select %7, %5, %6 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : tensor<1x10x82xi1>, tensor<1x10x82xf64>
// SEL-NEXT:    return %8 : tensor<1x10x82xf64>
// SEL-NEXT:  }

sdy.mesh @mesh2 = <["z"=1, "x"=2, "y"=5]>
func.func @main2(%arg0: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh2, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x10x82xf64> {sdy.sharding = #sdy.sharding<@mesh2, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg0 [11:12, 7:17, 0:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>) -> tensor<1x10x80xf64>
    %1 = "enzymexla.extend"(%0) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 1 : i64}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x80xf64>) -> tensor<1x10x82xf64>
    return %1 : tensor<1x10x82xf64>
}

// CHECK: sdy.mesh @mesh2 = <["z"=1, "x"=2, "y"=5]>
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

// PAD: sdy.mesh @mesh2 = <["z"=1, "x"=2, "y"=5]>
// PAD-NEXT: func.func @main2(%arg0: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh2, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x10x82xf64> {sdy.sharding = #sdy.sharding<@mesh2, [{"z"}, {"y"}, {"x"}]>}) {
// PAD-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// PAD-NEXT:     %0 = stablehlo.slice %arg0 [11:12, 7:17, 0:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>) -> tensor<1x10x80xf64>
// PAD-NEXT:     %1 = stablehlo.slice %0 [0:1, 0:10, 0:1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x80xf64>) -> tensor<1x10x1xf64>
// PAD-NEXT:     %2 = stablehlo.slice %0 [0:1, 0:10, 79:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x80xf64>) -> tensor<1x10x1xf64>
// PAD-NEXT:     %3 = stablehlo.pad %1, %cst, low = [0, 0, 0], high = [0, 0, 81], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x1xf64>, tensor<f64>) -> tensor<1x10x82xf64>
// PAD-NEXT:     %4 = stablehlo.pad %2, %cst, low = [0, 0, 81], high = [0, 0, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x1xf64>, tensor<f64>) -> tensor<1x10x82xf64>
// PAD-NEXT:     %5 = stablehlo.pad %0, %cst, low = [0, 0, 1], high = [0, 0, 1], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x80xf64>, tensor<f64>) -> tensor<1x10x82xf64>
// PAD-NEXT:     %6 = stablehlo.add %3, %5 {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : tensor<1x10x82xf64>
// PAD-NEXT:     %7 = stablehlo.add %6, %4 {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : tensor<1x10x82xf64>
// PAD-NEXT:     return %7 : tensor<1x10x82xf64>
// PAD-NEXT: }

// SEL:  func.func @main2(%arg0: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh2, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x10x82xf64> {sdy.sharding = #sdy.sharding<@mesh2, [{"z"}, {"y"}, {"x"}]>}) {
// SEL-NEXT:    %c = stablehlo.constant dense<81> : tensor<1x10x82xi32>
// SEL-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<1x10x82xi32>
// SEL-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// SEL-NEXT:    %0 = stablehlo.slice %arg0 [11:12, 7:17, 0:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>) -> tensor<1x10x80xf64>
// SEL-NEXT:    %1 = stablehlo.pad %0, %cst, low = [0, 0, 1], high = [0, 0, 1], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x80xf64>, tensor<f64>) -> tensor<1x10x82xf64>
// SEL-NEXT:    %2 = stablehlo.iota dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : tensor<1x10x82xi32>
// SEL-NEXT:    %3 = stablehlo.pad %0, %cst, low = [0, 0, 0], high = [0, 0, 2], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x80xf64>, tensor<f64>) -> tensor<1x10x82xf64>
// SEL-NEXT:    %4 = stablehlo.compare  LT, %2, %c_0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x82xi32>, tensor<1x10x82xi32>) -> tensor<1x10x82xi1>
// SEL-NEXT:    %5 = stablehlo.select %4, %3, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : tensor<1x10x82xi1>, tensor<1x10x82xf64>
// SEL-NEXT:    %6 = stablehlo.pad %0, %cst, low = [0, 0, 2], high = [0, 0, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x80xf64>, tensor<f64>) -> tensor<1x10x82xf64>
// SEL-NEXT:    %7 = stablehlo.compare  LT, %2, %c {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x82xi32>, tensor<1x10x82xi32>) -> tensor<1x10x82xi1>
// SEL-NEXT:    %8 = stablehlo.select %7, %5, %6 {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : tensor<1x10x82xi1>, tensor<1x10x82xf64>
// SEL-NEXT:    return %8 : tensor<1x10x82xf64>
// SEL-NEXT:  }

