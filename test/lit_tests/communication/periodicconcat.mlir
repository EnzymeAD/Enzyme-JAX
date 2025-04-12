// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{periodic_concat=1 concat_to_pad_comm=0})" %s | FileCheck %s
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{periodic_concat=0 concat_to_pad_comm=1})" %s | FileCheck %s --check-prefix=PAD

sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=4]>
func.func @main1(%arg0: tensor<528x1024x2048xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<512x1022x2046xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg0 [8:520, 1:1023, 2034:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x6xf64>
    %1 = stablehlo.slice %arg0 [8:520, 1:1023, 8:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x2032xf64>
    %2 = stablehlo.slice %arg0 [8:520, 1:1023, 8:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x8xf64>
    %3 = stablehlo.concatenate %0, %1, %2, dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<512x1022x6xf64>, tensor<512x1022x2032xf64>, tensor<512x1022x8xf64>) -> tensor<512x1022x2046xf64>
    return %3 : tensor<512x1022x2046xf64>
}

// CHECK: sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=4]>
// CHECK-NEXT:  func.func @main1(%arg0: tensor<528x1024x2048xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<512x1022x2046xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [8:520, 1:1023, 8:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x2032xf64>
// CHECK-NEXT:    %1 = sdy.manual_computation(%0, %0) in_shardings=[<@mesh1, [{"z"}, {"y"}, {"x"}]>, <@mesh1, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh1, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "x", "y"} (%arg1: tensor<512x255x508xf64>, %arg2: tensor<512x255x508xf64>) {
// CHECK-NEXT:      %c = stablehlo.constant dense<2> : tensor<ui32>
// CHECK-NEXT:      %c_0 = stablehlo.constant dense<1> : tensor<ui32>
// CHECK-NEXT:      %c_1 = stablehlo.constant dense<4> : tensor<ui32>
// CHECK-NEXT:      %c_2 = stablehlo.constant dense<0> : tensor<ui32>
// CHECK-NEXT:      %3 = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:      %4 = stablehlo.remainder %3, %c_1 : tensor<ui32>
// CHECK-NEXT:      %5 = stablehlo.compare  EQ, %4, %c_2 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %6 = stablehlo.add %3, %c_0 : tensor<ui32>
// CHECK-NEXT:      %7 = stablehlo.remainder %6, %c_1 : tensor<ui32>
// CHECK-NEXT:      %8 = stablehlo.compare  EQ, %7, %c_2 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %9 = stablehlo.not %5 : tensor<i1>
// CHECK-NEXT:      %10 = stablehlo.not %8 : tensor<i1>
// CHECK-NEXT:      %11 = stablehlo.and %9, %10 : tensor<i1>
// CHECK-NEXT:      %12 = "stablehlo.if"(%11) ({
// CHECK-NEXT:        %13 = stablehlo.remainder %3, %c_1 : tensor<ui32>
// CHECK-NEXT:        %14 = stablehlo.compare  LT, %4, %c : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %15 = "stablehlo.if"(%14) ({
// CHECK-NEXT:          %18 = stablehlo.slice %arg1 [0:512, 0:255, 500:508] : (tensor<512x255x508xf64>) -> tensor<512x255x8xf64>
// CHECK-NEXT:          stablehlo.return %18 : tensor<512x255x8xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %18 = stablehlo.slice %arg1 [0:512, 0:255, 0:8] : (tensor<512x255x508xf64>) -> tensor<512x255x8xf64>
// CHECK-NEXT:          stablehlo.return %18 : tensor<512x255x8xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<512x255x8xf64>
// CHECK-NEXT{LITERAL}:        %16 = "stablehlo.collective_permute"(%15) <{channel_handle = #stablehlo.channel_handle<handle = 5, type = 0>, source_target_pairs = dense<[[0, 4], [1, 5], [2, 6], [3, 7], [12, 8], [13, 9], [14, 10], [15, 11]]> : tensor<8x2xi64>}> : (tensor<512x255x8xf64>) -> tensor<512x255x8xf64>
// CHECK-NEXT:        %17 = "stablehlo.if"(%14) ({
// CHECK-NEXT:          %18 = stablehlo.concatenate %16, %arg2, dim = 2 : (tensor<512x255x8xf64>, tensor<512x255x508xf64>) -> tensor<512x255x516xf64>
// CHECK-NEXT:          %19 = stablehlo.multiply %13, %c_1 : tensor<ui32>
// CHECK-NEXT:          %20 = stablehlo.dynamic_slice %18, %c_2, %c_2, %19, sizes = [512, 255, 512] : (tensor<512x255x516xf64>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<512x255x512xf64>
// CHECK-NEXT:          stablehlo.return %20 : tensor<512x255x512xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %18 = stablehlo.concatenate %arg2, %16, dim = 2 : (tensor<512x255x508xf64>, tensor<512x255x8xf64>) -> tensor<512x255x516xf64>
// CHECK-NEXT:          %19 = stablehlo.multiply %13, %c_1 : tensor<ui32>
// CHECK-NEXT:          %20 = stablehlo.subtract %c_1, %19 : tensor<ui32>
// CHECK-NEXT:          %21 = stablehlo.dynamic_slice %18, %c_2, %c_2, %20, sizes = [512, 255, 512] : (tensor<512x255x516xf64>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<512x255x512xf64>
// CHECK-NEXT:          stablehlo.return %21 : tensor<512x255x512xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<512x255x512xf64>
// CHECK-NEXT:        stablehlo.return %17 : tensor<512x255x512xf64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        %13 = "stablehlo.if"(%5) ({
// CHECK-NEXT:          %16 = stablehlo.slice %arg1 [0:512, 0:255, 0:8] : (tensor<512x255x508xf64>) -> tensor<512x255x8xf64>
// CHECK-NEXT:          stablehlo.return %16 : tensor<512x255x8xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %16 = stablehlo.slice %arg1 [0:512, 0:255, 500:508] : (tensor<512x255x508xf64>) -> tensor<512x255x8xf64>
// CHECK-NEXT:          stablehlo.return %16 : tensor<512x255x8xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<512x255x8xf64>
// CHECK-NEXT{LITERAL}:        %14 = "stablehlo.collective_permute"(%13) <{channel_handle = #stablehlo.channel_handle<handle = 6, type = 0>, source_target_pairs = dense<[[0, 12], [1, 13], [2, 14], [3, 15], [12, 0], [13, 1], [14, 2], [15, 3]]> : tensor<8x2xi64>}> : (tensor<512x255x8xf64>) -> tensor<512x255x8xf64>
// CHECK-NEXT:        %15 = "stablehlo.if"(%5) ({
// CHECK-NEXT:          %16 = stablehlo.slice %arg2 [0:512, 0:255, 0:504] : (tensor<512x255x508xf64>) -> tensor<512x255x504xf64>
// CHECK-NEXT:          %17 = stablehlo.concatenate %14, %16, dim = 2 : (tensor<512x255x8xf64>, tensor<512x255x504xf64>) -> tensor<512x255x512xf64>
// CHECK-NEXT:          stablehlo.return %17 : tensor<512x255x512xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %16 = stablehlo.slice %arg2 [0:512, 0:255, 4:508] : (tensor<512x255x508xf64>) -> tensor<512x255x504xf64>
// CHECK-NEXT:          %17 = stablehlo.concatenate %16, %14, dim = 2 : (tensor<512x255x504xf64>, tensor<512x255x8xf64>) -> tensor<512x255x512xf64>
// CHECK-NEXT:          stablehlo.return %17 : tensor<512x255x512xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<512x255x512xf64>
// CHECK-NEXT:        stablehlo.return %15 : tensor<512x255x512xf64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<512x255x512xf64>
// CHECK-NEXT:      sdy.return %12 : tensor<512x255x512xf64>
// CHECK-NEXT:    } : (tensor<512x1022x2032xf64>, tensor<512x1022x2032xf64>) -> tensor<512x1022x2048xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %1 [0:512, 0:1022, 2:2048] : (tensor<512x1022x2048xf64>) -> tensor<512x1022x2046xf64>
// CHECK-NEXT:    return %2 : tensor<512x1022x2046xf64>
// CHECK-NEXT:  }

// PAD: sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=4]>
// PAD-NEXT: func.func @main1(%arg0: tensor<528x1024x2048xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<512x1022x2046xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
// PAD-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// PAD-NEXT:     %0 = stablehlo.slice %arg0 [8:520, 1:1023, 2034:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x6xf64>
// PAD-NEXT:     %1 = stablehlo.slice %arg0 [8:520, 1:1023, 8:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x2032xf64>
// PAD-NEXT:     %2 = stablehlo.slice %arg0 [8:520, 1:1023, 8:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x8xf64>
// PAD-NEXT:     %3 = stablehlo.pad %0, %cst, low = [0, 0, 0], high = [0, 0, 2040], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<512x1022x6xf64>, tensor<f64>) -> tensor<512x1022x2046xf64>
// PAD-NEXT:     %4 = stablehlo.pad %1, %cst, low = [0, 0, 6], high = [0, 0, 8], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<512x1022x2032xf64>, tensor<f64>) -> tensor<512x1022x2046xf64>
// PAD-NEXT:     %5 = stablehlo.pad %2, %cst, low = [0, 0, 2038], high = [0, 0, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<512x1022x8xf64>, tensor<f64>) -> tensor<512x1022x2046xf64>
// PAD-NEXT:     %6 = stablehlo.add %3, %4 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : tensor<512x1022x2046xf64>
// PAD-NEXT:     %7 = stablehlo.add %6, %5 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : tensor<512x1022x2046xf64>
// PAD-NEXT:     return %7 : tensor<512x1022x2046xf64>
// PAD-NEXT: }

sdy.mesh @mesh2 = <["z"=1, "x"=8, "y"=4]>
func.func @main2(%arg0: tensor<528x1024x2048xf64> {sdy.sharding = #sdy.sharding<@mesh2, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<512x1022x2057xf64> {sdy.sharding = #sdy.sharding<@mesh2, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg0 [8:520, 1:1023, 2028:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x12xf64>
    %1 = stablehlo.slice %arg0 [8:520, 1:1023, 8:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x2032xf64>
    %2 = stablehlo.slice %arg0 [8:520, 1:1023, 8:21] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x13xf64>
    %3 = stablehlo.concatenate %0, %1, %2, dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<512x1022x12xf64>, tensor<512x1022x2032xf64>, tensor<512x1022x13xf64>) -> tensor<512x1022x2057xf64>
    return %3 : tensor<512x1022x2057xf64>
}

// CHECK-NEXT:  sdy.mesh @mesh2 = <["z"=1, "x"=8, "y"=4]>
// CHECK-NEXT:  func.func @main2(%arg0: tensor<528x1024x2048xf64> {sdy.sharding = #sdy.sharding<@mesh2, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<512x1022x2057xf64> {sdy.sharding = #sdy.sharding<@mesh2, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [8:520, 1:1023, 8:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x2032xf64>
// CHECK-NEXT:    %1 = sdy.manual_computation(%0, %0) in_shardings=[<@mesh2, [{"z"}, {"y"}, {"x"}]>, <@mesh2, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh2, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "x", "y"} (%arg1: tensor<512x255x254xf64>, %arg2: tensor<512x255x254xf64>) {
// CHECK-NEXT:      %c = stablehlo.constant dense<12> : tensor<ui32>
// CHECK-NEXT:      %c_0 = stablehlo.constant dense<4> : tensor<ui32>
// CHECK-NEXT:      %c_1 = stablehlo.constant dense<1> : tensor<ui32>
// CHECK-NEXT:      %c_2 = stablehlo.constant dense<8> : tensor<ui32>
// CHECK-NEXT:      %c_3 = stablehlo.constant dense<0> : tensor<ui32>
// CHECK-NEXT:      %3 = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:      %4 = stablehlo.remainder %3, %c_2 : tensor<ui32>
// CHECK-NEXT:      %5 = stablehlo.compare  EQ, %4, %c_3 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %6 = stablehlo.add %3, %c_1 : tensor<ui32>
// CHECK-NEXT:      %7 = stablehlo.remainder %6, %c_2 : tensor<ui32>
// CHECK-NEXT:      %8 = stablehlo.compare  EQ, %7, %c_3 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %9 = stablehlo.not %5 : tensor<i1>
// CHECK-NEXT:      %10 = stablehlo.not %8 : tensor<i1>
// CHECK-NEXT:      %11 = stablehlo.and %9, %10 : tensor<i1>
// CHECK-NEXT:      %12 = "stablehlo.if"(%11) ({
// CHECK-NEXT:        %13 = stablehlo.remainder %3, %c_2 : tensor<ui32>
// CHECK-NEXT:        %14 = stablehlo.compare  LT, %4, %c_0 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %15 = "stablehlo.if"(%14) ({
// CHECK-NEXT:          %18 = stablehlo.slice %arg1 [0:512, 0:255, 238:254] : (tensor<512x255x254xf64>) -> tensor<512x255x16xf64>
// CHECK-NEXT:          stablehlo.return %18 : tensor<512x255x16xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %18 = stablehlo.slice %arg1 [0:512, 0:255, 0:16] : (tensor<512x255x254xf64>) -> tensor<512x255x16xf64>
// CHECK-NEXT:          stablehlo.return %18 : tensor<512x255x16xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<512x255x16xf64>
// CHECK-NEXT{LITERAL}:        %16 = "stablehlo.collective_permute"(%15) <{channel_handle = #stablehlo.channel_handle<handle = 3, type = 0>, source_target_pairs = dense<[[0, 4], [1, 5], [2, 6], [3, 7], [4, 8], [5, 9], [6, 10], [7, 11], [8, 12], [9, 13], [10, 14], [11, 15], [20, 16], [21, 17], [22, 18], [23, 19], [24, 20], [25, 21], [26, 22], [27, 23], [28, 24], [29, 25], [30, 26], [31, 27]]> : tensor<24x2xi64>}> : (tensor<512x255x16xf64>) -> tensor<512x255x16xf64>
// CHECK-NEXT:        %17 = "stablehlo.if"(%14) ({
// CHECK-NEXT:          %18 = stablehlo.concatenate %16, %arg2, dim = 2 : (tensor<512x255x16xf64>, tensor<512x255x254xf64>) -> tensor<512x255x270xf64>
// CHECK-NEXT:          %19 = stablehlo.multiply %13, %c_0 : tensor<ui32>
// CHECK-NEXT:          %20 = stablehlo.dynamic_slice %18, %c_3, %c_3, %19, sizes = [512, 255, 258] : (tensor<512x255x270xf64>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<512x255x258xf64>
// CHECK-NEXT:          stablehlo.return %20 : tensor<512x255x258xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %18 = stablehlo.concatenate %arg2, %16, dim = 2 : (tensor<512x255x254xf64>, tensor<512x255x16xf64>) -> tensor<512x255x270xf64>
// CHECK-NEXT:          %19 = stablehlo.multiply %13, %c_0 : tensor<ui32>
// CHECK-NEXT:          %20 = stablehlo.subtract %c, %19 : tensor<ui32>
// CHECK-NEXT:          %21 = stablehlo.dynamic_slice %18, %c_3, %c_3, %20, sizes = [512, 255, 258] : (tensor<512x255x270xf64>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<512x255x258xf64>
// CHECK-NEXT:          stablehlo.return %21 : tensor<512x255x258xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<512x255x258xf64>
// CHECK-NEXT:        stablehlo.return %17 : tensor<512x255x258xf64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        %13 = "stablehlo.if"(%5) ({
// CHECK-NEXT:          %16 = stablehlo.slice %arg1 [0:512, 0:255, 0:16] : (tensor<512x255x254xf64>) -> tensor<512x255x16xf64>
// CHECK-NEXT:          stablehlo.return %16 : tensor<512x255x16xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %16 = stablehlo.slice %arg1 [0:512, 0:255, 238:254] : (tensor<512x255x254xf64>) -> tensor<512x255x16xf64>
// CHECK-NEXT:          stablehlo.return %16 : tensor<512x255x16xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<512x255x16xf64>
// CHECK-NEXT{LITERAL}:        %14 = "stablehlo.collective_permute"(%13) <{channel_handle = #stablehlo.channel_handle<handle = 4, type = 0>, source_target_pairs = dense<[[0, 28], [1, 29], [2, 30], [3, 31], [28, 0], [29, 1], [30, 2], [31, 3]]> : tensor<8x2xi64>}> : (tensor<512x255x16xf64>) -> tensor<512x255x16xf64>
// CHECK-NEXT:        %15 = "stablehlo.if"(%5) ({
// CHECK-NEXT:          %16 = stablehlo.slice %arg2 [0:512, 0:255, 0:242] : (tensor<512x255x254xf64>) -> tensor<512x255x242xf64>
// CHECK-NEXT:          %17 = stablehlo.concatenate %14, %16, dim = 2 : (tensor<512x255x16xf64>, tensor<512x255x242xf64>) -> tensor<512x255x258xf64>
// CHECK-NEXT:          stablehlo.return %17 : tensor<512x255x258xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %16 = stablehlo.slice %arg2 [0:512, 0:255, 12:254] : (tensor<512x255x254xf64>) -> tensor<512x255x242xf64>
// CHECK-NEXT:          %17 = stablehlo.concatenate %16, %14, dim = 2 : (tensor<512x255x242xf64>, tensor<512x255x16xf64>) -> tensor<512x255x258xf64>
// CHECK-NEXT:          stablehlo.return %17 : tensor<512x255x258xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<512x255x258xf64>
// CHECK-NEXT:        stablehlo.return %15 : tensor<512x255x258xf64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<512x255x258xf64>
// CHECK-NEXT:      sdy.return %12 : tensor<512x255x258xf64>
// CHECK-NEXT:    } : (tensor<512x1022x2032xf64>, tensor<512x1022x2032xf64>) -> tensor<512x1022x2064xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %1 [0:512, 0:1022, 4:2061] : (tensor<512x1022x2064xf64>) -> tensor<512x1022x2057xf64>
// CHECK-NEXT:    return %2 : tensor<512x1022x2057xf64>
// CHECK-NEXT:  }

// PAD: sdy.mesh @mesh2 = <["z"=1, "x"=8, "y"=4]>
// PAD-NEXT: func.func @main2(%arg0: tensor<528x1024x2048xf64> {sdy.sharding = #sdy.sharding<@mesh2, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<512x1022x2057xf64> {sdy.sharding = #sdy.sharding<@mesh2, [{"z"}, {"y"}, {"x"}]>}) {
// PAD-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// PAD-NEXT:     %0 = stablehlo.slice %arg0 [8:520, 1:1023, 2028:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x12xf64>
// PAD-NEXT:     %1 = stablehlo.slice %arg0 [8:520, 1:1023, 8:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x2032xf64>
// PAD-NEXT:     %2 = stablehlo.slice %arg0 [8:520, 1:1023, 8:21] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x13xf64>
// PAD-NEXT:     %3 = stablehlo.pad %0, %cst, low = [0, 0, 0], high = [0, 0, 2045], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<512x1022x12xf64>, tensor<f64>) -> tensor<512x1022x2057xf64>
// PAD-NEXT:     %4 = stablehlo.pad %1, %cst, low = [0, 0, 12], high = [0, 0, 13], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<512x1022x2032xf64>, tensor<f64>) -> tensor<512x1022x2057xf64>
// PAD-NEXT:     %5 = stablehlo.pad %2, %cst, low = [0, 0, 2044], high = [0, 0, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<512x1022x13xf64>, tensor<f64>) -> tensor<512x1022x2057xf64>
// PAD-NEXT:     %6 = stablehlo.add %3, %4 {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : tensor<512x1022x2057xf64>
// PAD-NEXT:     %7 = stablehlo.add %6, %5 {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : tensor<512x1022x2057xf64>
// PAD-NEXT:     return %7 : tensor<512x1022x2057xf64>
// PAD-NEXT: }

sdy.mesh @mesh3 = <["z"=1, "x"=8, "y"=4]>
func.func @main3(%arg0: tensor<528x1024x2048xf64> {sdy.sharding = #sdy.sharding<@mesh3, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<512x1022x2032xf64> {sdy.sharding = #sdy.sharding<@mesh3, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<512x1022x2057xf64> {sdy.sharding = #sdy.sharding<@mesh3, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg0 [8:520, 1:1023, 2028:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh3, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x12xf64>
    %2 = stablehlo.slice %arg0 [8:520, 1:1023, 8:21] {sdy.sharding = #sdy.sharding_per_value<[<@mesh3, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x13xf64>
    %3 = stablehlo.concatenate %0, %arg1, %2, dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh3, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<512x1022x12xf64>, tensor<512x1022x2032xf64>, tensor<512x1022x13xf64>) -> tensor<512x1022x2057xf64>
    return %3 : tensor<512x1022x2057xf64>
}

// CHECK: sdy.mesh @mesh3 = <["z"=1, "x"=8, "y"=4]>
// CHECK-NEXT:  func.func @main3(%arg0: tensor<528x1024x2048xf64> {sdy.sharding = #sdy.sharding<@mesh3, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<512x1022x2032xf64> {sdy.sharding = #sdy.sharding<@mesh3, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<512x1022x2057xf64> {sdy.sharding = #sdy.sharding<@mesh3, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [8:520, 1:1023, 8:2040] : (tensor<528x1024x2048xf64>) -> tensor<512x1022x2032xf64>
// CHECK-NEXT:    %1 = sdy.manual_computation(%0, %arg1) in_shardings=[<@mesh3, [{"z"}, {"y"}, {"x"}]>, <@mesh3, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh3, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "x", "y"} (%arg2: tensor<512x255x254xf64>, %arg3: tensor<512x255x254xf64>) {
// CHECK-NEXT:      %c = stablehlo.constant dense<12> : tensor<ui32>
// CHECK-NEXT:      %c_0 = stablehlo.constant dense<4> : tensor<ui32>
// CHECK-NEXT:      %c_1 = stablehlo.constant dense<1> : tensor<ui32>
// CHECK-NEXT:      %c_2 = stablehlo.constant dense<8> : tensor<ui32>
// CHECK-NEXT:      %c_3 = stablehlo.constant dense<0> : tensor<ui32>
// CHECK-NEXT:      %3 = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:      %4 = stablehlo.remainder %3, %c_2 : tensor<ui32>
// CHECK-NEXT:      %5 = stablehlo.compare  EQ, %4, %c_3 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %6 = stablehlo.add %3, %c_1 : tensor<ui32>
// CHECK-NEXT:      %7 = stablehlo.remainder %6, %c_2 : tensor<ui32>
// CHECK-NEXT:      %8 = stablehlo.compare  EQ, %7, %c_3 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %9 = stablehlo.not %5 : tensor<i1>
// CHECK-NEXT:      %10 = stablehlo.not %8 : tensor<i1>
// CHECK-NEXT:      %11 = stablehlo.and %9, %10 : tensor<i1>
// CHECK-NEXT:      %12 = "stablehlo.if"(%11) ({
// CHECK-NEXT:        %13 = stablehlo.remainder %3, %c_2 : tensor<ui32>
// CHECK-NEXT:        %14 = stablehlo.compare  LT, %4, %c_0 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %15 = "stablehlo.if"(%14) ({
// CHECK-NEXT:          %18 = stablehlo.slice %arg2 [0:512, 0:255, 238:254] : (tensor<512x255x254xf64>) -> tensor<512x255x16xf64>
// CHECK-NEXT:          stablehlo.return %18 : tensor<512x255x16xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %18 = stablehlo.slice %arg2 [0:512, 0:255, 0:16] : (tensor<512x255x254xf64>) -> tensor<512x255x16xf64>
// CHECK-NEXT:          stablehlo.return %18 : tensor<512x255x16xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<512x255x16xf64>
// CHECK-NEXT{LITERAL}:        %16 = "stablehlo.collective_permute"(%15) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[0, 4], [1, 5], [2, 6], [3, 7], [4, 8], [5, 9], [6, 10], [7, 11], [8, 12], [9, 13], [10, 14], [11, 15], [20, 16], [21, 17], [22, 18], [23, 19], [24, 20], [25, 21], [26, 22], [27, 23], [28, 24], [29, 25], [30, 26], [31, 27]]> : tensor<24x2xi64>}> : (tensor<512x255x16xf64>) -> tensor<512x255x16xf64>
// CHECK-NEXT:        %17 = "stablehlo.if"(%14) ({
// CHECK-NEXT:          %18 = stablehlo.concatenate %16, %arg3, dim = 2 : (tensor<512x255x16xf64>, tensor<512x255x254xf64>) -> tensor<512x255x270xf64>
// CHECK-NEXT:          %19 = stablehlo.multiply %13, %c_0 : tensor<ui32>
// CHECK-NEXT:          %20 = stablehlo.dynamic_slice %18, %c_3, %c_3, %19, sizes = [512, 255, 258] : (tensor<512x255x270xf64>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<512x255x258xf64>
// CHECK-NEXT:          stablehlo.return %20 : tensor<512x255x258xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %18 = stablehlo.concatenate %arg3, %16, dim = 2 : (tensor<512x255x254xf64>, tensor<512x255x16xf64>) -> tensor<512x255x270xf64>
// CHECK-NEXT:          %19 = stablehlo.multiply %13, %c_0 : tensor<ui32>
// CHECK-NEXT:          %20 = stablehlo.subtract %c, %19 : tensor<ui32>
// CHECK-NEXT:          %21 = stablehlo.dynamic_slice %18, %c_3, %c_3, %20, sizes = [512, 255, 258] : (tensor<512x255x270xf64>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<512x255x258xf64>
// CHECK-NEXT:          stablehlo.return %21 : tensor<512x255x258xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<512x255x258xf64>
// CHECK-NEXT:        stablehlo.return %17 : tensor<512x255x258xf64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        %13 = "stablehlo.if"(%5) ({
// CHECK-NEXT:          %16 = stablehlo.slice %arg2 [0:512, 0:255, 0:16] : (tensor<512x255x254xf64>) -> tensor<512x255x16xf64>
// CHECK-NEXT:          stablehlo.return %16 : tensor<512x255x16xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %16 = stablehlo.slice %arg2 [0:512, 0:255, 238:254] : (tensor<512x255x254xf64>) -> tensor<512x255x16xf64>
// CHECK-NEXT:          stablehlo.return %16 : tensor<512x255x16xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<512x255x16xf64>
// CHECK-NEXT{LITERAL}:        %14 = "stablehlo.collective_permute"(%13) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 0>, source_target_pairs = dense<[[0, 28], [1, 29], [2, 30], [3, 31], [28, 0], [29, 1], [30, 2], [31, 3]]> : tensor<8x2xi64>}> : (tensor<512x255x16xf64>) -> tensor<512x255x16xf64>
// CHECK-NEXT:        %15 = "stablehlo.if"(%5) ({
// CHECK-NEXT:          %16 = stablehlo.slice %arg3 [0:512, 0:255, 0:242] : (tensor<512x255x254xf64>) -> tensor<512x255x242xf64>
// CHECK-NEXT:          %17 = stablehlo.concatenate %14, %16, dim = 2 : (tensor<512x255x16xf64>, tensor<512x255x242xf64>) -> tensor<512x255x258xf64>
// CHECK-NEXT:          stablehlo.return %17 : tensor<512x255x258xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %16 = stablehlo.slice %arg3 [0:512, 0:255, 12:254] : (tensor<512x255x254xf64>) -> tensor<512x255x242xf64>
// CHECK-NEXT:          %17 = stablehlo.concatenate %16, %14, dim = 2 : (tensor<512x255x242xf64>, tensor<512x255x16xf64>) -> tensor<512x255x258xf64>
// CHECK-NEXT:          stablehlo.return %17 : tensor<512x255x258xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<512x255x258xf64>
// CHECK-NEXT:        stablehlo.return %15 : tensor<512x255x258xf64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<512x255x258xf64>
// CHECK-NEXT:      sdy.return %12 : tensor<512x255x258xf64>
// CHECK-NEXT:    } : (tensor<512x1022x2032xf64>, tensor<512x1022x2032xf64>) -> tensor<512x1022x2064xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %1 [0:512, 0:1022, 4:2061] : (tensor<512x1022x2064xf64>) -> tensor<512x1022x2057xf64>
// CHECK-NEXT:    return %2 : tensor<512x1022x2057xf64>
// CHECK-NEXT:  }

// PAD: sdy.mesh @mesh3 = <["z"=1, "x"=8, "y"=4]>
// PAD-NEXT: func.func @main3(%arg0: tensor<528x1024x2048xf64> {sdy.sharding = #sdy.sharding<@mesh3, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<512x1022x2032xf64> {sdy.sharding = #sdy.sharding<@mesh3, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<512x1022x2057xf64> {sdy.sharding = #sdy.sharding<@mesh3, [{"z"}, {"y"}, {"x"}]>}) {
// PAD-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// PAD-NEXT:     %0 = stablehlo.slice %arg0 [8:520, 1:1023, 2028:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh3, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x12xf64>
// PAD-NEXT:     %1 = stablehlo.slice %arg0 [8:520, 1:1023, 8:21] {sdy.sharding = #sdy.sharding_per_value<[<@mesh3, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x13xf64>
// PAD-NEXT:     %2 = stablehlo.pad %0, %cst, low = [0, 0, 0], high = [0, 0, 2045], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh3, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<512x1022x12xf64>, tensor<f64>) -> tensor<512x1022x2057xf64>
// PAD-NEXT:     %3 = stablehlo.pad %arg1, %cst, low = [0, 0, 12], high = [0, 0, 13], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh3, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<512x1022x2032xf64>, tensor<f64>) -> tensor<512x1022x2057xf64>
// PAD-NEXT:     %4 = stablehlo.pad %1, %cst, low = [0, 0, 2044], high = [0, 0, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh3, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<512x1022x13xf64>, tensor<f64>) -> tensor<512x1022x2057xf64>
// PAD-NEXT:     %5 = stablehlo.add %2, %3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh3, [{"z"}, {"y"}, {"x"}]>]>} : tensor<512x1022x2057xf64>
// PAD-NEXT:     %6 = stablehlo.add %5, %4 {sdy.sharding = #sdy.sharding_per_value<[<@mesh3, [{"z"}, {"y"}, {"x"}]>]>} : tensor<512x1022x2057xf64>
// PAD-NEXT:     return %6 : tensor<512x1022x2057xf64>
// PAD-NEXT: }
