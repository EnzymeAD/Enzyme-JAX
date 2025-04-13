// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{periodic_concat=1 concat_to_pad_comm=0})" %s | FileCheck %s

sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=5]>
func.func @main1(%arg0: tensor<528x1024x2048xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<512x1022x2046xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg0 [8:520, 1:1023, 2034:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x6xf64>
    %1 = stablehlo.slice %arg0 [8:520, 1:1023, 8:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x2032xf64>
    %2 = stablehlo.slice %arg0 [8:520, 1:1023, 8:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x8xf64>
    %3 = stablehlo.concatenate %0, %1, %2, dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<512x1022x6xf64>, tensor<512x1022x2032xf64>, tensor<512x1022x8xf64>) -> tensor<512x1022x2046xf64>
    return %3 : tensor<512x1022x2046xf64>
}

// CHECK:  func.func @main1(%arg0: tensor<528x1024x2048xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<512x1022x2046xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [8:520, 1:1023, 8:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x2032xf64>
// CHECK-NEXT:    %1 = stablehlo.pad %0, %cst, low = [0, 0, 0], high = [0, 3, 0], interior = [0, 0, 0] : (tensor<512x1022x2032xf64>, tensor<f64>) -> tensor<512x1025x2032xf64>
// CHECK-NEXT:    %2 = stablehlo.pad %0, %cst, low = [0, 0, 0], high = [0, 3, 0], interior = [0, 0, 0] : (tensor<512x1022x2032xf64>, tensor<f64>) -> tensor<512x1025x2032xf64>
// CHECK-NEXT:    %3 = sdy.manual_computation(%1, %2) in_shardings=[<@mesh1, [{"z"}, {"y"}, {"x"}]>, <@mesh1, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh1, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "x", "y"} (%arg1: tensor<512x205x508xf64>, %arg2: tensor<512x205x508xf64>) {
// CHECK-NEXT:      %c = stablehlo.constant dense<2> : tensor<ui32>
// CHECK-NEXT:      %c_0 = stablehlo.constant dense<1> : tensor<ui32>
// CHECK-NEXT:      %c_1 = stablehlo.constant dense<4> : tensor<ui32>
// CHECK-NEXT:      %c_2 = stablehlo.constant dense<0> : tensor<ui32>
// CHECK-NEXT:      %5 = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:      %6 = stablehlo.remainder %5, %c_1 : tensor<ui32>
// CHECK-NEXT:      %7 = stablehlo.compare  EQ, %6, %c_2 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %8 = stablehlo.add %5, %c_0 : tensor<ui32>
// CHECK-NEXT:      %9 = stablehlo.remainder %8, %c_1 : tensor<ui32>
// CHECK-NEXT:      %10 = stablehlo.compare  EQ, %9, %c_2 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %11 = stablehlo.not %7 : tensor<i1>
// CHECK-NEXT:      %12 = stablehlo.not %10 : tensor<i1>
// CHECK-NEXT:      %13 = stablehlo.and %11, %12 : tensor<i1>
// CHECK-NEXT:      %14 = "stablehlo.if"(%13) ({
// CHECK-NEXT:        %15 = stablehlo.remainder %5, %c_1 : tensor<ui32>
// CHECK-NEXT:        %16 = stablehlo.compare  LT, %6, %c : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %17 = "stablehlo.if"(%16) ({
// CHECK-NEXT:          %20 = stablehlo.slice %arg1 [0:512, 0:205, 500:508] : (tensor<512x205x508xf64>) -> tensor<512x205x8xf64>
// CHECK-NEXT:          stablehlo.return %20 : tensor<512x205x8xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %20 = stablehlo.slice %arg1 [0:512, 0:205, 0:8] : (tensor<512x205x508xf64>) -> tensor<512x205x8xf64>
// CHECK-NEXT:          stablehlo.return %20 : tensor<512x205x8xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<512x205x8xf64>
// CHECK-NEXT{LITERAL}:        %18 = "stablehlo.collective_permute"(%17) <{channel_handle = #stablehlo.channel_handle<handle = 5, type = 0>, source_target_pairs = dense<[[0, 5], [1, 6], [2, 7], [3, 8], [4, 9], [15, 10], [16, 11], [17, 12], [18, 13], [19, 14]]> : tensor<10x2xi64>}> : (tensor<512x205x8xf64>) -> tensor<512x205x8xf64>
// CHECK-NEXT:        %19 = "stablehlo.if"(%16) ({
// CHECK-NEXT:          %20 = stablehlo.concatenate %18, %arg2, dim = 2 : (tensor<512x205x8xf64>, tensor<512x205x508xf64>) -> tensor<512x205x516xf64>
// CHECK-NEXT:          %21 = stablehlo.multiply %15, %c_1 : tensor<ui32>
// CHECK-NEXT:          %22 = stablehlo.dynamic_slice %20, %c_2, %c_2, %21, sizes = [512, 205, 512] : (tensor<512x205x516xf64>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<512x205x512xf64>
// CHECK-NEXT:          stablehlo.return %22 : tensor<512x205x512xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %20 = stablehlo.concatenate %arg2, %18, dim = 2 : (tensor<512x205x508xf64>, tensor<512x205x8xf64>) -> tensor<512x205x516xf64>
// CHECK-NEXT:          %21 = stablehlo.multiply %15, %c_1 : tensor<ui32>
// CHECK-NEXT:          %22 = stablehlo.subtract %c_1, %21 : tensor<ui32>
// CHECK-NEXT:          %23 = stablehlo.dynamic_slice %20, %c_2, %c_2, %22, sizes = [512, 205, 512] : (tensor<512x205x516xf64>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<512x205x512xf64>
// CHECK-NEXT:          stablehlo.return %23 : tensor<512x205x512xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<512x205x512xf64>
// CHECK-NEXT:        stablehlo.return %19 : tensor<512x205x512xf64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        %15 = "stablehlo.if"(%7) ({
// CHECK-NEXT:          %18 = stablehlo.slice %arg1 [0:512, 0:205, 0:8] : (tensor<512x205x508xf64>) -> tensor<512x205x8xf64>
// CHECK-NEXT:          stablehlo.return %18 : tensor<512x205x8xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %18 = stablehlo.slice %arg1 [0:512, 0:205, 500:508] : (tensor<512x205x508xf64>) -> tensor<512x205x8xf64>
// CHECK-NEXT:          stablehlo.return %18 : tensor<512x205x8xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<512x205x8xf64>
// CHECK-NEXT{LITERAL}:        %16 = "stablehlo.collective_permute"(%15) <{channel_handle = #stablehlo.channel_handle<handle = 6, type = 0>, source_target_pairs = dense<[[0, 15], [1, 16], [2, 17], [3, 18], [4, 19], [15, 0], [16, 1], [17, 2], [18, 3], [19, 4]]> : tensor<10x2xi64>}> : (tensor<512x205x8xf64>) -> tensor<512x205x8xf64>
// CHECK-NEXT:        %17 = "stablehlo.if"(%7) ({
// CHECK-NEXT:          %18 = stablehlo.slice %arg2 [0:512, 0:205, 0:504] : (tensor<512x205x508xf64>) -> tensor<512x205x504xf64>
// CHECK-NEXT:          %19 = stablehlo.concatenate %16, %18, dim = 2 : (tensor<512x205x8xf64>, tensor<512x205x504xf64>) -> tensor<512x205x512xf64>
// CHECK-NEXT:          stablehlo.return %19 : tensor<512x205x512xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %18 = stablehlo.slice %arg2 [0:512, 0:205, 4:508] : (tensor<512x205x508xf64>) -> tensor<512x205x504xf64>
// CHECK-NEXT:          %19 = stablehlo.concatenate %18, %16, dim = 2 : (tensor<512x205x504xf64>, tensor<512x205x8xf64>) -> tensor<512x205x512xf64>
// CHECK-NEXT:          stablehlo.return %19 : tensor<512x205x512xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<512x205x512xf64>
// CHECK-NEXT:        stablehlo.return %17 : tensor<512x205x512xf64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<512x205x512xf64>
// CHECK-NEXT:      sdy.return %14 : tensor<512x205x512xf64>
// CHECK-NEXT:    } : (tensor<512x1025x2032xf64>, tensor<512x1025x2032xf64>) -> tensor<512x1025x2048xf64>
// CHECK-NEXT:    %4 = stablehlo.slice %3 [0:512, 0:1022, 2:2048] : (tensor<512x1025x2048xf64>) -> tensor<512x1022x2046xf64>
// CHECK-NEXT:    return %4 : tensor<512x1022x2046xf64>
// CHECK-NEXT:  }



sdy.mesh @mesh2 = <["z"=1, "x"=8, "y"=5]>
func.func @main2(%arg0: tensor<528x1024x2048xf64> {sdy.sharding = #sdy.sharding<@mesh2, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<512x1022x2057xf64> {sdy.sharding = #sdy.sharding<@mesh2, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg0 [8:520, 1:1023, 2028:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x12xf64>
    %1 = stablehlo.slice %arg0 [8:520, 1:1023, 8:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x2032xf64>
    %2 = stablehlo.slice %arg0 [8:520, 1:1023, 8:21] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x13xf64>
    %3 = stablehlo.concatenate %0, %1, %2, dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<512x1022x12xf64>, tensor<512x1022x2032xf64>, tensor<512x1022x13xf64>) -> tensor<512x1022x2057xf64>
    return %3 : tensor<512x1022x2057xf64>
}

// CHECK:  func.func @main2(%arg0: tensor<528x1024x2048xf64> {sdy.sharding = #sdy.sharding<@mesh2, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<512x1022x2057xf64> {sdy.sharding = #sdy.sharding<@mesh2, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [8:520, 1:1023, 8:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x2032xf64>
// CHECK-NEXT:    %1 = stablehlo.pad %0, %cst, low = [0, 0, 0], high = [0, 3, 0], interior = [0, 0, 0] : (tensor<512x1022x2032xf64>, tensor<f64>) -> tensor<512x1025x2032xf64>
// CHECK-NEXT:    %2 = stablehlo.pad %0, %cst, low = [0, 0, 0], high = [0, 3, 0], interior = [0, 0, 0] : (tensor<512x1022x2032xf64>, tensor<f64>) -> tensor<512x1025x2032xf64>
// CHECK-NEXT:    %3 = sdy.manual_computation(%1, %2) in_shardings=[<@mesh2, [{"z"}, {"y"}, {"x"}]>, <@mesh2, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh2, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "x", "y"} (%arg1: tensor<512x205x254xf64>, %arg2: tensor<512x205x254xf64>) {
// CHECK-NEXT:      %c = stablehlo.constant dense<12> : tensor<ui32>
// CHECK-NEXT:      %c_0 = stablehlo.constant dense<4> : tensor<ui32>
// CHECK-NEXT:      %c_1 = stablehlo.constant dense<1> : tensor<ui32>
// CHECK-NEXT:      %c_2 = stablehlo.constant dense<8> : tensor<ui32>
// CHECK-NEXT:      %c_3 = stablehlo.constant dense<0> : tensor<ui32>
// CHECK-NEXT:      %5 = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:      %6 = stablehlo.remainder %5, %c_2 : tensor<ui32>
// CHECK-NEXT:      %7 = stablehlo.compare  EQ, %6, %c_3 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %8 = stablehlo.add %5, %c_1 : tensor<ui32>
// CHECK-NEXT:      %9 = stablehlo.remainder %8, %c_2 : tensor<ui32>
// CHECK-NEXT:      %10 = stablehlo.compare  EQ, %9, %c_3 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %11 = stablehlo.not %7 : tensor<i1>
// CHECK-NEXT:      %12 = stablehlo.not %10 : tensor<i1>
// CHECK-NEXT:      %13 = stablehlo.and %11, %12 : tensor<i1>
// CHECK-NEXT:      %14 = "stablehlo.if"(%13) ({
// CHECK-NEXT:        %15 = stablehlo.remainder %5, %c_2 : tensor<ui32>
// CHECK-NEXT:        %16 = stablehlo.compare  LT, %6, %c_0 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %17 = "stablehlo.if"(%16) ({
// CHECK-NEXT:          %20 = stablehlo.slice %arg1 [0:512, 0:205, 238:254] : (tensor<512x205x254xf64>) -> tensor<512x205x16xf64>
// CHECK-NEXT:          stablehlo.return %20 : tensor<512x205x16xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %20 = stablehlo.slice %arg1 [0:512, 0:205, 0:16] : (tensor<512x205x254xf64>) -> tensor<512x205x16xf64>
// CHECK-NEXT:          stablehlo.return %20 : tensor<512x205x16xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<512x205x16xf64>
// CHECK-NEXT{LITERAL}:        %18 = "stablehlo.collective_permute"(%17) <{channel_handle = #stablehlo.channel_handle<handle = 3, type = 0>, source_target_pairs = dense<[[0, 5], [1, 6], [2, 7], [3, 8], [4, 9], [5, 10], [6, 11], [7, 12], [8, 13], [9, 14], [10, 15], [11, 16], [12, 17], [13, 18], [14, 19], [25, 20], [26, 21], [27, 22], [28, 23], [29, 24], [30, 25], [31, 26], [32, 27], [33, 28], [34, 29], [35, 30], [36, 31], [37, 32], [38, 33], [39, 34]]> : tensor<30x2xi64>}> : (tensor<512x205x16xf64>) -> tensor<512x205x16xf64>
// CHECK-NEXT:        %19 = "stablehlo.if"(%16) ({
// CHECK-NEXT:          %20 = stablehlo.concatenate %18, %arg2, dim = 2 : (tensor<512x205x16xf64>, tensor<512x205x254xf64>) -> tensor<512x205x270xf64>
// CHECK-NEXT:          %21 = stablehlo.multiply %15, %c_0 : tensor<ui32>
// CHECK-NEXT:          %22 = stablehlo.dynamic_slice %20, %c_3, %c_3, %21, sizes = [512, 205, 258] : (tensor<512x205x270xf64>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<512x205x258xf64>
// CHECK-NEXT:          stablehlo.return %22 : tensor<512x205x258xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %20 = stablehlo.concatenate %arg2, %18, dim = 2 : (tensor<512x205x254xf64>, tensor<512x205x16xf64>) -> tensor<512x205x270xf64>
// CHECK-NEXT:          %21 = stablehlo.multiply %15, %c_0 : tensor<ui32>
// CHECK-NEXT:          %22 = stablehlo.subtract %c, %21 : tensor<ui32>
// CHECK-NEXT:          %23 = stablehlo.dynamic_slice %20, %c_3, %c_3, %22, sizes = [512, 205, 258] : (tensor<512x205x270xf64>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<512x205x258xf64>
// CHECK-NEXT:          stablehlo.return %23 : tensor<512x205x258xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<512x205x258xf64>
// CHECK-NEXT:        stablehlo.return %19 : tensor<512x205x258xf64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        %15 = "stablehlo.if"(%7) ({
// CHECK-NEXT:          %18 = stablehlo.slice %arg1 [0:512, 0:205, 0:16] : (tensor<512x205x254xf64>) -> tensor<512x205x16xf64>
// CHECK-NEXT:          stablehlo.return %18 : tensor<512x205x16xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %18 = stablehlo.slice %arg1 [0:512, 0:205, 238:254] : (tensor<512x205x254xf64>) -> tensor<512x205x16xf64>
// CHECK-NEXT:          stablehlo.return %18 : tensor<512x205x16xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<512x205x16xf64>
// CHECK-NEXT{LITERAL}:        %16 = "stablehlo.collective_permute"(%15) <{channel_handle = #stablehlo.channel_handle<handle = 4, type = 0>, source_target_pairs = dense<[[0, 35], [1, 36], [2, 37], [3, 38], [4, 39], [35, 0], [36, 1], [37, 2], [38, 3], [39, 4]]> : tensor<10x2xi64>}> : (tensor<512x205x16xf64>) -> tensor<512x205x16xf64>
// CHECK-NEXT:        %17 = "stablehlo.if"(%7) ({
// CHECK-NEXT:          %18 = stablehlo.slice %arg2 [0:512, 0:205, 0:242] : (tensor<512x205x254xf64>) -> tensor<512x205x242xf64>
// CHECK-NEXT:          %19 = stablehlo.concatenate %16, %18, dim = 2 : (tensor<512x205x16xf64>, tensor<512x205x242xf64>) -> tensor<512x205x258xf64>
// CHECK-NEXT:          stablehlo.return %19 : tensor<512x205x258xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %18 = stablehlo.slice %arg2 [0:512, 0:205, 12:254] : (tensor<512x205x254xf64>) -> tensor<512x205x242xf64>
// CHECK-NEXT:          %19 = stablehlo.concatenate %18, %16, dim = 2 : (tensor<512x205x242xf64>, tensor<512x205x16xf64>) -> tensor<512x205x258xf64>
// CHECK-NEXT:          stablehlo.return %19 : tensor<512x205x258xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<512x205x258xf64>
// CHECK-NEXT:        stablehlo.return %17 : tensor<512x205x258xf64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<512x205x258xf64>
// CHECK-NEXT:      sdy.return %14 : tensor<512x205x258xf64>
// CHECK-NEXT:    } : (tensor<512x1025x2032xf64>, tensor<512x1025x2032xf64>) -> tensor<512x1025x2064xf64>
// CHECK-NEXT:    %4 = stablehlo.slice %3 [0:512, 0:1022, 4:2061] : (tensor<512x1025x2064xf64>) -> tensor<512x1022x2057xf64>
// CHECK-NEXT:    return %4 : tensor<512x1022x2057xf64>
// CHECK-NEXT:  }

sdy.mesh @mesh3 = <["z"=1, "x"=8, "y"=5]>
func.func @main3(%arg0: tensor<528x1024x2048xf64> {sdy.sharding = #sdy.sharding<@mesh3, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<512x1022x2032xf64> {sdy.sharding = #sdy.sharding<@mesh3, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<512x1022x2057xf64> {sdy.sharding = #sdy.sharding<@mesh3, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg0 [8:520, 1:1023, 2028:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh3, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x12xf64>
    %2 = stablehlo.slice %arg0 [8:520, 1:1023, 8:21] {sdy.sharding = #sdy.sharding_per_value<[<@mesh3, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x13xf64>
    %3 = stablehlo.concatenate %0, %arg1, %2, dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh3, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<512x1022x12xf64>, tensor<512x1022x2032xf64>, tensor<512x1022x13xf64>) -> tensor<512x1022x2057xf64>
    return %3 : tensor<512x1022x2057xf64>
}

// CHECK:  func.func @main3(%arg0: tensor<528x1024x2048xf64> {sdy.sharding = #sdy.sharding<@mesh3, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<512x1022x2032xf64> {sdy.sharding = #sdy.sharding<@mesh3, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<512x1022x2057xf64> {sdy.sharding = #sdy.sharding<@mesh3, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [8:520, 1:1023, 8:2040] : (tensor<528x1024x2048xf64>) -> tensor<512x1022x2032xf64>
// CHECK-NEXT:    %1 = stablehlo.pad %0, %cst, low = [0, 0, 0], high = [0, 3, 0], interior = [0, 0, 0] : (tensor<512x1022x2032xf64>, tensor<f64>) -> tensor<512x1025x2032xf64>
// CHECK-NEXT:    %2 = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 3, 0], interior = [0, 0, 0] : (tensor<512x1022x2032xf64>, tensor<f64>) -> tensor<512x1025x2032xf64>
// CHECK-NEXT:    %3 = sdy.manual_computation(%1, %2) in_shardings=[<@mesh3, [{"z"}, {"y"}, {"x"}]>, <@mesh3, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh3, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "x", "y"} (%arg2: tensor<512x205x254xf64>, %arg3: tensor<512x205x254xf64>) {
// CHECK-NEXT:      %c = stablehlo.constant dense<12> : tensor<ui32>
// CHECK-NEXT:      %c_0 = stablehlo.constant dense<4> : tensor<ui32>
// CHECK-NEXT:      %c_1 = stablehlo.constant dense<1> : tensor<ui32>
// CHECK-NEXT:      %c_2 = stablehlo.constant dense<8> : tensor<ui32>
// CHECK-NEXT:      %c_3 = stablehlo.constant dense<0> : tensor<ui32>
// CHECK-NEXT:      %5 = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:      %6 = stablehlo.remainder %5, %c_2 : tensor<ui32>
// CHECK-NEXT:      %7 = stablehlo.compare  EQ, %6, %c_3 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %8 = stablehlo.add %5, %c_1 : tensor<ui32>
// CHECK-NEXT:      %9 = stablehlo.remainder %8, %c_2 : tensor<ui32>
// CHECK-NEXT:      %10 = stablehlo.compare  EQ, %9, %c_3 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %11 = stablehlo.not %7 : tensor<i1>
// CHECK-NEXT:      %12 = stablehlo.not %10 : tensor<i1>
// CHECK-NEXT:      %13 = stablehlo.and %11, %12 : tensor<i1>
// CHECK-NEXT:      %14 = "stablehlo.if"(%13) ({
// CHECK-NEXT:        %15 = stablehlo.remainder %5, %c_2 : tensor<ui32>
// CHECK-NEXT:        %16 = stablehlo.compare  LT, %6, %c_0 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %17 = "stablehlo.if"(%16) ({
// CHECK-NEXT:          %20 = stablehlo.slice %arg2 [0:512, 0:205, 238:254] : (tensor<512x205x254xf64>) -> tensor<512x205x16xf64>
// CHECK-NEXT:          stablehlo.return %20 : tensor<512x205x16xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %20 = stablehlo.slice %arg2 [0:512, 0:205, 0:16] : (tensor<512x205x254xf64>) -> tensor<512x205x16xf64>
// CHECK-NEXT:          stablehlo.return %20 : tensor<512x205x16xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<512x205x16xf64>
// CHECK-NEXT{LITERAL}:        %18 = "stablehlo.collective_permute"(%17) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[0, 5], [1, 6], [2, 7], [3, 8], [4, 9], [5, 10], [6, 11], [7, 12], [8, 13], [9, 14], [10, 15], [11, 16], [12, 17], [13, 18], [14, 19], [25, 20], [26, 21], [27, 22], [28, 23], [29, 24], [30, 25], [31, 26], [32, 27], [33, 28], [34, 29], [35, 30], [36, 31], [37, 32], [38, 33], [39, 34]]> : tensor<30x2xi64>}> : (tensor<512x205x16xf64>) -> tensor<512x205x16xf64>
// CHECK-NEXT:        %19 = "stablehlo.if"(%16) ({
// CHECK-NEXT:          %20 = stablehlo.concatenate %18, %arg3, dim = 2 : (tensor<512x205x16xf64>, tensor<512x205x254xf64>) -> tensor<512x205x270xf64>
// CHECK-NEXT:          %21 = stablehlo.multiply %15, %c_0 : tensor<ui32>
// CHECK-NEXT:          %22 = stablehlo.dynamic_slice %20, %c_3, %c_3, %21, sizes = [512, 205, 258] : (tensor<512x205x270xf64>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<512x205x258xf64>
// CHECK-NEXT:          stablehlo.return %22 : tensor<512x205x258xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %20 = stablehlo.concatenate %arg3, %18, dim = 2 : (tensor<512x205x254xf64>, tensor<512x205x16xf64>) -> tensor<512x205x270xf64>
// CHECK-NEXT:          %21 = stablehlo.multiply %15, %c_0 : tensor<ui32>
// CHECK-NEXT:          %22 = stablehlo.subtract %c, %21 : tensor<ui32>
// CHECK-NEXT:          %23 = stablehlo.dynamic_slice %20, %c_3, %c_3, %22, sizes = [512, 205, 258] : (tensor<512x205x270xf64>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<512x205x258xf64>
// CHECK-NEXT:          stablehlo.return %23 : tensor<512x205x258xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<512x205x258xf64>
// CHECK-NEXT:        stablehlo.return %19 : tensor<512x205x258xf64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        %15 = "stablehlo.if"(%7) ({
// CHECK-NEXT:          %18 = stablehlo.slice %arg2 [0:512, 0:205, 0:16] : (tensor<512x205x254xf64>) -> tensor<512x205x16xf64>
// CHECK-NEXT:          stablehlo.return %18 : tensor<512x205x16xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %18 = stablehlo.slice %arg2 [0:512, 0:205, 238:254] : (tensor<512x205x254xf64>) -> tensor<512x205x16xf64>
// CHECK-NEXT:          stablehlo.return %18 : tensor<512x205x16xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<512x205x16xf64>
// CHECK-NEXT{LITERAL}:        %16 = "stablehlo.collective_permute"(%15) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 0>, source_target_pairs = dense<[[0, 35], [1, 36], [2, 37], [3, 38], [4, 39], [35, 0], [36, 1], [37, 2], [38, 3], [39, 4]]> : tensor<10x2xi64>}> : (tensor<512x205x16xf64>) -> tensor<512x205x16xf64>
// CHECK-NEXT:        %17 = "stablehlo.if"(%7) ({
// CHECK-NEXT:          %18 = stablehlo.slice %arg3 [0:512, 0:205, 0:242] : (tensor<512x205x254xf64>) -> tensor<512x205x242xf64>
// CHECK-NEXT:          %19 = stablehlo.concatenate %16, %18, dim = 2 : (tensor<512x205x16xf64>, tensor<512x205x242xf64>) -> tensor<512x205x258xf64>
// CHECK-NEXT:          stablehlo.return %19 : tensor<512x205x258xf64>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %18 = stablehlo.slice %arg3 [0:512, 0:205, 12:254] : (tensor<512x205x254xf64>) -> tensor<512x205x242xf64>
// CHECK-NEXT:          %19 = stablehlo.concatenate %18, %16, dim = 2 : (tensor<512x205x242xf64>, tensor<512x205x16xf64>) -> tensor<512x205x258xf64>
// CHECK-NEXT:          stablehlo.return %19 : tensor<512x205x258xf64>
// CHECK-NEXT:        }) : (tensor<i1>) -> tensor<512x205x258xf64>
// CHECK-NEXT:        stablehlo.return %17 : tensor<512x205x258xf64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<512x205x258xf64>
// CHECK-NEXT:      sdy.return %14 : tensor<512x205x258xf64>
// CHECK-NEXT:    } : (tensor<512x1025x2032xf64>, tensor<512x1025x2032xf64>) -> tensor<512x1025x2064xf64>
// CHECK-NEXT:    %4 = stablehlo.slice %3 [0:512, 0:1022, 4:2061] : (tensor<512x1025x2064xf64>) -> tensor<512x1022x2057xf64>
// CHECK-NEXT:    return %4 : tensor<512x1022x2057xf64>
// CHECK-NEXT:  }