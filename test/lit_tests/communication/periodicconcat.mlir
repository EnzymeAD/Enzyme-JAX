// RUN: enzymexlamlir-opt --optimize-communication %s | FileCheck %s

sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=4]>
func.func @main1(%arg0: tensor<528x1024x2048xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<512x1022x2046xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg0 [8:520, 1:1023, 2034:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x6xf64>
    %1 = stablehlo.slice %arg0 [8:520, 1:1023, 8:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x2032xf64>
    %2 = stablehlo.slice %arg0 [8:520, 1:1023, 8:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x8xf64>
    %3 = stablehlo.concatenate %0, %1, %2, dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<512x1022x6xf64>, tensor<512x1022x2032xf64>, tensor<512x1022x8xf64>) -> tensor<512x1022x2046xf64>
    return %3 : tensor<512x1022x2046xf64>
}

// CHECK-LABEL: func.func @main1
// CHECK: sdy.manual_computation
// CHECK: "stablehlo.if"
// CHECK: "stablehlo.if"
// CHECK{LITERAL}: %16 = "stablehlo.collective_permute"(%15) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[0, 4], [1, 5], [2, 6], [3, 7], [12, 8], [13, 9], [14, 10], [15, 11]]> : tensor<8x2xi64>}> : (tensor<512x255x4xf64>) -> tensor<512x255x4xf64>
// CHECK: }, {
// CHECK{LITERAL}: %16 = "stablehlo.collective_permute"(%15) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[0, 4], [1, 5], [2, 6], [3, 7], [12, 8], [13, 9], [14, 10], [15, 11]]> : tensor<8x2xi64>}> : (tensor<512x255x4xf64>) -> tensor<512x255x4xf64>
// CHECK: }, {
// CHECK{LITERAL}: %14 = "stablehlo.collective_permute"(%13) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[0, 12], [1, 13], [2, 14], [3, 15]]> : tensor<4x2xi64>}> : (tensor<512x255x8xf64>) -> tensor<512x255x8xf64>
// CHECK{LITERAL}: %16 = "stablehlo.collective_permute"(%15) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[12, 0], [13, 1], [14, 2], [15, 3]]> : tensor<4x2xi64>}> : (tensor<512x255x8xf64>) -> tensor<512x255x8xf64>
// CHECK: "stablehlo.if"
// CHECK: stablehlo.slice
// CHECK: stablehlo.concatenate
// CHECK: stablehlo.return
// CHECK: }, {
// CHECK: stablehlo.slice
// CHECK: stablehlo.concatenate
// CHECK: stablehlo.return

sdy.mesh @mesh2 = <["z"=1, "x"=8, "y"=4]>
func.func @main2(%arg0: tensor<528x1024x2048xf64> {sdy.sharding = #sdy.sharding<@mesh2, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<512x1022x2057xf64> {sdy.sharding = #sdy.sharding<@mesh2, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg0 [8:520, 1:1023, 2028:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x12xf64>
    %1 = stablehlo.slice %arg0 [8:520, 1:1023, 8:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x2032xf64>
    %2 = stablehlo.slice %arg0 [8:520, 1:1023, 8:21] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x13xf64>
    %3 = stablehlo.concatenate %0, %1, %2, dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<512x1022x12xf64>, tensor<512x1022x2032xf64>, tensor<512x1022x13xf64>) -> tensor<512x1022x2057xf64>
    return %3 : tensor<512x1022x2057xf64>
}

// CHECK-LABEL: func.func @main2
// CHECK: sdy.manual_computation
// CHECK: "stablehlo.if"
// CHECK: "stablehlo.if"
// CHECK{LITERAL}: %16 = "stablehlo.collective_permute"(%15) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[0, 4], [1, 5], [2, 6], [3, 7], [4, 8], [5, 9], [6, 10], [7, 11], [8, 12], [9, 13], [10, 14], [11, 15], [20, 16], [21, 17], [22, 18], [23, 19], [24, 20], [25, 21], [26, 22], [27, 23], [28, 24], [29, 25], [30, 26], [31, 27]]> : tensor<24x2xi64>}> : (tensor<512x255x12xf64>) -> tensor<512x255x12xf64>
// CHECK: }, {
// CHECK{LITERAL}: %16 = "stablehlo.collective_permute"(%15) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[0, 4], [1, 5], [2, 6], [3, 7], [4, 8], [5, 9], [6, 10], [7, 11], [8, 12], [9, 13], [10, 14], [11, 15], [20, 16], [21, 17], [22, 18], [23, 19], [24, 20], [25, 21], [26, 22], [27, 23], [28, 24], [29, 25], [30, 26], [31, 27]]> : tensor<24x2xi64>}> : (tensor<512x255x12xf64>) -> tensor<512x255x12xf64>
// CHECK: }, {
// CHECK{LITERAL}: %14 = "stablehlo.collective_permute"(%13) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[0, 28], [1, 29], [2, 30], [3, 31]]> : tensor<4x2xi64>}> : (tensor<512x255x16xf64>) -> tensor<512x255x16xf64>
// CHECK{LITERAL}: %16 = "stablehlo.collective_permute"(%15) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[28, 0], [29, 1], [30, 2], [31, 3]]> : tensor<4x2xi64>}> : (tensor<512x255x16xf64>) -> tensor<512x255x16xf64>
// CHECK: "stablehlo.if"
// CHECK: stablehlo.slice
// CHECK: stablehlo.concatenate
// CHECK: stablehlo.return
// CHECK: }, {
// CHECK: stablehlo.slice
// CHECK: stablehlo.concatenate
// CHECK: stablehlo.return

sdy.mesh @mesh3 = <["z"=1, "x"=8, "y"=4]>
func.func @main3(%arg0: tensor<528x1024x2048xf64> {sdy.sharding = #sdy.sharding<@mesh3, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<512x1022x2032xf64> {sdy.sharding = #sdy.sharding<@mesh3, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<512x1022x2057xf64> {sdy.sharding = #sdy.sharding<@mesh3, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg0 [8:520, 1:1023, 2028:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh3, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x12xf64>
    %2 = stablehlo.slice %arg0 [8:520, 1:1023, 8:21] {sdy.sharding = #sdy.sharding_per_value<[<@mesh3, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x13xf64>
    %3 = stablehlo.concatenate %0, %arg1, %2, dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh3, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<512x1022x12xf64>, tensor<512x1022x2032xf64>, tensor<512x1022x13xf64>) -> tensor<512x1022x2057xf64>
    return %3 : tensor<512x1022x2057xf64>
}

// CHECK-LABEL: func.func @main3
// CHECK: sdy.manual_computation
// CHECK: "stablehlo.if"
// CHECK: "stablehlo.if"
// CHECK{LITERAL}: %16 = "stablehlo.collective_permute"(%15) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[0, 4], [1, 5], [2, 6], [3, 7], [4, 8], [5, 9], [6, 10], [7, 11], [8, 12], [9, 13], [10, 14], [11, 15], [20, 16], [21, 17], [22, 18], [23, 19], [24, 20], [25, 21], [26, 22], [27, 23], [28, 24], [29, 25], [30, 26], [31, 27]]> : tensor<24x2xi64>}> : (tensor<512x255x12xf64>) -> tensor<512x255x12xf64>
// CHECK: }, {
// CHECK{LITERAL}: %16 = "stablehlo.collective_permute"(%15) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[0, 4], [1, 5], [2, 6], [3, 7], [4, 8], [5, 9], [6, 10], [7, 11], [8, 12], [9, 13], [10, 14], [11, 15], [20, 16], [21, 17], [22, 18], [23, 19], [24, 20], [25, 21], [26, 22], [27, 23], [28, 24], [29, 25], [30, 26], [31, 27]]> : tensor<24x2xi64>}> : (tensor<512x255x12xf64>) -> tensor<512x255x12xf64>
// CHECK: }, {
// CHECK{LITERAL}: %14 = "stablehlo.collective_permute"(%13) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[0, 28], [1, 29], [2, 30], [3, 31]]> : tensor<4x2xi64>}> : (tensor<512x255x16xf64>) -> tensor<512x255x16xf64>
// CHECK{LITERAL}: %16 = "stablehlo.collective_permute"(%15) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[28, 0], [29, 1], [30, 2], [31, 3]]> : tensor<4x2xi64>}> : (tensor<512x255x16xf64>) -> tensor<512x255x16xf64>
// CHECK: "stablehlo.if"
// CHECK: stablehlo.slice
// CHECK: stablehlo.concatenate
// CHECK: stablehlo.return
// CHECK: }, {
// CHECK: stablehlo.slice
// CHECK: stablehlo.concatenate
// CHECK: stablehlo.return
