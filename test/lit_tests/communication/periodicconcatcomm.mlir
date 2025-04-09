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
// CHECK: "stablehlo.collective_permute"
// CHECK: }, {
// CHECK: "stablehlo.collective_permute"
// CHECK: }, {
// CHECK: "stablehlo.collective_permute"
// CHECK: "stablehlo.collective_permute"
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
// CHECK: "stablehlo.collective_permute"
// CHECK: }, {
// CHECK: "stablehlo.collective_permute"
// CHECK: }, {
// CHECK: "stablehlo.collective_permute"
// CHECK: "stablehlo.collective_permute"
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
// CHECK: "stablehlo.collective_permute"
// CHECK: }, {
// CHECK: "stablehlo.collective_permute"
// CHECK: }, {
// CHECK: "stablehlo.collective_permute"
// CHECK: "stablehlo.collective_permute"
// CHECK: "stablehlo.if"
// CHECK: stablehlo.slice
// CHECK: stablehlo.concatenate
// CHECK: stablehlo.return
// CHECK: }, {
// CHECK: stablehlo.slice
// CHECK: stablehlo.concatenate
// CHECK: stablehlo.return
