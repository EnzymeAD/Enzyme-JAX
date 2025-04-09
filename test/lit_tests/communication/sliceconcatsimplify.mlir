// RUN: enzymexlamlir-opt --optimize-communication %s | FileCheck %s

// module {
//     sdy.mesh @mesh = <["z"=1, "x"=4, "y"=4]>
//     func.func @main1(%arg0: tensor<528x1024x2048xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<512x1022x2046xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
//         %0 = stablehlo.slice %arg0 [8:520, 1:1023, 2034:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x6xf64>
//         %1 = stablehlo.slice %arg0 [8:520, 1:1023, 8:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x2032xf64>
//         %2 = stablehlo.slice %arg0 [8:520, 1:1023, 8:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x8xf64>
//         %3 = stablehlo.concatenate %0, %1, %2, dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<512x1022x6xf64>, tensor<512x1022x2032xf64>, tensor<512x1022x8xf64>) -> tensor<512x1022x2046xf64>
//         return %3 : tensor<512x1022x2046xf64>
//     }
// }

// nicest case (working)
// module {
//     sdy.mesh @mesh = <["z"=1, "x"=8, "y"=4]>
//     func.func @main2(%arg0: tensor<528x1024x2048xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<512x1022x2048xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
//         %0 = stablehlo.slice %arg0 [8:520, 1:1023, 2032:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x8xf64>
//         %1 = stablehlo.slice %arg0 [8:520, 1:1023, 8:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x2032xf64>
//         %2 = stablehlo.slice %arg0 [8:520, 1:1023, 8:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x8xf64>
//         %3 = stablehlo.concatenate %0, %1, %2, dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<512x1022x8xf64>, tensor<512x1022x2032xf64>, tensor<512x1022x8xf64>) -> tensor<512x1022x2048xf64>
//         return %3 : tensor<512x1022x2048xf64>
//     }
// }

// requires padding I (working)
// module {
//     sdy.mesh @mesh = <["z"=1, "x"=8, "y"=4]>
//     func.func @main2(%arg0: tensor<528x1024x2048xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<512x1022x2057xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
//         %0 = stablehlo.slice %arg0 [8:520, 1:1023, 2028:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x12xf64>
//         %1 = stablehlo.slice %arg0 [8:520, 1:1023, 8:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x2032xf64>
//         %2 = stablehlo.slice %arg0 [8:520, 1:1023, 8:21] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x13xf64>
//         %3 = stablehlo.concatenate %0, %1, %2, dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<512x1022x12xf64>, tensor<512x1022x2032xf64>, tensor<512x1022x13xf64>) -> tensor<512x1022x2057xf64>
//         return %3 : tensor<512x1022x2057xf64>
//     }
// }

// innermost not same as others
// module {
//     sdy.mesh @mesh = <["z"=1, "x"=8, "y"=4]>
//     func.func @main2(%arg0: tensor<528x1024x2048xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<512x1022x2032xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<512x1022x2057xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
//         %0 = stablehlo.slice %arg0 [8:520, 1:1023, 2028:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x12xf64>
//         %2 = stablehlo.slice %arg0 [8:520, 1:1023, 8:21] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x13xf64>
//         %3 = stablehlo.concatenate %0, %arg1, %2, dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<512x1022x12xf64>, tensor<512x1022x2032xf64>, tensor<512x1022x13xf64>) -> tensor<512x1022x2057xf64>
//         return %3 : tensor<512x1022x2057xf64>
//     }
// }

// module {
//     sdy.mesh @mesh = <["z"=1, "x"=8, "y"=4]>
//     func.func @main2(%arg0: tensor<528x1024x2048xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<512x1022x2048xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<512x1022x2057xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
//         %0 = stablehlo.slice %arg0 [8:520, 1:1023, 2028:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x12xf64>
//         %1 = stablehlo.slice %arg1 [0:512, 0:1022, 8:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<512x1022x2048xf64>) -> tensor<512x1022x2032xf64>
//         %2 = stablehlo.slice %arg0 [8:520, 1:1023, 8:21] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x13xf64>
//         %3 = stablehlo.concatenate %0, %1, %2, dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<512x1022x12xf64>, tensor<512x1022x2032xf64>, tensor<512x1022x13xf64>) -> tensor<512x1022x2057xf64>
//         return %3 : tensor<512x1022x2057xf64>
//     }
// }

// module {
//     sdy.mesh @mesh = <["z"=1, "x"=8, "y"=4]>
//     func.func @main2(%arg0: tensor<528x1024x2048xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<512x1022x2032xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<512x1022x2057xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
//         %0 = stablehlo.slice %arg0 [8:520, 1:1023, 2028:2040] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x12xf64>
//         %2 = stablehlo.slice %arg0 [8:520, 1:1023, 8:21] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x13xf64>
//         %3 = stablehlo.concatenate %0, %arg1, %2, dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<512x1022x12xf64>, tensor<512x1022x2032xf64>, tensor<512x1022x13xf64>) -> tensor<512x1022x2057xf64>
//         return %3 : tensor<512x1022x2057xf64>
//     }
// }
