// RUN: enzymexlamlir-opt --optimize-communication %s | FileCheck %s

sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=4]>
func.func @main(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<4x8x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg0 [8:12, 8:16, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x96xf64>) -> tensor<4x8x80xf64>
    %1 = "enzymexla.rotate"(%0) <{amount = 2 : si32, dimension = 2 : si32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x8x80xf64>) -> tensor<4x8x80xf64>
    return %1 : tensor<4x8x80xf64>
}
