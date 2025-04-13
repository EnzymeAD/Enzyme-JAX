// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{extend_comm=1 extend_to_pad_comm=0})" %s | FileCheck %s

sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=5]>
func.func @main(%arg0: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x10x82xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg0 [11:12, 7:17, 0:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>) -> tensor<1x10x80xf64>
    %1 = "enzymexla.extend"(%0) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 1 : i64}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x10x80xf64>) -> tensor<1x10x82xf64>
    return %1 : tensor<1x10x82xf64>
}
