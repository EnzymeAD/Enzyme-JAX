// RUN: enzymexlamlir-opt --optimize-communication %s | FileCheck %s

module {
  sdy.mesh @mesh = <["x"=2, "y"=1, "z"=1]>
  func.func @wrap(%7175 : tensor<3x10x80xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> tensor<3x10x82xf64> {
    %7542 = "enzymexla.wrap"(%7175) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 1 : i64}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>]>} : (tensor<3x10x80xf64>) -> tensor<3x10x82xf64>
    stablehlo.return %7542 : tensor<3x10x82xf64>
  }
}
