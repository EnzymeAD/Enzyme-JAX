// RUN: enzymexlamlir-opt --optimize-communication %s | FileCheck %s

module {
  sdy.mesh @mesh = <["x"=2, "y"=2, "z"=1]>
  func.func @main(%13259 : tensor<20x48x80xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}, %arg31 : tensor<20x48x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> tensor<20x48x96xf64> {
    %13252 = stablehlo.slice %arg31 [0:20, 0:48, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>]>} : (tensor<20x48x96xf64>) -> tensor<20x48x8xf64>
    %13253 = stablehlo.slice %arg31 [0:20, 0:48, 88:96] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>]>} : (tensor<20x48x96xf64>) -> tensor<20x48x8xf64>
    %13260 = stablehlo.concatenate %13252, %13259, %13253, dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>]>} : (tensor<20x48x8xf64>, tensor<20x48x80xf64>, tensor<20x48x8xf64>) -> tensor<20x48x96xf64>
    return %13260 : tensor<20x48x96xf64>
  }
}

// CHECK: func.func
