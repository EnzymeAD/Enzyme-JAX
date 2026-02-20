module {
  sdy.mesh @mesh = <["x"=2, "y"=2]>
  func.func @foo(%arg24: tensor<20x1536x3072xf32> {enzymexla.memory_effects = [], sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {"x"}]>}) -> (tensor<4x1519x3056xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {"x"}]>}, tensor<4x1519x3056xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {"x"}]>}) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %0:2 = "enzymexla.multi_slice"(%arg24) <{amount = 1 : i32, dimension = 1 : i32, limit_indices = array<i64: 12, 1527, 3064>, start_indices = array<i64: 8, 8, 8>, strides = array<i64: 1, 1, 1>}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}, {"y"}]>, <@mesh, [{}, {"x"}, {"y"}]>]>} : (tensor<20x1536x3072xf32>) -> (tensor<4x1519x3056xf32>, tensor<4x1519x3056xf32>)
    return %0#0, %0#1 : tensor<4x1519x3056xf32>, tensor<4x1519x3056xf32>
  }
}
