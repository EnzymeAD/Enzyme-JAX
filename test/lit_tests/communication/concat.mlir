sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=4]>
func.func @main1(%arg0: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x83xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg1 [0:20, 0:24, 0:3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>) -> tensor<20x24x3xf64>
    %1 = stablehlo.concatenate %arg0, %0, dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>, tensor<20x24x3xf64>) -> tensor<20x24x83xf64>
    return %1 : tensor<20x24x83xf64>
}
