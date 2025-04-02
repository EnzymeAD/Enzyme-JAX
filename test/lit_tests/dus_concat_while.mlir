
module @"reactant_loop!" attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main() -> (tensor<1x39x62xf32>) {
    %c_270 = stablehlo.constant dense<1> : tensor<i64>
    %c_266 = stablehlo.constant dense<7> : tensor<i64>
    %c_271 = stablehlo.constant dense<0> : tensor<i64>
    %arg29 = "test.get"() : () -> (tensor<1x39x62xf32>)
    %arg39 = "test.get"() : () -> (tensor<i64>)
    %15:2 = stablehlo.while(%iterArg = %c_271, %iterArg_289 = %arg29) : tensor<i64>, tensor<1x39x62xf32>
     cond {
      %35 = stablehlo.compare  LT, %iterArg, %arg39 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %35 : tensor<i1>
    } do {
      %79 = "test.get"() : () -> (tensor<1x24x48xf32>)
      %80 = stablehlo.dynamic_update_slice %iterArg_289, %79, %c_271, %c_266, %c_266 : (tensor<1x39x62xf32>, tensor<1x24x48xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x39x62xf32>
      %89 = stablehlo.slice %80 [0:1, 0:39, 7:55] : (tensor<1x39x62xf32>) -> tensor<1x39x48xf32>
      %90 = stablehlo.slice %80 [0:1, 0:39, 48:55] : (tensor<1x39x62xf32>) -> tensor<1x39x7xf32>
      %91 = stablehlo.slice %80 [0:1, 0:39, 7:14] : (tensor<1x39x62xf32>) -> tensor<1x39x7xf32>
      %92 = stablehlo.concatenate %90, %89, %91, dim = 2 : (tensor<1x39x7xf32>, tensor<1x39x48xf32>, tensor<1x39x7xf32>) -> tensor<1x39x62xf32>
      %add = stablehlo.add %iterArg, %c_270 : tensor<i64>
      stablehlo.return %add, %92 : tensor<i64>, tensor<1x39x62xf32>
    }
    return %15#1 : tensor<1x39x62xf32>
  }
}
