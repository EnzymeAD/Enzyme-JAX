// RUN: enzymexlamlir-opt --sdy-propagation-pipeline '--sdy-insert-explicit-reshards=enable-full-version=true' --sdy-reshard-to-collectives %s | FileCheck %s
sdy.mesh @mesh = <["x"=4, "y"=2]>
func.func @dotadd(%arg0: tensor<32x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>},
                 %arg1: tensor<8x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>},
                 %arg2: tensor<32x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<32x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
    %0 = stablehlo.dot %arg0, %arg1 : (tensor<32x8xf32>, tensor<8x64xf32>) -> tensor<32x64xf32>
    %1 = stablehlo.add %0, %arg2 : (tensor<32x64xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
    return %1 : tensor<32x64xf32>
}


func.func @dotadd2(%arg0: tensor<32x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<8x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}, %arg2: tensor<32x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<32x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
    %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<32x8xf32>, tensor<8x64xf32>) -> tensor<32x64xf32>
    %1 = stablehlo.add %0, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : tensor<32x64xf32>
    return %1 : tensor<32x64xf32>
}

func.func @dottest(%lhs : tensor<8x32xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"x"}, {"y"}]>},
%rhs : tensor<32x16xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x16xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  %0 = stablehlo.dot %lhs, %rhs {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>}
  : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}