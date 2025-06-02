// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{periodic_concat=1 concat_to_pad_comm=0})" %s | FileCheck %s --check-prefix=CPERM
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{periodic_concat=0 concat_to_pad_comm=1})" %s | FileCheck %s --check-prefix=PAD

module {
  sdy.mesh @mesh = <["x"=2, "y"=2, "z"=1]>
  func.func @main(%13259 : tensor<20x48x80xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}, %arg31 : tensor<20x48x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> tensor<20x48x96xf64> {
    %13252 = stablehlo.slice %arg31 [0:20, 0:48, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x48x96xf64>) -> tensor<20x48x8xf64>
    %13253 = stablehlo.slice %arg31 [0:20, 0:48, 88:96] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x48x96xf64>) -> tensor<20x48x8xf64>
    %13260 = stablehlo.concatenate %13252, %13259, %13253, dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x48x8xf64>, tensor<20x48x80xf64>, tensor<20x48x8xf64>) -> tensor<20x48x96xf64>
    return %13260 : tensor<20x48x96xf64>
  }
}

// CPERM: func.func

// PAD: sdy.mesh @mesh = <["x"=2, "y"=2, "z"=1]>
// PAD-NEXT:  func.func @main(%arg0: tensor<20x48x80xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<20x48x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> tensor<20x48x96xf64> {
// PAD-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// PAD-NEXT:    %0 = stablehlo.slice %arg1 [0:20, 0:48, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x48x96xf64>) -> tensor<20x48x8xf64>
// PAD-NEXT:    %1 = stablehlo.slice %arg1 [0:20, 0:48, 88:96] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x48x96xf64>) -> tensor<20x48x8xf64>
// PAD-NEXT:    %2 = stablehlo.pad %0, %cst, low = [0, 0, 0], high = [0, 0, 88], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x48x8xf64>, tensor<f64>) -> tensor<20x48x96xf64>
// PAD-NEXT:    %3 = stablehlo.pad %arg0, %cst, low = [0, 0, 8], high = [0, 0, 8], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x48x80xf64>, tensor<f64>) -> tensor<20x48x96xf64>
// PAD-NEXT:    %4 = stablehlo.pad %1, %cst, low = [0, 0, 88], high = [0, 0, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x48x8xf64>, tensor<f64>) -> tensor<20x48x96xf64>
// PAD-NEXT:    %5 = stablehlo.add %2, %3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : tensor<20x48x96xf64>
// PAD-NEXT:    %6 = stablehlo.add %5, %4 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : tensor<20x48x96xf64>
// PAD-NEXT:    return %6 : tensor<20x48x96xf64>
// PAD-NEXT:  }
