// RUN: enzymexlamlir-opt --optimize-communication %s | FileCheck %s

sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=4]>
func.func @main1(%arg0: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x83xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg1 [0:20, 0:24, 0:3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>) -> tensor<20x24x3xf64>
    %1 = stablehlo.concatenate %arg0, %0, dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>, tensor<20x24x3xf64>) -> tensor<20x24x83xf64>
    return %1 : tensor<20x24x83xf64>
}

func.func @main2(%arg0: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x81xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg1 [0:20, 0:24, 0:1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>) -> tensor<20x24x1xf64>
    %1 = stablehlo.concatenate %0, %arg0, dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x1xf64>, tensor<20x24x80xf64>) -> tensor<20x24x81xf64>
    return %1 : tensor<20x24x81xf64>
}

// CHECK: module {
// CHECK-NEXT:   sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=4]>
// CHECK-NEXT:   func.func @main1(%arg0: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x83xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.slice %arg1 [0:20, 0:24, 0:3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>) -> tensor<20x24x3xf64>
// CHECK-NEXT:     %1 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 0, 3], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>, tensor<f64>) -> tensor<20x24x83xf64>
// CHECK-NEXT:     %2 = stablehlo.pad %0, %cst, low = [0, 0, 80], high = [0, 0, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x3xf64>, tensor<f64>) -> tensor<20x24x83xf64>
// CHECK-NEXT:     %3 = stablehlo.add %1, %2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : tensor<20x24x83xf64>
// CHECK-NEXT:     return %3 : tensor<20x24x83xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @main2(%arg0: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x81xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.slice %arg1 [0:20, 0:24, 0:1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>) -> tensor<20x24x1xf64>
// CHECK-NEXT:     %1 = stablehlo.pad %0, %cst, low = [0, 0, 0], high = [0, 0, 80], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x1xf64>, tensor<f64>) -> tensor<20x24x81xf64>
// CHECK-NEXT:     %2 = stablehlo.pad %arg0, %cst, low = [0, 0, 1], high = [0, 0, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>, tensor<f64>) -> tensor<20x24x81xf64>
// CHECK-NEXT:     %3 = stablehlo.add %1, %2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : tensor<20x24x81xf64>
// CHECK-NEXT:     return %3 : tensor<20x24x81xf64>
// CHECK-NEXT:   }
// CHECK-NEXT: }
