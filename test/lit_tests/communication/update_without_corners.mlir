// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{updatewithoutcorners_to_select=1})" %s | FileCheck %s --check-prefix=PAD

sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=4]>
func.func @main(%arg0: tensor<1x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<1x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
    %1 = "enzymexla.update_without_corners"(%arg0, %arg1) <{dimensionX = 1 : i64, x1 = 4 : i64, x2 = 5 : i64, dimensionY = 2 : i64, y1 = 6 : i64, y2 = 7 : i64}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x24x96xf64>, tensor<1x24x96xf64>) -> tensor<1x24x96xf64>
    return %1 : tensor<1x24x96xf64>
}

// CHECK:  func.func @main(%arg0: tensor<1x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<1x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:    %c = stablehlo.constant dense<89> : tensor<1x24x96xi32>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<6> : tensor<1x24x96xi32>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<19> : tensor<1x24x96xi32>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<4> : tensor<1x24x96xi32>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : tensor<1x24x96xi32>
// CHECK-NEXT:    %1 = stablehlo.iota dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : tensor<1x24x96xi32>
// CHECK-NEXT:    %2 = stablehlo.compare  LT, %0, %c_2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x24x96xi32>, tensor<1x24x96xi32>) -> tensor<1x24x96xi1>
// CHECK-NEXT:    %3 = stablehlo.compare  GE, %0, %c_1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x24x96xi32>, tensor<1x24x96xi32>) -> tensor<1x24x96xi1>
// CHECK-NEXT:    %4 = stablehlo.or %2, %3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : tensor<1x24x96xi1>
// CHECK-NEXT:    %5 = stablehlo.compare  LT, %1, %c_0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x24x96xi32>, tensor<1x24x96xi32>) -> tensor<1x24x96xi1>
// CHECK-NEXT:    %6 = stablehlo.compare  GE, %1, %c {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x24x96xi32>, tensor<1x24x96xi32>) -> tensor<1x24x96xi1>
// CHECK-NEXT:    %7 = stablehlo.or %5, %6 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : tensor<1x24x96xi1>
// CHECK-NEXT:    %8 = stablehlo.and %4, %7 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : tensor<1x24x96xi1>
// CHECK-NEXT:    %9 = stablehlo.select %8, %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : tensor<1x24x96xi1>, tensor<1x24x96xf64>
// CHECK-NEXT:    return %9 : tensor<1x24x96xf64>
// CHECK-NEXT:  }
