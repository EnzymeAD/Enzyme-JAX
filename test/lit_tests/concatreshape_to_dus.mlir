// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=concatreshape_to_onedim_dus"  --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module {
  sdy.mesh @mesh = <["x"=2, "y"=1, "z"=1]>
  func.func @main(%arg23 : tensor<20x48x96xf64>) -> tensor<34x80xf64> {
    %370 = stablehlo.slice %arg23 [12:13, 7:8, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x48x96xf64>) -> tensor<1x1x80xf64>
    %337 = stablehlo.slice %arg23 [11:12, 8:40, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x48x96xf64>) -> tensor<1x32x80xf64>
    %371 = stablehlo.slice %arg23 [12:13, 40:41, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x48x96xf64>) -> tensor<1x1x80xf64>

    %656 = stablehlo.reshape %370 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<1x1x80xf64>) -> tensor<1x80xf64>
    %530 = stablehlo.reshape %337 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<1x32x80xf64>) -> tensor<32x80xf64>
    %657 = stablehlo.reshape %371 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<1x1x80xf64>) -> tensor<1x80xf64>
    
    %658 = stablehlo.concatenate %656, %530, %657, dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<1x80xf64>, tensor<32x80xf64>, tensor<1x80xf64>) -> tensor<34x80xf64>

    return %658 : tensor<34x80xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<20x48x96xf64>) -> tensor<34x80xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [11:12, 8:40, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x48x96xf64>) -> tensor<1x32x80xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [12:13, 7:41, 8:88] : (tensor<20x48x96xf64>) -> tensor<1x34x80xf64>
// CHECK-NEXT:    %2 = stablehlo.dynamic_update_slice %1, %0, %c_0, %c, %c_0 : (tensor<1x34x80xf64>, tensor<1x32x80xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x34x80xf64>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<1x34x80xf64>) -> tensor<34x80xf64>
// CHECK-NEXT:    return %3 : tensor<34x80xf64>
// CHECK-NEXT:  }
