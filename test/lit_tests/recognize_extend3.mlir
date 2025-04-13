// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=recognize_extend"  --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module {
  sdy.mesh @mesh = <["x"=2, "y"=1, "z"=1]>
  func.func @main(%arg23 : tensor<20x48x96xf64>) -> tensor<34x80xf64> {

    %506 = stablehlo.slice %arg23 [11:12, 8:9, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x48x96xf64>) -> tensor<1x1x80xf64>
    %507 = stablehlo.reshape %506 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<1x1x80xf64>) -> tensor<1x80xf64> 
    
    %335 = stablehlo.slice %arg23 [11:12, 8:40, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x48x96xf64>) -> tensor<1x32x80xf64>

    %508 = stablehlo.reshape %335 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<1x32x80xf64>) -> tensor<32x80xf64> 
    
    %509 = stablehlo.slice %arg23 [11:12, 39:40, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x48x96xf64>) -> tensor<1x1x80xf64>

    %510 = stablehlo.reshape %509 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<1x1x80xf64>) -> tensor<1x80xf64>

    %511 = stablehlo.concatenate %507, %508, %510, dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<1x80xf64>, tensor<32x80xf64>, tensor<1x80xf64>) -> tensor<34x80xf64> 

    return %511 : tensor<34x80xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<20x48x96xf64>) -> tensor<34x80xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [11:12, 8:40, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x48x96xf64>) -> tensor<1x32x80xf64>
// CHECK-NEXT:    %1 = "enzymexla.extend"(%0) <{dimension = 1 : i64, lhs = 1 : i64, rhs = 1 : i64}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x32x80xf64>) -> tensor<1x34x80xf64>
// CHECK-NEXT:    %2 = stablehlo.reshape %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<1x34x80xf64>) -> tensor<34x80xf64>
// CHECK-NEXT:    return %2 : tensor<34x80xf64>
// CHECK-NEXT:  }