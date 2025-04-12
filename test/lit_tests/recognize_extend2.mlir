// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=recognize_extend" --transform-interpreter --enzyme-hlo-remove-transform %s --split-input-file | FileCheck %s

 sdy.mesh @mesh = <["x"=2, "y"=1, "z"=1]>
  func.func @main(%5743: tensor<4x8x80xf64>) -> (tensor<3x8x80xf64>) {
      %12500 = stablehlo.slice %5743 [0:1, 0:8, 0:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>]>} : (tensor<4x8x80xf64>) -> tensor<1x8x80xf64>
      %12501 = stablehlo.slice %5743 [0:2, 0:8, 0:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>]>} : (tensor<4x8x80xf64>) -> tensor<2x8x80xf64>
      %RES = stablehlo.concatenate %12500, %12501, dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>]>} : (tensor<1x8x80xf64>, tensor<2x8x80xf64>) -> tensor<3x8x80xf64>
      func.return %RES :  tensor<3x8x80xf64>
  }

  func.func @main2(%arg29 : tensor<1x24x96xf64>, %5744 : tensor<4x8x80xf64>) -> tensor<3x8x80xf64>{
    %12500 = stablehlo.slice %5744 [3:4, 0:8, 0:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>]>} : (tensor<4x8x80xf64>) -> tensor<1x8x80xf64>
    %12501 = stablehlo.slice %5744 [2:4, 0:8, 0:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>]>} : (tensor<4x8x80xf64>) -> tensor<2x8x80xf64>
    %12502 = stablehlo.concatenate %12501, %12500, dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>]>} : (tensor<2x8x80xf64>, tensor<1x8x80xf64>) -> tensor<3x8x80xf64>
    return %12502 : tensor<3x8x80xf64>
  }

// CHECK:    func.func @main(%arg0: tensor<4x8x80xf64>) -> tensor<3x8x80xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:2, 0:8, 0:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>]>} : (tensor<4x8x80xf64>) -> tensor<2x8x80xf64>
// CHECK-NEXT:    %1 = "enzymexla.extend"(%0) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 0 : i64}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>]>} : (tensor<2x8x80xf64>) -> tensor<3x8x80xf64>
// CHECK-NEXT:    return %1 : tensor<3x8x80xf64>
// CHECK-NEXT:  }

// CHECK:  func.func @main2(%arg0: tensor<1x24x96xf64>, %arg1: tensor<4x8x80xf64>) -> tensor<3x8x80xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [2:4, 0:8, 0:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>]>} : (tensor<4x8x80xf64>) -> tensor<2x8x80xf64>
// CHECK-NEXT:    %1 = "enzymexla.extend"(%0) <{dimension = 0 : i64, lhs = 0 : i64, rhs = 1 : i64}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>]>} : (tensor<2x8x80xf64>) -> tensor<3x8x80xf64>
// CHECK-NEXT:    return %1 : tensor<3x8x80xf64>
// CHECK-NEXT:  }