// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=concat_to_onedim_dusslice"  --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module attributes {transform.with_named_sequence} {
  sdy.mesh @mesh = <["x"=2, "y"=2]>
  func.func @"loop!"(%arg0: tensor<6x1522x3056xf64>, %arg1: tensor<4x1x3056xf64>) -> tensor<6x1x3056xf64> {
    %0 = stablehlo.slice %arg0 [5:6, 0:1, 0:3056] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<6x1522x3056xf64>) -> tensor<1x1x3056xf64>
    %2 = stablehlo.slice %arg0 [0:1, 0:1, 0:3056] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<6x1522x3056xf64>) -> tensor<1x1x3056xf64>
    %3 = stablehlo.concatenate %2, %arg1, %0, dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<1x1x3056xf64>, tensor<4x1x3056xf64>, tensor<1x1x3056xf64>) -> tensor<6x1x3056xf64>
    stablehlo.return %3 : tensor<6x1x3056xf64>
  }
}

// CHECK:  func.func @"loop!"(%arg0: tensor<6x1522x3056xf64>, %arg1: tensor<4x1x3056xf64>) -> tensor<6x1x3056xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:6, 0:1, 0:3056] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<6x1522x3056xf64>) -> tensor<6x1x3056xf64>
// CHECK-NEXT:    %1 = stablehlo.dynamic_update_slice %0, %arg1, %c, %c_0, %c_0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<6x1x3056xf64>, tensor<4x1x3056xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<6x1x3056xf64>
// CHECK-NEXT:    stablehlo.return %1 : tensor<6x1x3056xf64>
// CHECK-NEXT:  }

