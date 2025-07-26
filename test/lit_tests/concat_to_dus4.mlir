// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=concat_to_onedim_dus"  --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

sdy.mesh @mesh = <["z"=1, "x"=4, "y"=4]>

func.func @main(%10148 : tensor<4x1520x3056xf64>, %10342 :  tensor<4x1x3056xf64>, %10341 : tensor<4x1515x3056xf64>, %10210 : tensor<4x1x3056xf64>) -> tensor<4x1519x3056xf64> {
  %10211 = stablehlo.slice %10148 [0:4, 0:1, 0:3056] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<4x1520x3056xf64>) -> tensor<4x1x3056xf64>
  %10212 = stablehlo.slice %10148 [0:4, 1518:1519, 0:3056] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<4x1520x3056xf64>) -> tensor<4x1x3056xf64>
  %10343 = stablehlo.concatenate %10211, %10342, %10341, %10210, %10212, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<4x1x3056xf64>, tensor<4x1x3056xf64>, tensor<4x1515x3056xf64>, tensor<4x1x3056xf64>, tensor<4x1x3056xf64>) -> tensor<4x1519x3056xf64>
  return %10343 : tensor<4x1519x3056xf64>
}

// CHECK:  func.func @main(%arg0: tensor<4x1520x3056xf64>, %arg1: tensor<4x1x3056xf64>, %arg2: tensor<4x1515x3056xf64>, %arg3: tensor<4x1x3056xf64>) -> tensor<4x1519x3056xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:4, 0:1, 0:3056] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<4x1520x3056xf64>) -> tensor<4x1x3056xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [0:4, 1518:1519, 0:3056] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<4x1520x3056xf64>) -> tensor<4x1x3056xf64>
// CHECK-NEXT:    %2 = stablehlo.concatenate %0, %arg1, %arg2, %arg3, %1, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<4x1x3056xf64>, tensor<4x1x3056xf64>, tensor<4x1515x3056xf64>, tensor<4x1x3056xf64>, tensor<4x1x3056xf64>) -> tensor<4x1519x3056xf64>
// CHECK-NEXT:    return %2 : tensor<4x1519x3056xf64>
// CHECK-NEXT:  }
