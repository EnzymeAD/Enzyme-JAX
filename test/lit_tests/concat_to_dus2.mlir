// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=concat_to_onedim_dusslice"  --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module {
  sdy.mesh @mesh = <["x"=2, "y"=1, "z"=1]>
  func.func @main(%arg29 : tensor<1x24x96xf64>, %5718 : tensor<1x8x80xf64>) -> tensor<1x24x80xf64> {
    %12475 = stablehlo.slice %arg29 [0:1, 0:8, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>]>} : (tensor<1x24x96xf64>) -> tensor<1x8x80xf64>
    %12476 = stablehlo.slice %arg29 [0:1, 16:24, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>]>} : (tensor<1x24x96xf64>) -> tensor<1x8x80xf64>
    %12477 = stablehlo.concatenate %12475, %5718, %12476, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>]>} : (tensor<1x8x80xf64>, tensor<1x8x80xf64>, tensor<1x8x80xf64>) -> tensor<1x24x80xf64>
    return %12477 : tensor<1x24x80xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<1x24x96xf64>, %arg1: tensor<1x8x80xf64>) -> tensor<1x24x80xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 0:24, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>]>} : (tensor<1x24x96xf64>) -> tensor<1x24x80xf64>
// CHECK-NEXT:    %1 = stablehlo.dynamic_update_slice %0, %arg1, %c_0, %c, %c_0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>]>} : (tensor<1x24x80xf64>, tensor<1x8x80xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x24x80xf64>
// CHECK-NEXT:    return %1 : tensor<1x24x80xf64>
// CHECK-NEXT:  }
