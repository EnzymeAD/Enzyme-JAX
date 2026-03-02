// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=use_multirotate_neutral_result" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL:   func.func @use_multirotate_neutral_result
// CHECK-SAME:         (%[[ARG:.+]]: tensor<4x1520x3056xf32>, %[[ARG1:.+]]: tensor<4x1520x3056xf32>
func.func @use_multirotate_neutral_result(%arg0: tensor<4x1520x3056xf32>, %arg1: tensor<4x1520x3056xf32>) -> (tensor<4x1520x3056xf32>, tensor<4x1520x3056xf32>, tensor<4x1520x3056xf32>, tensor<4x1520x3056xf32>, tensor<4x1520x3056xf32>) {
    // CHECK: stablehlo.add %[[ARG]], %[[ARG1]]
    // CHECK: stablehlo.add %[[ARG]], %[[ARG]]
    // CHECK: %[[ROTATED:.+]]:6 = "enzymexla.multi_rotate"(%[[ARG]])
    // CHECK: stablehlo.add %[[ROTATED]]#2, %[[ARG1]]
    // CHECK: stablehlo.add %[[ROTATED]]#2, %[[ROTATED]]#2
    %0 = stablehlo.add %arg0, %arg1 : tensor<4x1520x3056xf32>
    %1 = stablehlo.add %arg0, %arg0 : tensor<4x1520x3056xf32>
    %2:6 = "enzymexla.multi_rotate"(%arg0) <{dimension = 2 : i32, left_amount = 2 : i32, right_amount = 3 : i32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>, <@mesh, [{}, {"y"}, {"x"}]>, <@mesh, [{}, {"y"}, {"x"}]>, <@mesh, [{}, {"y"}, {"x"}]>, <@mesh, [{}, {"y"}, {"x"}]>, <@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<4x1520x3056xf32>) -> (tensor<4x1520x3056xf32>, tensor<4x1520x3056xf32>, tensor<4x1520x3056xf32>, tensor<4x1520x3056xf32>, tensor<4x1520x3056xf32>, tensor<4x1520x3056xf32>)
    %3 = stablehlo.add %arg0, %arg1 : tensor<4x1520x3056xf32>
    %4 = stablehlo.add %arg0, %arg0 : tensor<4x1520x3056xf32>
    return %0, %1, %2#1, %3, %4 : tensor<4x1520x3056xf32>, tensor<4x1520x3056xf32>, tensor<4x1520x3056xf32>, tensor<4x1520x3056xf32>, tensor<4x1520x3056xf32>
}
