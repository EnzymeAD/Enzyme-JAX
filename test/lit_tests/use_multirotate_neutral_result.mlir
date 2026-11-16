// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=use_multirotate_neutral_result" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s
// RUN: enzymexlamlir-opt %s --optimize-communication="rotate_to_pad_comm=0 rotate_spmd=1" | FileCheck %s --check-prefix=SPMD

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

// CHECK-LABEL:   func.func @preserve_independent_halo_exchanges
// CHECK-SAME:         (%[[ARG:.+]]: tensor<4x8x16xf32>)
func.func @preserve_independent_halo_exchanges(%arg0: tensor<4x8x16xf32>) -> (tensor<4x8x16xf32>, tensor<4x8x16xf32>, tensor<4x8x16xf32>) {
    // CHECK: %[[XROT:.+]]:3 = "enzymexla.multi_rotate"(%[[ARG]]) <{dimension = 2 : i32, left_amount = 1 : i32, right_amount = 1 : i32}>
    // CHECK: %[[YROT:.+]]:3 = "enzymexla.multi_rotate"(%[[ARG]]) <{dimension = 1 : i32, left_amount = 1 : i32, right_amount = 1 : i32}>
    // CHECK: stablehlo.add %[[YROT]]#1, %[[XROT]]#0
    %xrot:3 = "enzymexla.multi_rotate"(%arg0) <{dimension = 2 : i32, left_amount = 1 : i32, right_amount = 1 : i32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>, <@mesh, [{}, {"y"}, {"x"}]>, <@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<4x8x16xf32>) -> (tensor<4x8x16xf32>, tensor<4x8x16xf32>, tensor<4x8x16xf32>)
    %yrot:3 = "enzymexla.multi_rotate"(%arg0) <{dimension = 1 : i32, left_amount = 1 : i32, right_amount = 1 : i32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>, <@mesh, [{}, {"y"}, {"x"}]>, <@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<4x8x16xf32>) -> (tensor<4x8x16xf32>, tensor<4x8x16xf32>, tensor<4x8x16xf32>)
    %sum = stablehlo.add %arg0, %xrot#0 : tensor<4x8x16xf32>
    return %xrot#0, %yrot#2, %sum : tensor<4x8x16xf32>, tensor<4x8x16xf32>, tensor<4x8x16xf32>
}

// SPMD-LABEL:    func.func @preserve_independent_rotates_spmd
// SPMD-SAME:          (%[[ARG:.+]]: tensor<4x8x16xf32>)
func.func @preserve_independent_rotates_spmd(%arg0: tensor<4x8x16xf32>) -> (tensor<4x8x16xf32>, tensor<4x8x16xf32>, tensor<4x8x16xf32>) {
    // SPMD: %[[XROT:.+]]:2 = stablehlo.custom_call @_SPMDInternalOp_MultiRotate(%[[ARG]]) {backend_config = "dimension=2,left_amount=1,right_amount=0,bufferize=0"
    // SPMD: %[[YROT:.+]]:2 = stablehlo.custom_call @_SPMDInternalOp_MultiRotate(%[[ARG]]) {backend_config = "dimension=1,left_amount=0,right_amount=1,bufferize=0"
    // SPMD: stablehlo.add %[[YROT]]#0, %[[XROT]]#0
    %xrot = "enzymexla.rotate"(%arg0) <{amount = 1 : i32, dimension = 2 : i32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
    %yrot = "enzymexla.rotate"(%arg0) <{amount = 7 : i32, dimension = 1 : i32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
    %sum = stablehlo.add %arg0, %xrot : tensor<4x8x16xf32>
    return %xrot, %yrot, %sum : tensor<4x8x16xf32>, tensor<4x8x16xf32>, tensor<4x8x16xf32>
}
