// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{dus_of_concat_slices=1 dus_to_pad_comm=0 dus_to_pad_manual_comp_comm=0})" %s | FileCheck %s

sdy.mesh @mesh1 = <["x"=4, "y"=4]>

func.func @dus_of_concat_slices(%arg0: tensor<20x128x128xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{}, {"x"}, {"y"}]>}) -> tensor<20x128x128xf32> {
    %c7 = stablehlo.constant dense<7> : tensor<i32>
    %c8 = stablehlo.constant dense<8> : tensor<i32>
    
    %slice1 = stablehlo.slice %arg0 [8:9, 8:128, 8:128] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{}, {"x"}, {"y"}]>]>} : (tensor<20x128x128xf32>) -> tensor<1x120x120xf32>
    %slice2 = stablehlo.slice %arg0 [8:12, 8:128, 8:128] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{}, {"x"}, {"y"}]>]>} : (tensor<20x128x128xf32>) -> tensor<4x120x120xf32>
    %slice3 = stablehlo.slice %arg0 [11:12, 8:128, 8:128] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{}, {"x"}, {"y"}]>]>} : (tensor<20x128x128xf32>) -> tensor<1x120x120xf32>
    
    %concat = stablehlo.concatenate %slice1, %slice2, %slice3, dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{}, {"x"}, {"y"}]>]>} : (tensor<1x120x120xf32>, tensor<4x120x120xf32>, tensor<1x120x120xf32>) -> tensor<6x120x120xf32>
    
    %dus = stablehlo.dynamic_update_slice %arg0, %concat, %c7, %c8, %c8 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{}, {"x"}, {"y"}]>]>} : (tensor<20x128x128xf32>, tensor<6x120x120xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x128x128xf32>
    
    return %dus : tensor<20x128x128xf32>
}

// CHECK-LABEL: func.func @dus_of_concat_slices
// CHECK-SAME: (%[[ARG0:.*]]: tensor<20x128x128xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{}, {"x"}, {"y"}]>}) -> tensor<20x128x128xf32> {
// CHECK-DAG:  %[[C7:.*]] = stablehlo.constant dense<7> : tensor<i32>
// CHECK-DAG:  %[[C8:.*]] = stablehlo.constant dense<8> : tensor<i32>
// CHECK-DAG:  %[[C12:.*]] = stablehlo.constant dense<12> : tensor<i32>
// CHECK-DAG:  %[[SLICE1:.*]] = stablehlo.slice %[[ARG0]] [8:9, 8:128, 8:128] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{}, {"x"}, {"y"}]>]>} : (tensor<20x128x128xf32>) -> tensor<1x120x120xf32>
// CHECK-DAG:  %[[SLICE3:.*]] = stablehlo.slice %[[ARG0]] [11:12, 8:128, 8:128] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{}, {"x"}, {"y"}]>]>} : (tensor<20x128x128xf32>) -> tensor<1x120x120xf32>
// CHECK:      %[[DUS1:.*]] = stablehlo.dynamic_update_slice %[[ARG0]], %[[SLICE1]], %[[C7]], %[[C8]], %[[C8]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{}, {"x"}, {"y"}]>]>} : (tensor<20x128x128xf32>, tensor<1x120x120xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x128x128xf32>
// CHECK:      %[[DUS2:.*]] = stablehlo.dynamic_update_slice %[[DUS1]], %[[SLICE3]], %[[C12]], %[[C8]], %[[C8]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{}, {"x"}, {"y"}]>]>} : (tensor<20x128x128xf32>, tensor<1x120x120xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x128x128xf32>
// CHECK:      return %[[DUS2]] : tensor<20x128x128xf32>
