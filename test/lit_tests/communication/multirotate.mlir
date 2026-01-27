// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{multirotate_spmd=1})" %s | FileCheck %s

sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=1]>

// Case 1: Left and Right Rotate (Amount 1)
func.func @multirotate_lr(%arg0: tensor<16x20xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {"y"}]>}) -> (tensor<16x20xf64>, tensor<16x20xf64>, tensor<16x20xf64>) {
    %1:3 = "enzymexla.multi_rotate"(%arg0) <{dimension = 0 : si32, left_amount = 1 : si32, right_amount = 1 : si32}> 
         {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"x"}, {"y"}]>, <@mesh1, [{"x"}, {"y"}]>, <@mesh1, [{"x"}, {"y"}]>]>} 
         : (tensor<16x20xf64>) -> (tensor<16x20xf64>, tensor<16x20xf64>, tensor<16x20xf64>)
    return %1#0, %1#1, %1#2 : tensor<16x20xf64>, tensor<16x20xf64>, tensor<16x20xf64>
}

// CHECK-LABEL:   func.func @multirotate_lr
// CHECK:           %[[MANUAL:.*]]:3 = sdy.manual_computation
// CHECK-DAG:         stablehlo.collective_permute
// CHECK-DAG:         stablehlo.collective_permute
// CHECK:             stablehlo.concatenate
// CHECK:           }
// CHECK:           return %[[MANUAL]]#0, %[[MANUAL]]#1, %[[MANUAL]]#2


// Case 2: Only Left Rotate (Amount 2)
func.func @multirotate_l(%arg0: tensor<16x20xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {"y"}]>}) -> (tensor<16x20xf64>, tensor<16x20xf64>, tensor<16x20xf64>) {
    %1:3 = "enzymexla.multi_rotate"(%arg0) <{dimension = 0 : si32, left_amount = 2 : si32, right_amount = 0 : si32}> 
         {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"x"}, {"y"}]>, <@mesh1, [{"x"}, {"y"}]>, <@mesh1, [{"x"}, {"y"}]>]>} 
         : (tensor<16x20xf64>) -> (tensor<16x20xf64>, tensor<16x20xf64>, tensor<16x20xf64>)
    return %1#0, %1#1, %1#2 : tensor<16x20xf64>, tensor<16x20xf64>, tensor<16x20xf64>
}

// CHECK-LABEL:   func.func @multirotate_l
// CHECK:           sdy.manual_computation
// CHECK:             stablehlo.collective_permute
// CHECK-NOT:         stablehlo.collective_permute
// CHECK:             stablehlo.concatenate
// CHECK:           }

// Case 3: Only Right Rotate (Amount 2)
func.func @multirotate_r(%arg0: tensor<16x20xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {"y"}]>}) -> (tensor<16x20xf64>, tensor<16x20xf64>, tensor<16x20xf64>) {
    %1:3 = "enzymexla.multi_rotate"(%arg0) <{dimension = 0 : si32, left_amount = 0 : si32, right_amount = 2 : si32}> 
         {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"x"}, {"y"}]>, <@mesh1, [{"x"}, {"y"}]>, <@mesh1, [{"x"}, {"y"}]>]>} 
         : (tensor<16x20xf64>) -> (tensor<16x20xf64>, tensor<16x20xf64>, tensor<16x20xf64>)
    return %1#0, %1#1, %1#2 : tensor<16x20xf64>, tensor<16x20xf64>, tensor<16x20xf64>
}

// CHECK-LABEL:   func.func @multirotate_r
// CHECK:           sdy.manual_computation
// CHECK:             stablehlo.collective_permute
// CHECK-NOT:         stablehlo.collective_permute
// CHECK:             stablehlo.concatenate
// CHECK:           }

// Case 4: Padding Case (17 elements, 4 shards -> 5, 4, 4, 4 elements (last shard has 3 padding))
// Shard size is 5 (ceil(17/4)).
// Full size 17.
// Start indices: 0, 5, 10, 15.
// Padding: 4*5 - 17 = 3.
// Last shard (index 3) has 2 valid elements (15, 16) and 3 padding.
// Left Neighbor of Shard 0 is Shard 3.
// Shard 0 pulls "Right" from Shard 3. Takes tail of Shard 3.
// Tail of Shard 3 is [2..5] (size 3) or similar.
// If Right Amount is 1. We want last valid element of Shard 3.
// Shard 3 has [15, 16, P, P, P] (logical indices 0, 1, 2, 3, 4).
// Last valid is index 1.
// Tail of size 1 is index 4 (P). We don't want index 4. We want index 1.
// The offset logic should subtract padding (3).
// Start index should be (5 - 1) - 3 = 1.
// Correct.

func.func @multirotate_padding(%arg0: tensor<17x20xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {"y"}]>}) -> (tensor<17x20xf64>, tensor<17x20xf64>, tensor<17x20xf64>) {
    // The input `arg0` is global 17x20.
    %1:3 = "enzymexla.multi_rotate"(%arg0) <{dimension = 0 : si32, left_amount = 1 : si32, right_amount = 1 : si32}> 
         {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"x"}, {"y"}]>, <@mesh1, [{"x"}, {"y"}]>, <@mesh1, [{"x"}, {"y"}]>]>} 
         : (tensor<17x20xf64>) -> (tensor<17x20xf64>, tensor<17x20xf64>, tensor<17x20xf64>)
    return %1#0, %1#1, %1#2 : tensor<17x20xf64>, tensor<17x20xf64>, tensor<17x20xf64>
}

// CHECK-LABEL:   func.func @multirotate_padding
// CHECK:           sdy.manual_computation
// CHECK:             stablehlo.partition_id
// CHECK:             stablehlo.convert
// CHECK:             stablehlo.compare
// CHECK:             stablehlo.select
// CHECK:             stablehlo.dynamic_slice
// CHECK:             stablehlo.collective_permute
// CHECK:             stablehlo.concatenate
// CHECK:           }

