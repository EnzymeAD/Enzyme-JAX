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

// Case 4: Padding Case (17 elements, 4 shards -> 5, 5, 5, 2+3 elements (last shard has 3 padding))

// CHECK-LABEL: func.func @multirotate_padding
// CHECK-SAME: (%[[ARG0:.*]]: tensor<17x20xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {"y"}]>})
func.func @multirotate_padding(%arg0: tensor<17x20xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {"y"}]>}) -> (tensor<17x20xf64>, tensor<17x20xf64>, tensor<17x20xf64>) {
    // Verify padding from 17 to 20 (ceil(17/4)*4 = 20, padding = 3)
    // CHECK: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    // CHECK: %[[PADDED:.*]] = stablehlo.pad %[[ARG0]], %[[ZERO]], low = [0, 0], high = [3, 0], interior = [0, 0]
    // CHECK-SAME: (tensor<17x20xf64>, tensor<f64>) -> tensor<20x20xf64>
    
    // Verify manual computation structure
    // CHECK: %[[MANUAL:.*]]:3 = sdy.manual_computation(%[[PADDED]])
    // CHECK-SAME: in_shardings=[<@mesh1, [{"x"}, {"y"}]>]
    // CHECK-SAME: out_shardings=[<@mesh1, [{"x"}, {"y"}]>, <@mesh1, [{"x"}, {"y"}]>, <@mesh1, [{"x"}, {"y"}]>]
    // CHECK-SAME: (%[[LOCAL_IN:.*]]: tensor<5x20xf64>)
    
    // Constants defined at the start of the block
    // CHECK: %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i32>
    // CHECK: %[[C3:.*]] = stablehlo.constant dense<3> : tensor<i32>
    // CHECK: %[[C4:.*]] = stablehlo.constant dense<4> : tensor<i32>
    
    // Verify left halo generation (slice head, send left)
    // CHECK: %[[LEFT_SLICE:.*]] = stablehlo.slice %[[LOCAL_IN]] [0:1, 0:20] : (tensor<5x20xf64>) -> tensor<1x20xf64>
    // CHECK: %[[LEFT_HALO:.*]] = "stablehlo.collective_permute"(%[[LEFT_SLICE]])
    // CHECK-SAME: source_target_pairs = dense<{{\[\[}}3, 0], [0, 1], [1, 2], [2, 3]]>
    // CHECK-SAME: : (tensor<1x20xf64>) -> tensor<1x20xf64>
    
    // Verify right halo with dynamic slice for padding handling
    // CHECK: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
    // CHECK: %[[PID_I32:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i32>
    // CHECK: %[[IS_LAST:.*]] = stablehlo.compare EQ, %[[PID_I32]], %[[C3]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
    // CHECK: %[[OFFSET:.*]] = stablehlo.select %[[IS_LAST]], %[[C3]], %[[C0]] : tensor<i1>, tensor<i32>
    // CHECK: %[[START_IDX:.*]] = stablehlo.subtract %[[C4]], %[[OFFSET]] : tensor<i32>
    // CHECK: %[[DYN_SLICE:.*]] = stablehlo.dynamic_slice %[[LOCAL_IN]], %[[START_IDX]], %[[C0]], sizes = [1, 20]
    // CHECK: %[[RIGHT_HALO:.*]] = "stablehlo.collective_permute"(%[[DYN_SLICE]])
    // CHECK-SAME: source_target_pairs = dense<{{\[\[}}1, 0], [2, 1], [3, 2], [0, 3]]>
    // CHECK-SAME: : (tensor<1x20xf64>) -> tensor<1x20xf64>
    
    // Verify concatenation order: [right_halo, local_input, left_halo]
    // CHECK: %[[SUPER:.*]] = stablehlo.concatenate %[[RIGHT_HALO]], %[[LOCAL_IN]], %[[LEFT_HALO]], dim = 0
    // CHECK-SAME: (tensor<1x20xf64>, tensor<5x20xf64>, tensor<1x20xf64>) -> tensor<7x20xf64>
    
    // Verify result slicing: amounts are [1, 0, -1], offsets are [1+1, 0+1, -1+1] = [2, 1, 0]
    // CHECK: %[[RES0:.*]] = stablehlo.slice %[[SUPER]] [2:7, 0:20] : (tensor<7x20xf64>) -> tensor<5x20xf64>
    // CHECK: %[[RES1:.*]] = stablehlo.slice %[[SUPER]] [1:6, 0:20] : (tensor<7x20xf64>) -> tensor<5x20xf64>
    // CHECK: %[[RES2:.*]] = stablehlo.slice %[[SUPER]] [0:5, 0:20] : (tensor<7x20xf64>) -> tensor<5x20xf64>
    // CHECK: sdy.return %[[RES0]], %[[RES1]], %[[RES2]]
    
    // Verify final slicing to remove padding (20 -> 17)
    // CHECK: } : (tensor<20x20xf64>) -> (tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>)
    // CHECK: %[[FINAL0:.*]] = stablehlo.slice %[[MANUAL]]#0 [0:17, 0:20] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"x"}, {"y"}]>]>} : (tensor<20x20xf64>) -> tensor<17x20xf64>
    // CHECK: %[[FINAL1:.*]] = stablehlo.slice %[[MANUAL]]#1 [0:17, 0:20] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"x"}, {"y"}]>]>} : (tensor<20x20xf64>) -> tensor<17x20xf64>
    // CHECK: %[[FINAL2:.*]] = stablehlo.slice %[[MANUAL]]#2 [0:17, 0:20] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"x"}, {"y"}]>]>} : (tensor<20x20xf64>) -> tensor<17x20xf64>
    // CHECK: return %[[FINAL0]], %[[FINAL1]], %[[FINAL2]]
    
    %1:3 = "enzymexla.multi_rotate"(%arg0) <{dimension = 0 : si32, left_amount = 1 : si32, right_amount = 1 : si32}> 
         {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"x"}, {"y"}]>, <@mesh1, [{"x"}, {"y"}]>, <@mesh1, [{"x"}, {"y"}]>]>} 
         : (tensor<17x20xf64>) -> (tensor<17x20xf64>, tensor<17x20xf64>, tensor<17x20xf64>)
    return %1#0, %1#1, %1#2 : tensor<17x20xf64>, tensor<17x20xf64>, tensor<17x20xf64>
}
