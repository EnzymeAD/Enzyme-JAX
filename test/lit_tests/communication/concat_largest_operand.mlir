// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{concat_largest_operand=1})" %s | FileCheck %s

sdy.mesh @mesh1 = <["x"=4]>

// Test: Concat with multiple operands where one is much larger
// Example from issue: concat of multiple slices with different sizes
func.func @test_largest_operand(%arg0: tensor<20x6128x12272xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {}, {}]>},
                                %arg1: tensor<20x6128x1xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {}, {}]>},
                                %arg2: tensor<20x6128x1xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {}, {}]>}) 
                                -> (tensor<20x6128x12274xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {}, {}]>}) {
    // Pattern: concat(small1, large, small2) where large >> small1 + small2
    %0 = stablehlo.concatenate %arg1, %arg0, %arg2, dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"x"}, {}, {}]>]>} : 
        (tensor<20x6128x1xf32>, tensor<20x6128x12272xf32>, tensor<20x6128x1xf32>) -> tensor<20x6128x12274xf32>
    return %0 : tensor<20x6128x12274xf32>
}

// CHECK-LABEL: func.func @test_largest_operand
// CHECK: stablehlo.concatenate
// TODO: Once implemented, should pad arg0 and DUS arg1 and arg2 into it

// Test 2: Concat with 5 operands as shown in issue
func.func @test_five_operands(%arg0: tensor<4x1x12272xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{}, {"x"}, {}]>},
                               %arg1: tensor<4x1x12272xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{}, {"x"}, {}]>},
                               %arg2: tensor<4x6124x12272xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{}, {"x"}, {}]>},
                               %arg3: tensor<4x1x12272xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{}, {"x"}, {}]>},
                               %arg4: tensor<4x1x12272xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{}, {"x"}, {}]>})
                               -> (tensor<4x6128x12272xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{}, {"x"}, {}]>}) {
    // Pattern from issue: concat of 5 operands where middle one (arg2) is largest
    %0 = stablehlo.concatenate %arg0, %arg1, %arg2, %arg3, %arg4, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{}, {"x"}, {}]>]>} : 
        (tensor<4x1x12272xf32>, tensor<4x1x12272xf32>, tensor<4x6124x12272xf32>, tensor<4x1x12272xf32>, tensor<4x1x12272xf32>) -> tensor<4x6128x12272xf32>
    return %0 : tensor<4x6128x12272xf32>
}

// CHECK-LABEL: func.func @test_five_operands
// CHECK: stablehlo.concatenate
// TODO: Once implemented, should pad arg2 to 6128 and DUS the 4 small operands into it
