// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{concat_slice_optimize=1})" %s | FileCheck %s

sdy.mesh @mesh1 = <["x"=4]>

// Test 1: Rotation pattern - concat(slice[end], slice[start])
// Pattern: concat(A[12271:12272], A[0:12272])
func.func @test_rotation(%arg0: tensor<20x6144x12272xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {}, {}]>}) -> (tensor<20x6144x12273xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {}, {}]>}) {
    %0 = stablehlo.slice %arg0 [0:20, 0:6144, 12271:12272] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"x"}, {}, {}]>]>} : (tensor<20x6144x12272xf32>) -> tensor<20x6144x1xf32>
    %1 = stablehlo.slice %arg0 [0:20, 0:6144, 0:12272] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"x"}, {}, {}]>]>} : (tensor<20x6144x12272xf32>) -> tensor<20x6144x12272xf32>
    %2 = stablehlo.concatenate %0, %1, dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"x"}, {}, {}]>]>} : (tensor<20x6144x1xf32>, tensor<20x6144x12272xf32>) -> tensor<20x6144x12273xf32>
    return %2 : tensor<20x6144x12273xf32>
}

// CHECK-LABEL: func.func @test_rotation
// CHECK: stablehlo.slice
// CHECK: stablehlo.slice
// CHECK: stablehlo.concatenate
// TODO: Once implemented, should transform into rotate operation

// Test 2: Asymmetric concat - one slice much smaller than other
func.func @test_asymmetric(%arg0: tensor<20x6144x12272xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {}, {}]>}) -> (tensor<20x6144x12300xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {}, {}]>}) {
    %0 = stablehlo.slice %arg0 [0:20, 0:6144, 0:10] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"x"}, {}, {}]>]>} : (tensor<20x6144x12272xf32>) -> tensor<20x6144x10xf32>
    %1 = stablehlo.slice %arg0 [0:20, 0:6144, 10:12300] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"x"}, {}, {}]>]>} : (tensor<20x6144x12272xf32>) -> tensor<20x6144x12262xf32>
    // Note: This won't match the pattern because second slice extends beyond source
    %2 = stablehlo.concatenate %0, %1, dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"x"}, {}, {}]>]>} : (tensor<20x6144x10xf32>, tensor<20x6144x12262xf32>) -> tensor<20x6144x12272xf32>
    return %2 : tensor<20x6144x12272xf32>
}

// CHECK-LABEL: func.func @test_asymmetric  
// CHECK: stablehlo.concatenate
// Pattern detection only - no transformation yet
