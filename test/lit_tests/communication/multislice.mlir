// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{multislice_spmd=1})" %s | FileCheck %s

sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=1]>

// Case 1: Left and Right Slice (Amount 1 each)
// CHECK-LABEL: func.func @multislice_lr
func.func @multislice_lr(%arg0: tensor<16x20xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {"y"}]>}) -> (tensor<4x20xf64>, tensor<4x20xf64>, tensor<4x20xf64>) {
    %0, %1, %2 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 4, 0>,
        limit_indices = array<i64: 8, 20>,
        strides = array<i64: 1, 1>,
        dimension = 0 : si32,
        left_amount = 1 : si32,
        right_amount = 1 : si32
    }> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"x"}, {"y"}]>, <@mesh1, [{"x"}, {"y"}]>, <@mesh1, [{"x"}, {"y"}]>]>}
       : (tensor<16x20xf64>) -> (tensor<4x20xf64>, tensor<4x20xf64>, tensor<4x20xf64>)
    return %0, %1, %2 : tensor<4x20xf64>, tensor<4x20xf64>, tensor<4x20xf64>
}

// Case 2: Only Left Slice (Amount 2)
// CHECK-LABEL: func.func @multislice_l
func.func @multislice_l(%arg0: tensor<16x20xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {"y"}]>}) -> (tensor<4x20xf64>, tensor<4x20xf64>, tensor<4x20xf64>) {
    %0, %1, %2 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 4, 0>,
        limit_indices = array<i64: 8, 20>,
        strides = array<i64: 1, 1>,
        dimension = 0 : si32,
        left_amount = 2 : si32,
        right_amount = 0 : si32
    }> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"x"}, {"y"}]>, <@mesh1, [{"x"}, {"y"}]>, <@mesh1, [{"x"}, {"y"}]>]>}
       : (tensor<16x20xf64>) -> (tensor<4x20xf64>, tensor<4x20xf64>, tensor<4x20xf64>)
    return %0, %1, %2 : tensor<4x20xf64>, tensor<4x20xf64>, tensor<4x20xf64>
}

// Case 3: Only Right Slice (Amount 2)
// CHECK-LABEL: func.func @multislice_r
func.func @multislice_r(%arg0: tensor<16x20xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {"y"}]>}) -> (tensor<4x20xf64>, tensor<4x20xf64>, tensor<4x20xf64>) {
    %0, %1, %2 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 4, 0>,
        limit_indices = array<i64: 8, 20>,
        strides = array<i64: 1, 1>,
        dimension = 0 : si32,
        left_amount = 0 : si32,
        right_amount = 2 : si32
    }> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"x"}, {"y"}]>, <@mesh1, [{"x"}, {"y"}]>, <@mesh1, [{"x"}, {"y"}]>]>}
       : (tensor<16x20xf64>) -> (tensor<4x20xf64>, tensor<4x20xf64>, tensor<4x20xf64>)
    return %0, %1, %2 : tensor<4x20xf64>, tensor<4x20xf64>, tensor<4x20xf64>
}

// Case 4: Padding Case (17 elements, 4 shards)
// CHECK-LABEL: func.func @multislice_padding
func.func @multislice_padding(%arg0: tensor<17x20xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {"y"}]>}) -> (tensor<5x20xf64>, tensor<5x20xf64>, tensor<5x20xf64>) {
    %0, %1, %2 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 5, 0>,
        limit_indices = array<i64: 10, 20>,
        strides = array<i64: 1, 1>,
        dimension = 0 : si32,
        left_amount = 1 : si32,
        right_amount = 1 : si32
    }> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"x"}, {"y"}]>, <@mesh1, [{"x"}, {"y"}]>, <@mesh1, [{"x"}, {"y"}]>]>}
       : (tensor<17x20xf64>) -> (tensor<5x20xf64>, tensor<5x20xf64>, tensor<5x20xf64>)
    return %0, %1, %2 : tensor<5x20xf64>, tensor<5x20xf64>, tensor<5x20xf64>
}

// Case 5: Non-zero start indices on non-sharded dimension
// CHECK-LABEL: func.func @multislice_nonzero_start
func.func @multislice_nonzero_start(%arg0: tensor<16x20xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {"y"}]>}) -> (tensor<4x10xf64>, tensor<4x10xf64>, tensor<4x10xf64>) {
    %0, %1, %2 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 4, 5>,
        limit_indices = array<i64: 8, 15>,
        strides = array<i64: 1, 1>,
        dimension = 0 : si32,
        left_amount = 1 : si32,
        right_amount = 1 : si32
    }> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"x"}, {"y"}]>, <@mesh1, [{"x"}, {"y"}]>, <@mesh1, [{"x"}, {"y"}]>]>}
       : (tensor<16x20xf64>) -> (tensor<4x10xf64>, tensor<4x10xf64>, tensor<4x10xf64>)
    return %0, %1, %2 : tensor<4x10xf64>, tensor<4x10xf64>, tensor<4x10xf64>
}
