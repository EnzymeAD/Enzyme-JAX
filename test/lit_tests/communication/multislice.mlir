// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{multislice_custom_call=1})" %s | FileCheck %s

sdy.mesh @mesh1 = <["x"=4, "y"=1]>
sdy.mesh @mesh2 = <["x"=4, "y"=2]>

// ============================================================
// CASES WHERE DETECTION FIRES
// ============================================================

// Cross-shard slice along dim 0, amount=2
//   Tensor dim 0 = 16, sharded by x=4 → shard_size=4, boundaries at 0,4,8,12
//   Slice window along dim 0 is [2, 7) = size 5
//     Slice 0: [2, 7)  → start in shard 0, end(6) in shard 1 ✓ cross-shard
//     Slice 1: [3, 8)  → start in shard 0, end(7) in shard 1 ✓ cross-shard
//     Slice 2: [4, 9)  → start in shard 1, end(8) in shard 2 ✓ cross-shard
//   Dim 1 is unsharded (y=1), full span [0,20) satisfied trivially.
// CHECK-LABEL: @fires_cross_shard_basic
// CHECK: stablehlo.custom_call @_SPMDOp_MultiSlice(%arg0) {backend_config = "dimension=0,amount=2,start_indices=[2, 0],limit_indices=[7, 20],strides=[1, 1]", sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"x"}, {}]>, <@mesh1, [{"x"}, {}]>, <@mesh1, [{"x"}, {}]>]>}
func.func @fires_cross_shard_basic(
    %arg0: tensor<16x20xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {}]>}
) -> (tensor<5x20xf64>, tensor<5x20xf64>, tensor<5x20xf64>) {
    %0:3 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 2, 0>,
        limit_indices = array<i64: 7, 20>,
        strides = array<i64: 1, 1>,
        dimension = 0 : i32,
        amount = 2 : i32
    }> {sdy.sharding = #sdy.sharding_per_value<[
        <@mesh1, [{"x"}, {}]>, <@mesh1, [{"x"}, {}]>, <@mesh1, [{"x"}, {}]>
    ]>} : (tensor<16x20xf64>) -> (tensor<5x20xf64>, tensor<5x20xf64>, tensor<5x20xf64>)
    return %0#0, %0#1, %0#2 : tensor<5x20xf64>, tensor<5x20xf64>, tensor<5x20xf64>
}

// Cross-shard near later boundary, amount=1
//   Slice window along dim 0 is [7, 12) = size 5
//     Slice 0: [7, 12)  → start in shard 1, end(11) in shard 2 ✓ cross-shard
//     Slice 1: [8, 13)  → start in shard 2, end(12) in shard 3 ✓ cross-shard
// CHECK-LABEL: @fires_cross_shard_later_boundary
// CHECK: stablehlo.custom_call @_SPMDOp_MultiSlice(%arg0) {backend_config = "dimension=0,amount=1,start_indices=[7, 0],limit_indices=[12, 20],strides=[1, 1]", sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"x"}, {}]>, <@mesh1, [{"x"}, {}]>]>}
func.func @fires_cross_shard_later_boundary(
    %arg0: tensor<16x20xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {}]>}
) -> (tensor<5x20xf64>, tensor<5x20xf64>) {
    %0:2 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 7, 0>,
        limit_indices = array<i64: 12, 20>,
        strides = array<i64: 1, 1>,
        dimension = 0 : i32,
        amount = 1 : i32
    }> {sdy.sharding = #sdy.sharding_per_value<[
        <@mesh1, [{"x"}, {}]>, <@mesh1, [{"x"}, {}]>
    ]>} : (tensor<16x20xf64>) -> (tensor<5x20xf64>, tensor<5x20xf64>)
    return %0#0, %0#1 : tensor<5x20xf64>, tensor<5x20xf64>
}

// Two sharded dims, slice along dim 0, dim 1 spans full extent
//   mesh2 has x=4, y=2. Tensor is 16x20, sharded [{"x"},{"y"}].
//   Dim 0: shard_size=4, dim 1: shard_size=10.
//   Dim 1 start/limit = [0, 20) → full span ✓
//   Slice window along dim 0 is [3, 8) = size 5
//     Slice 0: [3, 8)  → shard 0 to shard 1 ✓
//     Slice 1: [4, 9)  → shard 1 to shard 2 ✓
// CHECK-LABEL: @fires_two_sharded_dims_full_span
// CHECK: stablehlo.custom_call @_SPMDOp_MultiSlice(%arg0) {backend_config = "dimension=0,amount=1,start_indices=[3, 0],limit_indices=[8, 20],strides=[1, 1]", sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"x"}, {}]>, <@mesh2, [{"x"}, {}]>]>}
func.func @fires_two_sharded_dims_full_span(
    %arg0: tensor<16x20xf64> {sdy.sharding = #sdy.sharding<@mesh2, [{"x"}, {"y"}]>}
) -> (tensor<5x20xf64>, tensor<5x20xf64>) {
    %0:2 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 3, 0>,
        limit_indices = array<i64: 8, 20>,
        strides = array<i64: 1, 1>,
        dimension = 0 : i32,
        amount = 1 : i32
    }> {sdy.sharding = #sdy.sharding_per_value<[
        <@mesh2, [{"x"}, {}]>, <@mesh2, [{"x"}, {}]>
    ]>} : (tensor<16x20xf64>) -> (tensor<5x20xf64>, tensor<5x20xf64>)
    return %0#0, %0#1 : tensor<5x20xf64>, tensor<5x20xf64>
}

// Cross-shard with indivisible dimension, amount=1
//   Tensor dim 0 = 14, sharded by x=4 → shard_size=ceil(14/4)=4, last shard size=2
//   Shard boundaries at 0, 4, 8, 12 (last shard covers [12,14))
//   Slice window along dim 0 is [9, 13) = size 4
//     Slice 0: [9, 13)  → start in shard 2, end(12) in shard 3 ✓ cross-shard
//     Slice 1: [10, 14) → start in shard 2, end(13) in shard 3 ✓ cross-shard
// CHECK-LABEL: @fires_cross_shard_indivisible
// CHECK: stablehlo.custom_call @_SPMDOp_MultiSlice(%arg0) {backend_config = "dimension=0,amount=1,start_indices=[9, 0],limit_indices=[13, 20],strides=[1, 1]", sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"x"}, {}]>, <@mesh1, [{"x"}, {}]>]>}
func.func @fires_cross_shard_indivisible(
    %arg0: tensor<14x20xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {}]>}
) -> (tensor<4x20xf64>, tensor<4x20xf64>) {
    %0:2 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 9, 0>,
        limit_indices = array<i64: 13, 20>,
        strides = array<i64: 1, 1>,
        dimension = 0 : i32,
        amount = 1 : i32
    }> {sdy.sharding = #sdy.sharding_per_value<[
        <@mesh1, [{"x"}, {}]>, <@mesh1, [{"x"}, {}]>
    ]>} : (tensor<14x20xf64>) -> (tensor<4x20xf64>, tensor<4x20xf64>)
    return %0#0, %0#1 : tensor<4x20xf64>, tensor<4x20xf64>
}

// ============================================================
// CASES WHERE DETECTION DOES NOT FIRE
// ============================================================

// Slice stays within a single shard (no boundary crossing)
//   Slice window along dim 0 is [0, 2) = size 2
//     Slice 0: [0, 2)  → shard 0 to shard 0 ✗ same shard
//     Slice 1: [1, 3)  → shard 0 to shard 0 ✗ same shard
// CHECK-LABEL: @no_fire_within_single_shard
// CHECK-NOT: stablehlo.custom_call @_SPMDOp_MultiSlice
// CHECK: enzymexla.multi_slice
func.func @no_fire_within_single_shard(
    %arg0: tensor<16x20xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {}]>}
) -> (tensor<2x20xf64>, tensor<2x20xf64>) {
    %0:2 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 0, 0>,
        limit_indices = array<i64: 2, 20>,
        strides = array<i64: 1, 1>,
        dimension = 0 : i32,
        amount = 1 : i32
    }> {sdy.sharding = #sdy.sharding_per_value<[
        <@mesh1, [{"x"}, {}]>, <@mesh1, [{"x"}, {}]>
    ]>} : (tensor<16x20xf64>) -> (tensor<2x20xf64>, tensor<2x20xf64>)
    return %0#0, %0#1 : tensor<2x20xf64>, tensor<2x20xf64>
}

// Non-unit strides (condition 1 fails)
// CHECK-LABEL: @no_fire_non_unit_strides
// CHECK-NOT: stablehlo.custom_call @_SPMDOp_MultiSlice
// CHECK: enzymexla.multi_slice
func.func @no_fire_non_unit_strides(
    %arg0: tensor<16x20xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {}]>}
) -> (tensor<3x20xf64>, tensor<3x20xf64>) {
    %0:2 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 2, 0>,
        limit_indices = array<i64: 8, 20>,
        strides = array<i64: 2, 1>,
        dimension = 0 : i32,
        amount = 1 : i32
    }> {sdy.sharding = #sdy.sharding_per_value<[
        <@mesh1, [{"x"}, {}]>, <@mesh1, [{"x"}, {}]>
    ]>} : (tensor<16x20xf64>) -> (tensor<3x20xf64>, tensor<3x20xf64>)
    return %0#0, %0#1 : tensor<3x20xf64>, tensor<3x20xf64>
}

// Other sharded dim doesn't span full extent (condition 2 fails)
//   mesh2 has x=4, y=2. Dim 1 is sharded with shard_size=10.
//   Dim 1 range [5, 15) does NOT span [0, 20) → fails condition 2.
// CHECK-LABEL: @no_fire_partial_span_other_sharded_dim
// CHECK-NOT: stablehlo.custom_call @_SPMDOp_MultiSlice
// CHECK: enzymexla.multi_slice
func.func @no_fire_partial_span_other_sharded_dim(
    %arg0: tensor<16x20xf64> {sdy.sharding = #sdy.sharding<@mesh2, [{"x"}, {"y"}]>}
) -> (tensor<5x10xf64>, tensor<5x10xf64>) {
    %0:2 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 3, 5>,
        limit_indices = array<i64: 8, 15>,
        strides = array<i64: 1, 1>,
        dimension = 0 : i32,
        amount = 1 : i32
    }> {sdy.sharding = #sdy.sharding_per_value<[
        <@mesh2, [{"x"}, {}]>, <@mesh2, [{"x"}, {}]>
    ]>} : (tensor<16x20xf64>) -> (tensor<5x10xf64>, tensor<5x10xf64>)
    return %0#0, %0#1 : tensor<5x10xf64>, tensor<5x10xf64>
}

// Not sharded along slice dimension (condition 3: numShards <= 1)
//   Slicing along dim 1, which has y=1 → not sharded on the operand.
//   Result shardings use mesh2 with y=2 on dim 1 so numDevicesAlongDimension > 1,
//   but the operand is not sharded along dim 1, so detectCrossShardPattern fails.
// CHECK-LABEL: @no_fire_unsharded_slice_dim
// CHECK-NOT: stablehlo.custom_call @_SPMDOp_MultiSlice
// CHECK: enzymexla.multi_slice
func.func @no_fire_unsharded_slice_dim(
    %arg0: tensor<16x20xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {}]>}
) -> (tensor<16x5xf64>, tensor<16x5xf64>) {
    %0:2 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 0, 3>,
        limit_indices = array<i64: 16, 8>,
        strides = array<i64: 1, 1>,
        dimension = 1 : i32,
        amount = 1 : i32
    }> {sdy.sharding = #sdy.sharding_per_value<[
        <@mesh2, [{}, {"y"}]>, <@mesh2, [{}, {"y"}]>
    ]>} : (tensor<16x20xf64>) -> (tensor<16x5xf64>, tensor<16x5xf64>)
    return %0#0, %0#1 : tensor<16x5xf64>, tensor<16x5xf64>
}

// Mixed — some slices cross, some don't (condition 3 partial fail)
//   Slice window along dim 0 is [1, 5) = size 4
//     Slice 0: [1, 5)  → start shard 0, end(4) shard 1 ✓ cross-shard
//     Slice 1: [2, 6)  → start shard 0, end(5) shard 1 ✓ cross-shard
//     Slice 2: [3, 7)  → start shard 0, end(6) shard 1 ✓ cross-shard
//     Slice 3: [4, 8)  → start shard 1, end(7) shard 1 ✗ same shard!
// CHECK-LABEL: @no_fire_mixed_cross_and_local
// CHECK-NOT: stablehlo.custom_call @_SPMDOp_MultiSlice
// CHECK: enzymexla.multi_slice
func.func @no_fire_mixed_cross_and_local(
    %arg0: tensor<16x20xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {}]>}
) -> (tensor<4x20xf64>, tensor<4x20xf64>, tensor<4x20xf64>, tensor<4x20xf64>) {
    %0:4 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 1, 0>,
        limit_indices = array<i64: 5, 20>,
        strides = array<i64: 1, 1>,
        dimension = 0 : i32,
        amount = 3 : i32
    }> {sdy.sharding = #sdy.sharding_per_value<[
        <@mesh1, [{"x"}, {}]>, <@mesh1, [{"x"}, {}]>,
        <@mesh1, [{"x"}, {}]>, <@mesh1, [{"x"}, {}]>
    ]>} : (tensor<16x20xf64>) -> (tensor<4x20xf64>, tensor<4x20xf64>, tensor<4x20xf64>, tensor<4x20xf64>)
    return %0#0, %0#1, %0#2, %0#3 : tensor<4x20xf64>, tensor<4x20xf64>, tensor<4x20xf64>, tensor<4x20xf64>
}
