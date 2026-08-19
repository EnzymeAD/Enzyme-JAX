// RUN: enzymexlamlir-opt %s --enzyme-batch | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-batch | stablehlo-translate - --interpret --allow-unregistered-dialect

// Numeric check that batching a dynamic_slice with per-batch-varying start
// indices preserves dynamic_slice semantics, including the clamping of
// out-of-bounds start indices (gather clamps the same way, so no explicit
// clamp is emitted).

func.func @batched(%operand: tensor<3x2x5xf64>, %i: tensor<3xi32>, %j: tensor<3xi32>) -> tensor<3x1x1xf64> {
    %0 = enzyme.batch @ds(%operand, %i, %j) {batch_shape = array<i64: 3>} : (tensor<3x2x5xf64>, tensor<3xi32>, tensor<3xi32>) -> tensor<3x1x1xf64>
    return %0 : tensor<3x1x1xf64>
}

func.func @ds(%operand: tensor<2x5xf64>, %i: tensor<i32>, %j: tensor<i32>) -> tensor<1x1xf64> {
    %0 = stablehlo.dynamic_slice %operand, %i, %j, sizes = [1, 1] : (tensor<2x5xf64>, tensor<i32>, tensor<i32>) -> tensor<1x1xf64>
    return %0 : tensor<1x1xf64>
}

func.func @main() {
    %operand = stablehlo.constant dense<[
      [[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]],
      [[10.0, 11.0, 12.0, 13.0, 14.0], [15.0, 16.0, 17.0, 18.0, 19.0]],
      [[20.0, 21.0, 22.0, 23.0, 24.0], [25.0, 26.0, 27.0, 28.0, 29.0]]
    ]> : tensor<3x2x5xf64>
    // Row index 7 for batch element 2 is out of bounds and must clamp to 1.
    %i = stablehlo.constant dense<[0, 1, 7]> : tensor<3xi32>
    %j = stablehlo.constant dense<[0, 3, 2]> : tensor<3xi32>

    %res = func.call @batched(%operand, %i, %j) : (tensor<3x2x5xf64>, tensor<3xi32>, tensor<3xi32>) -> tensor<3x1x1xf64>

    // batch 0 -> operand[0][0][0] = 0, batch 1 -> operand[1][1][3] = 18,
    // batch 2 -> operand[2][clamp(7)=1][2] = 27
    %expected = stablehlo.constant dense<[[[0.0]], [[18.0]], [[27.0]]]> : tensor<3x1x1xf64>
    "check.expect_eq"(%res, %expected) : (tensor<3x1x1xf64>, tensor<3x1x1xf64>) -> ()

    // Cross-check each batch element against a plain, unbatched dynamic_slice.
    %s0 = stablehlo.slice %operand [0:1, 0:2, 0:5] : (tensor<3x2x5xf64>) -> tensor<1x2x5xf64>
    %s0r = stablehlo.reshape %s0 : (tensor<1x2x5xf64>) -> tensor<2x5xf64>
    %i0 = stablehlo.constant dense<0> : tensor<i32>
    %j0 = stablehlo.constant dense<0> : tensor<i32>
    %ref0 = func.call @ds(%s0r, %i0, %j0) : (tensor<2x5xf64>, tensor<i32>, tensor<i32>) -> tensor<1x1xf64>
    %got0 = stablehlo.slice %res [0:1, 0:1, 0:1] : (tensor<3x1x1xf64>) -> tensor<1x1x1xf64>
    %got0r = stablehlo.reshape %got0 : (tensor<1x1x1xf64>) -> tensor<1x1xf64>
    "check.expect_eq"(%got0r, %ref0) : (tensor<1x1xf64>, tensor<1x1xf64>) -> ()

    %s1 = stablehlo.slice %operand [1:2, 0:2, 0:5] : (tensor<3x2x5xf64>) -> tensor<1x2x5xf64>
    %s1r = stablehlo.reshape %s1 : (tensor<1x2x5xf64>) -> tensor<2x5xf64>
    %i1 = stablehlo.constant dense<1> : tensor<i32>
    %j1 = stablehlo.constant dense<3> : tensor<i32>
    %ref1 = func.call @ds(%s1r, %i1, %j1) : (tensor<2x5xf64>, tensor<i32>, tensor<i32>) -> tensor<1x1xf64>
    %got1 = stablehlo.slice %res [1:2, 0:1, 0:1] : (tensor<3x1x1xf64>) -> tensor<1x1x1xf64>
    %got1r = stablehlo.reshape %got1 : (tensor<1x1x1xf64>) -> tensor<1x1xf64>
    "check.expect_eq"(%got1r, %ref1) : (tensor<1x1xf64>, tensor<1x1xf64>) -> ()

    %s2 = stablehlo.slice %operand [2:3, 0:2, 0:5] : (tensor<3x2x5xf64>) -> tensor<1x2x5xf64>
    %s2r = stablehlo.reshape %s2 : (tensor<1x2x5xf64>) -> tensor<2x5xf64>
    %i2 = stablehlo.constant dense<7> : tensor<i32>
    %j2 = stablehlo.constant dense<2> : tensor<i32>
    %ref2 = func.call @ds(%s2r, %i2, %j2) : (tensor<2x5xf64>, tensor<i32>, tensor<i32>) -> tensor<1x1xf64>
    %got2 = stablehlo.slice %res [2:3, 0:1, 0:1] : (tensor<3x1x1xf64>) -> tensor<1x1x1xf64>
    %got2r = stablehlo.reshape %got2 : (tensor<1x1x1xf64>) -> tensor<1x1xf64>
    "check.expect_eq"(%got2r, %ref2) : (tensor<1x1xf64>, tensor<1x1xf64>) -> ()

    return
}

// CHECK: func.func private @batched_ds(%arg0: tensor<3x2x5xf64>, %arg1: tensor<3xi32>, %arg2: tensor<3xi32>) -> tensor<3x1x1xf64> {
// No clamp here: gather clamps its start indices exactly like dynamic_slice.
// CHECK-NEXT:    %[[R0:.+]] = stablehlo.reshape %arg1 : (tensor<3xi32>) -> tensor<3x1xi32>
// CHECK-NEXT:    %[[R1:.+]] = stablehlo.reshape %arg2 : (tensor<3xi32>) -> tensor<3x1xi32>
// CHECK-NEXT:    %[[IDX:.+]] = stablehlo.concatenate %[[R0]], %[[R1]], dim = 1 : (tensor<3x1xi32>, tensor<3x1xi32>) -> tensor<3x2xi32>
// CHECK-NEXT:    %[[RES:.+]] = "stablehlo.gather"(%arg0, %[[IDX]]) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1, 2], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<3x2x5xf64>, tensor<3x2xi32>) -> tensor<3x1x1xf64>
// CHECK-NEXT:    return %[[RES]] : tensor<3x1x1xf64>
// CHECK-NEXT: }
