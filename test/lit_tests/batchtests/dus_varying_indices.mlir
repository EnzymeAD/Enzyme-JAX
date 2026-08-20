// RUN: enzymexlamlir-opt %s --enzyme-batch | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-batch | stablehlo-translate - --interpret --allow-unregistered-dialect

// Numeric check that batching a dynamic_update_slice with per-batch-varying
// start indices preserves dynamic_update_slice semantics. Batch element 2 uses
// an out-of-bounds row index: dynamic_update_slice clamps it, while scatter
// would drop the update, so the lowering has to materialize the clamp.

func.func @batched(%operand: tensor<3x2x5xf64>, %update: tensor<3x1x1xf64>, %i: tensor<3xi32>, %j: tensor<3xi32>) -> tensor<3x2x5xf64> {
    %0 = enzyme.batch @dus(%operand, %update, %i, %j) {batch_shape = array<i64: 3>} : (tensor<3x2x5xf64>, tensor<3x1x1xf64>, tensor<3xi32>, tensor<3xi32>) -> tensor<3x2x5xf64>
    return %0 : tensor<3x2x5xf64>
}

func.func @dus(%operand: tensor<2x5xf64>, %update: tensor<1x1xf64>, %i: tensor<i32>, %j: tensor<i32>) -> tensor<2x5xf64> {
    %0 = stablehlo.dynamic_update_slice %operand, %update, %i, %j : (tensor<2x5xf64>, tensor<1x1xf64>, tensor<i32>, tensor<i32>) -> tensor<2x5xf64>
    return %0 : tensor<2x5xf64>
}

func.func @main() {
    %operand = stablehlo.constant dense<0.0> : tensor<3x2x5xf64>
    %update = stablehlo.constant dense<[[[1.0]], [[2.0]], [[3.0]]]> : tensor<3x1x1xf64>
    // Row index 7 for batch element 2 is out of bounds and must clamp to 1.
    %i = stablehlo.constant dense<[0, 1, 7]> : tensor<3xi32>
    %j = stablehlo.constant dense<[0, 3, 2]> : tensor<3xi32>

    %res = func.call @batched(%operand, %update, %i, %j) : (tensor<3x2x5xf64>, tensor<3x1x1xf64>, tensor<3xi32>, tensor<3xi32>) -> tensor<3x2x5xf64>

    %expected = stablehlo.constant dense<[
      [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
      [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0, 0.0]],
      [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0, 0.0]]
    ]> : tensor<3x2x5xf64>
    "check.expect_eq"(%res, %expected) : (tensor<3x2x5xf64>, tensor<3x2x5xf64>) -> ()

    // Cross-check each batch element against a plain, unbatched
    // dynamic_update_slice evaluated by the interpreter.
    %zero2x5 = stablehlo.constant dense<0.0> : tensor<2x5xf64>
    %u0 = stablehlo.constant dense<1.0> : tensor<1x1xf64>
    %i0 = stablehlo.constant dense<0> : tensor<i32>
    %j0 = stablehlo.constant dense<0> : tensor<i32>
    %ref0 = func.call @dus(%zero2x5, %u0, %i0, %j0) : (tensor<2x5xf64>, tensor<1x1xf64>, tensor<i32>, tensor<i32>) -> tensor<2x5xf64>
    %got0 = stablehlo.slice %res [0:1, 0:2, 0:5] : (tensor<3x2x5xf64>) -> tensor<1x2x5xf64>
    %got0r = stablehlo.reshape %got0 : (tensor<1x2x5xf64>) -> tensor<2x5xf64>
    "check.expect_eq"(%got0r, %ref0) : (tensor<2x5xf64>, tensor<2x5xf64>) -> ()

    %u1 = stablehlo.constant dense<2.0> : tensor<1x1xf64>
    %i1 = stablehlo.constant dense<1> : tensor<i32>
    %j1 = stablehlo.constant dense<3> : tensor<i32>
    %ref1 = func.call @dus(%zero2x5, %u1, %i1, %j1) : (tensor<2x5xf64>, tensor<1x1xf64>, tensor<i32>, tensor<i32>) -> tensor<2x5xf64>
    %got1 = stablehlo.slice %res [1:2, 0:2, 0:5] : (tensor<3x2x5xf64>) -> tensor<1x2x5xf64>
    %got1r = stablehlo.reshape %got1 : (tensor<1x2x5xf64>) -> tensor<2x5xf64>
    "check.expect_eq"(%got1r, %ref1) : (tensor<2x5xf64>, tensor<2x5xf64>) -> ()

    // Out-of-bounds start index: the unbatched op clamps, so the batched one must too.
    %u2 = stablehlo.constant dense<3.0> : tensor<1x1xf64>
    %i2 = stablehlo.constant dense<7> : tensor<i32>
    %j2 = stablehlo.constant dense<2> : tensor<i32>
    %ref2 = func.call @dus(%zero2x5, %u2, %i2, %j2) : (tensor<2x5xf64>, tensor<1x1xf64>, tensor<i32>, tensor<i32>) -> tensor<2x5xf64>
    %got2 = stablehlo.slice %res [2:3, 0:2, 0:5] : (tensor<3x2x5xf64>) -> tensor<1x2x5xf64>
    %got2r = stablehlo.reshape %got2 : (tensor<1x2x5xf64>) -> tensor<2x5xf64>
    "check.expect_eq"(%got2r, %ref2) : (tensor<2x5xf64>, tensor<2x5xf64>) -> ()

    return
}

// CHECK: func.func private @batched_dus(%arg0: tensor<3x2x5xf64>, %arg1: tensor<3x1x1xf64>, %arg2: tensor<3xi32>, %arg3: tensor<3xi32>) -> tensor<3x2x5xf64> {
// The dynamic_update_slice clamp has to be materialized: scatter drops
// out-of-bounds updates rather than clamping them.
// CHECK-NEXT:    %[[LO0:.+]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %[[HI0:.+]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:    %[[C0:.+]] = stablehlo.clamp %[[LO0]], %arg2, %[[HI0]] : (tensor<i32>, tensor<3xi32>, tensor<i32>) -> tensor<3xi32>
// CHECK-NEXT:    %[[R0:.+]] = stablehlo.reshape %[[C0]] : (tensor<3xi32>) -> tensor<3x1xi32>
// CHECK-NEXT:    %[[LO1:.+]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %[[HI1:.+]] = stablehlo.constant dense<4> : tensor<i32>
// CHECK-NEXT:    %[[C1:.+]] = stablehlo.clamp %[[LO1]], %arg3, %[[HI1]] : (tensor<i32>, tensor<3xi32>, tensor<i32>) -> tensor<3xi32>
// CHECK-NEXT:    %[[R1:.+]] = stablehlo.reshape %[[C1]] : (tensor<3xi32>) -> tensor<3x1xi32>
// CHECK-NEXT:    %[[IDX:.+]] = stablehlo.concatenate %[[R0]], %[[R1]], dim = 1 : (tensor<3x1xi32>, tensor<3x1xi32>) -> tensor<3x2xi32>
// CHECK-NEXT:    %[[RES:.+]] = "stablehlo.scatter"(%arg0, %[[IDX]], %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg4: tensor<f64>, %arg5: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg5 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<3x2x5xf64>, tensor<3x2xi32>, tensor<3x1x1xf64>) -> tensor<3x2x5xf64>
// CHECK-NEXT:    return %[[RES]] : tensor<3x2x5xf64>
// CHECK-NEXT: }
