// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=dus_dus_subsuming" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt="max_constant_expansion=0" --inline --canonicalize --symbol-dce | stablehlo-translate --interpret

// Repeated partial writes leave a long chain of disjoint rectangles. All
// updates come from a smaller tensor, so a full-shaped piecewise select cannot
// represent the result. Reject before computing provenance for the whole chain.
// The read from the penultimate DUS is disjoint from the final update. Moving
// it past that write would just be undone by SliceOfDynamicUpdate.
// CHECK-LABEL: func.func @partial_updates
// CHECK-NOT: enzymexla.piecewise_select
// CHECK-COUNT-24: stablehlo.dynamic_update_slice
// CHECK-NOT: enzymexla.piecewise_select
// CHECK: return
func.func @partial_updates(%dest: tensor<40x8xi32>, %update: tensor<2x4xi32>) -> (tensor<40x8xi32>, tensor<2x1xi32>) {
  %left = stablehlo.slice %update [0:2, 0:1] : (tensor<2x4xi32>) -> tensor<2x1xi32>
  %right = stablehlo.slice %update [0:2, 1:4] : (tensor<2x4xi32>) -> tensor<2x3xi32>
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %c3 = stablehlo.constant dense<3> : tensor<i32>
  %c4 = stablehlo.constant dense<4> : tensor<i32>
  %c6 = stablehlo.constant dense<6> : tensor<i32>
  %c7 = stablehlo.constant dense<7> : tensor<i32>
  %c9 = stablehlo.constant dense<9> : tensor<i32>
  %c10 = stablehlo.constant dense<10> : tensor<i32>
  %c12 = stablehlo.constant dense<12> : tensor<i32>
  %c13 = stablehlo.constant dense<13> : tensor<i32>
  %c15 = stablehlo.constant dense<15> : tensor<i32>
  %c16 = stablehlo.constant dense<16> : tensor<i32>
  %c18 = stablehlo.constant dense<18> : tensor<i32>
  %c19 = stablehlo.constant dense<19> : tensor<i32>
  %c21 = stablehlo.constant dense<21> : tensor<i32>
  %c22 = stablehlo.constant dense<22> : tensor<i32>
  %c24 = stablehlo.constant dense<24> : tensor<i32>
  %c25 = stablehlo.constant dense<25> : tensor<i32>
  %c27 = stablehlo.constant dense<27> : tensor<i32>
  %c28 = stablehlo.constant dense<28> : tensor<i32>
  %c30 = stablehlo.constant dense<30> : tensor<i32>
  %c31 = stablehlo.constant dense<31> : tensor<i32>
  %c33 = stablehlo.constant dense<33> : tensor<i32>
  %c34 = stablehlo.constant dense<34> : tensor<i32>
  %v0 = stablehlo.dynamic_update_slice %dest, %left, %c0, %c7 : (tensor<40x8xi32>, tensor<2x1xi32>, tensor<i32>, tensor<i32>) -> tensor<40x8xi32>
  %v1 = stablehlo.dynamic_update_slice %v0, %right, %c1, %c0 : (tensor<40x8xi32>, tensor<2x3xi32>, tensor<i32>, tensor<i32>) -> tensor<40x8xi32>
  %v2 = stablehlo.dynamic_update_slice %v1, %left, %c3, %c7 : (tensor<40x8xi32>, tensor<2x1xi32>, tensor<i32>, tensor<i32>) -> tensor<40x8xi32>
  %v3 = stablehlo.dynamic_update_slice %v2, %right, %c4, %c0 : (tensor<40x8xi32>, tensor<2x3xi32>, tensor<i32>, tensor<i32>) -> tensor<40x8xi32>
  %v4 = stablehlo.dynamic_update_slice %v3, %left, %c6, %c7 : (tensor<40x8xi32>, tensor<2x1xi32>, tensor<i32>, tensor<i32>) -> tensor<40x8xi32>
  %v5 = stablehlo.dynamic_update_slice %v4, %right, %c7, %c0 : (tensor<40x8xi32>, tensor<2x3xi32>, tensor<i32>, tensor<i32>) -> tensor<40x8xi32>
  %v6 = stablehlo.dynamic_update_slice %v5, %left, %c9, %c7 : (tensor<40x8xi32>, tensor<2x1xi32>, tensor<i32>, tensor<i32>) -> tensor<40x8xi32>
  %v7 = stablehlo.dynamic_update_slice %v6, %right, %c10, %c0 : (tensor<40x8xi32>, tensor<2x3xi32>, tensor<i32>, tensor<i32>) -> tensor<40x8xi32>
  %v8 = stablehlo.dynamic_update_slice %v7, %left, %c12, %c7 : (tensor<40x8xi32>, tensor<2x1xi32>, tensor<i32>, tensor<i32>) -> tensor<40x8xi32>
  %v9 = stablehlo.dynamic_update_slice %v8, %right, %c13, %c0 : (tensor<40x8xi32>, tensor<2x3xi32>, tensor<i32>, tensor<i32>) -> tensor<40x8xi32>
  %v10 = stablehlo.dynamic_update_slice %v9, %left, %c15, %c7 : (tensor<40x8xi32>, tensor<2x1xi32>, tensor<i32>, tensor<i32>) -> tensor<40x8xi32>
  %v11 = stablehlo.dynamic_update_slice %v10, %right, %c16, %c0 : (tensor<40x8xi32>, tensor<2x3xi32>, tensor<i32>, tensor<i32>) -> tensor<40x8xi32>
  %v12 = stablehlo.dynamic_update_slice %v11, %left, %c18, %c7 : (tensor<40x8xi32>, tensor<2x1xi32>, tensor<i32>, tensor<i32>) -> tensor<40x8xi32>
  %v13 = stablehlo.dynamic_update_slice %v12, %right, %c19, %c0 : (tensor<40x8xi32>, tensor<2x3xi32>, tensor<i32>, tensor<i32>) -> tensor<40x8xi32>
  %v14 = stablehlo.dynamic_update_slice %v13, %left, %c21, %c7 : (tensor<40x8xi32>, tensor<2x1xi32>, tensor<i32>, tensor<i32>) -> tensor<40x8xi32>
  %v15 = stablehlo.dynamic_update_slice %v14, %right, %c22, %c0 : (tensor<40x8xi32>, tensor<2x3xi32>, tensor<i32>, tensor<i32>) -> tensor<40x8xi32>
  %v16 = stablehlo.dynamic_update_slice %v15, %left, %c24, %c7 : (tensor<40x8xi32>, tensor<2x1xi32>, tensor<i32>, tensor<i32>) -> tensor<40x8xi32>
  %v17 = stablehlo.dynamic_update_slice %v16, %right, %c25, %c0 : (tensor<40x8xi32>, tensor<2x3xi32>, tensor<i32>, tensor<i32>) -> tensor<40x8xi32>
  %v18 = stablehlo.dynamic_update_slice %v17, %left, %c27, %c7 : (tensor<40x8xi32>, tensor<2x1xi32>, tensor<i32>, tensor<i32>) -> tensor<40x8xi32>
  %v19 = stablehlo.dynamic_update_slice %v18, %right, %c28, %c0 : (tensor<40x8xi32>, tensor<2x3xi32>, tensor<i32>, tensor<i32>) -> tensor<40x8xi32>
  %v20 = stablehlo.dynamic_update_slice %v19, %left, %c30, %c7 : (tensor<40x8xi32>, tensor<2x1xi32>, tensor<i32>, tensor<i32>) -> tensor<40x8xi32>
  %v21 = stablehlo.dynamic_update_slice %v20, %right, %c31, %c0 : (tensor<40x8xi32>, tensor<2x3xi32>, tensor<i32>, tensor<i32>) -> tensor<40x8xi32>
  %v22 = stablehlo.dynamic_update_slice %v21, %left, %c33, %c7 : (tensor<40x8xi32>, tensor<2x1xi32>, tensor<i32>, tensor<i32>) -> tensor<40x8xi32>
  %read = stablehlo.slice %v22 [0:2, 7:8] : (tensor<40x8xi32>) -> tensor<2x1xi32>
  %v23 = stablehlo.dynamic_update_slice %v22, %right, %c34, %c0 : (tensor<40x8xi32>, tensor<2x3xi32>, tensor<i32>, tensor<i32>) -> tensor<40x8xi32>
  return %v23, %read : tensor<40x8xi32>, tensor<2x1xi32>
}

// Check the written cells and the untouched gaps after the default pipeline.
func.func @main() {
  %dest = stablehlo.constant dense<0> : tensor<40x8xi32>
  %update = stablehlo.constant dense<[[1, 2, 3, 4], [5, 6, 7, 8]]> : tensor<2x4xi32>
  %result, %read = func.call @partial_updates(%dest, %update) : (tensor<40x8xi32>, tensor<2x4xi32>) -> (tensor<40x8xi32>, tensor<2x1xi32>)
  %expected = stablehlo.constant dense<[
    [0, 0, 0, 0, 0, 0, 0, 1],
    [2, 3, 4, 0, 0, 0, 0, 5],
    [6, 7, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [2, 3, 4, 0, 0, 0, 0, 5],
    [6, 7, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [2, 3, 4, 0, 0, 0, 0, 5],
    [6, 7, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [2, 3, 4, 0, 0, 0, 0, 5],
    [6, 7, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [2, 3, 4, 0, 0, 0, 0, 5],
    [6, 7, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [2, 3, 4, 0, 0, 0, 0, 5],
    [6, 7, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [2, 3, 4, 0, 0, 0, 0, 5],
    [6, 7, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [2, 3, 4, 0, 0, 0, 0, 5],
    [6, 7, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [2, 3, 4, 0, 0, 0, 0, 5],
    [6, 7, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [2, 3, 4, 0, 0, 0, 0, 5],
    [6, 7, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [2, 3, 4, 0, 0, 0, 0, 5],
    [6, 7, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [2, 3, 4, 0, 0, 0, 0, 5],
    [6, 7, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
  ]> : tensor<40x8xi32>
  check.expect_eq %result, %expected : tensor<40x8xi32>
  %expected_read = stablehlo.constant dense<[[1], [5]]> : tensor<2x1xi32>
  check.expect_eq %read, %expected_read : tensor<2x1xi32>
  return
}
