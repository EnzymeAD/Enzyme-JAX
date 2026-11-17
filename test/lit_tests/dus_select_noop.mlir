// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=dus_select_noop" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s
// RUN: stablehlo-translate %s --interpret
// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | stablehlo-translate - --interpret

module {
  // Issue #1926: common-ancestor slice in true branch is noop to DUS.
  func.func @dus_select_common_root_slice(
      %root: tensor<5x6x4xf32>,
      %upd: tensor<2x3x2xf32>,
      %pred: tensor<i1>) -> tensor<5x6x2xf32> {
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %c2 = stablehlo.constant dense<2> : tensor<i32>
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %old = stablehlo.slice %root [0:5, 0:6, 1:3] : (tensor<5x6x4xf32>) -> tensor<5x6x2xf32>
    %noop = stablehlo.slice %root [1:3, 2:5, 1:3] : (tensor<5x6x4xf32>) -> tensor<2x3x2xf32>
    %sel = stablehlo.select %pred, %noop, %upd : tensor<i1>, tensor<2x3x2xf32>
    %res = stablehlo.dynamic_update_slice %old, %sel, %c1, %c2, %c0 : (tensor<5x6x2xf32>, tensor<2x3x2xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x6x2xf32>
    return %res : tensor<5x6x2xf32>
  }
  // CHECK-LABEL: func.func @dus_select_common_root_slice
  // CHECK-DAG:     %[[C1:.+]] = stablehlo.constant dense<1> : tensor<i32>
  // CHECK-DAG:     %[[C2:.+]] = stablehlo.constant dense<2> : tensor<i32>
  // CHECK-DAG:     %[[C0:.+]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK:         %[[OLD:.+]] = stablehlo.slice %arg0 [0:5, 0:6, 1:3]
  // CHECK-NEXT:    %[[DUS:.+]] = stablehlo.dynamic_update_slice %[[OLD]], %arg1, %[[C1]], %[[C2]], %[[C0]]
  // CHECK-NEXT:    %[[SEL:.+]] = stablehlo.select %arg2, %[[OLD]], %[[DUS]]
  // CHECK-NEXT:    return %[[SEL]]

  // Direct slice in false branch is noop to DUS.
  func.func @dus_select_false_noop(
      %old: tensor<4x5xf32>,
      %upd: tensor<2x3xf32>,
      %pred: tensor<i1>) -> tensor<4x5xf32> {
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %noop = stablehlo.slice %old [1:3, 1:4] : (tensor<4x5xf32>) -> tensor<2x3xf32>
    %sel = stablehlo.select %pred, %upd, %noop : tensor<i1>, tensor<2x3xf32>
    %res = stablehlo.dynamic_update_slice %old, %sel, %c1, %c1 : (tensor<4x5xf32>, tensor<2x3xf32>, tensor<i32>, tensor<i32>) -> tensor<4x5xf32>
    return %res : tensor<4x5xf32>
  }
  // CHECK-LABEL: func.func @dus_select_false_noop
  // CHECK-NEXT:    %[[C1:.+]] = stablehlo.constant dense<1> : tensor<i32>
  // CHECK-NEXT:    %[[DUS:.+]] = stablehlo.dynamic_update_slice %arg0, %arg1, %[[C1]], %[[C1]]
  // CHECK-NEXT:    %[[SEL:.+]] = stablehlo.select %arg2, %[[DUS]], %arg0
  // CHECK-NEXT:    return %[[SEL]]

  // DynamicSlice at identical dynamic start indices is noop to DUS.
  func.func @dus_select_dynamic_slice_noop(
      %old: tensor<4x5xf32>,
      %upd: tensor<2x2xf32>,
      %i: tensor<i32>,
      %j: tensor<i32>,
      %pred: tensor<i1>) -> tensor<4x5xf32> {
    %noop = stablehlo.dynamic_slice %old, %i, %j, sizes = [2, 2] : (tensor<4x5xf32>, tensor<i32>, tensor<i32>) -> tensor<2x2xf32>
    %sel = stablehlo.select %pred, %noop, %upd : tensor<i1>, tensor<2x2xf32>
    %res = stablehlo.dynamic_update_slice %old, %sel, %i, %j : (tensor<4x5xf32>, tensor<2x2xf32>, tensor<i32>, tensor<i32>) -> tensor<4x5xf32>
    return %res : tensor<4x5xf32>
  }
  // CHECK-LABEL: func.func @dus_select_dynamic_slice_noop
  // CHECK-NEXT:    %[[DUS:.+]] = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3
  // CHECK-NEXT:    %[[SEL:.+]] = stablehlo.select %arg4, %arg0, %[[DUS]]
  // CHECK-NEXT:    return %[[SEL]]

  func.func @main() {
    %iota120 = stablehlo.iota dim = 0 : tensor<120xf32>
    %root = stablehlo.reshape %iota120 : (tensor<120xf32>) -> tensor<5x6x4xf32>
    %upd3 = stablehlo.constant dense<-9.0> : tensor<2x3x2xf32>
    %t = stablehlo.constant dense<true> : tensor<i1>
    %f = stablehlo.constant dense<false> : tensor<i1>

    %r_true = func.call @dus_select_common_root_slice(%root, %upd3, %t) : (tensor<5x6x4xf32>, tensor<2x3x2xf32>, tensor<i1>) -> tensor<5x6x2xf32>
    %exp_old = stablehlo.slice %root [0:5, 0:6, 1:3] : (tensor<5x6x4xf32>) -> tensor<5x6x2xf32>
    check.expect_almost_eq %r_true, %exp_old : tensor<5x6x2xf32>

    %r_false = func.call @dus_select_common_root_slice(%root, %upd3, %f) : (tensor<5x6x4xf32>, tensor<2x3x2xf32>, tensor<i1>) -> tensor<5x6x2xf32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %c2 = stablehlo.constant dense<2> : tensor<i32>
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %exp_upd = stablehlo.dynamic_update_slice %exp_old, %upd3, %c1, %c2, %c0 : (tensor<5x6x2xf32>, tensor<2x3x2xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x6x2xf32>
    check.expect_almost_eq %r_false, %exp_upd : tensor<5x6x2xf32>

    %iota20 = stablehlo.iota dim = 0 : tensor<20xf32>
    %old2 = stablehlo.reshape %iota20 : (tensor<20xf32>) -> tensor<4x5xf32>
    %upd2 = stablehlo.constant dense<42.0> : tensor<2x3xf32>
    %rf_t = func.call @dus_select_false_noop(%old2, %upd2, %t) : (tensor<4x5xf32>, tensor<2x3xf32>, tensor<i1>) -> tensor<4x5xf32>
    %rf_f = func.call @dus_select_false_noop(%old2, %upd2, %f) : (tensor<4x5xf32>, tensor<2x3xf32>, tensor<i1>) -> tensor<4x5xf32>
    check.expect_almost_eq %rf_f, %old2 : tensor<4x5xf32>
    %exp_rf_t = stablehlo.dynamic_update_slice %old2, %upd2, %c1, %c1 : (tensor<4x5xf32>, tensor<2x3xf32>, tensor<i32>, tensor<i32>) -> tensor<4x5xf32>
    check.expect_almost_eq %rf_t, %exp_rf_t : tensor<4x5xf32>

    %upd22 = stablehlo.constant dense<77.0> : tensor<2x2xf32>
    %c99 = stablehlo.constant dense<99> : tensor<i32>
    %rd_t = func.call @dus_select_dynamic_slice_noop(%old2, %upd22, %c99, %c1, %t) : (tensor<4x5xf32>, tensor<2x2xf32>, tensor<i32>, tensor<i32>, tensor<i1>) -> tensor<4x5xf32>
    %rd_f = func.call @dus_select_dynamic_slice_noop(%old2, %upd22, %c99, %c1, %f) : (tensor<4x5xf32>, tensor<2x2xf32>, tensor<i32>, tensor<i32>, tensor<i1>) -> tensor<4x5xf32>
    check.expect_almost_eq %rd_t, %old2 : tensor<4x5xf32>
    %exp_rd_f = stablehlo.dynamic_update_slice %old2, %upd22, %c99, %c1 : (tensor<4x5xf32>, tensor<2x2xf32>, tensor<i32>, tensor<i32>) -> tensor<4x5xf32>
    check.expect_almost_eq %rd_f, %exp_rd_f : tensor<4x5xf32>

    return
  }
}
