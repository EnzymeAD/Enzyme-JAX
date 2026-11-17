// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=dusdus_to_dusextend;lower_extend},transform-interpreter,enzyme-hlo-remove-transform)" | FileCheck %s
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=dusdus_to_dusextend;lower_extend},transform-interpreter,enzyme-hlo-remove-transform)" | stablehlo-translate - -allow-unregistered-dialect --interpret

module {
  // 1. Exact reproducer from #1370: lhs = 2, rhs = 1 on size-1 dimension.
  // CHECK-LABEL: func.func private @extend_unit_dim_oversized
  // CHECK-NEXT: %[[C:.*]] = stablehlo.concatenate %arg0, %arg0, %arg0, %arg0, dim = 0 : (tensor<1x2x1xf32>, tensor<1x2x1xf32>, tensor<1x2x1xf32>, tensor<1x2x1xf32>) -> tensor<4x2x1xf32>
  // CHECK-NEXT: return %[[C]] : tensor<4x2x1xf32>
  func.func private @extend_unit_dim_oversized(%x: tensor<1x2x1xf32>) -> tensor<4x2x1xf32> {
    %e = "enzymexla.extend"(%x) <{dimension = 0 : i64, lhs = 2 : i64, rhs = 1 : i64}> : (tensor<1x2x1xf32>) -> tensor<4x2x1xf32>
    return %e : tensor<4x2x1xf32>
  }

  // 2. Non-unit dimension with lhs > size and rhs > size (q*n + r).
  // CHECK-LABEL: func.func private @extend_multi_dim_oversized
  // CHECK-NEXT: %[[L:.*]] = stablehlo.slice %arg0 [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
  // CHECK-NEXT: %[[R:.*]] = stablehlo.slice %arg0 [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
  // CHECK-NEXT: %[[C:.*]] = stablehlo.concatenate %arg0, %[[L]], %arg0, %[[R]], %arg0, dim = 0 : (tensor<2x2xf32>, tensor<1x2xf32>, tensor<2x2xf32>, tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<8x2xf32>
  // CHECK-NEXT: return %[[C]] : tensor<8x2xf32>
  func.func private @extend_multi_dim_oversized(%x: tensor<2x2xf32>) -> tensor<8x2xf32> {
    %e = "enzymexla.extend"(%x) <{dimension = 0 : i64, lhs = 3 : i64, rhs = 3 : i64}> : (tensor<2x2xf32>) -> tensor<8x2xf32>
    return %e : tensor<8x2xf32>
  }

  // 3. DUSDUSToDUSExtend must NOT fold when the update slice reads past the middle window.
  // operand = [10, 20, 30, 40]:
  //   dus2 writes slice2 = operand[2:3] = [30] at idx 3 -> [10, 20, 30, 30]
  //   dus  writes slice1 = operand[2:4] = [30, 40] at idx 0 -> [30, 40, 30, 30]
  // CHECK-LABEL: func.func private @dusdus_slice_past_middle_no_fold
  // CHECK: stablehlo.dynamic_update_slice
  // CHECK: stablehlo.dynamic_update_slice
  func.func private @dusdus_slice_past_middle_no_fold(%x: tensor<4xf32>) -> tensor<4xf32> {
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c3 = stablehlo.constant dense<3> : tensor<i32>
    %s2 = stablehlo.slice %x [2:3] : (tensor<4xf32>) -> tensor<1xf32>
    %d2 = stablehlo.dynamic_update_slice %x, %s2, %c3 : (tensor<4xf32>, tensor<1xf32>, tensor<i32>) -> tensor<4xf32>
    %s1 = stablehlo.slice %x [2:4] : (tensor<4xf32>) -> tensor<2xf32>
    %d1 = stablehlo.dynamic_update_slice %d2, %s1, %c0 : (tensor<4xf32>, tensor<2xf32>, tensor<i32>) -> tensor<4xf32>
    return %d1 : tensor<4xf32>
  }

  func.func @main() {
    %u = stablehlo.constant dense<[[[3.0], [7.0]]]> : tensor<1x2x1xf32>
    %ru = func.call @extend_unit_dim_oversized(%u) : (tensor<1x2x1xf32>) -> tensor<4x2x1xf32>
    check.expect_almost_eq_const %ru, dense<[
      [[3.0], [7.0]],
      [[3.0], [7.0]],
      [[3.0], [7.0]],
      [[3.0], [7.0]]
    ]> : tensor<4x2x1xf32>

    %m = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %rm = func.call @extend_multi_dim_oversized(%m) : (tensor<2x2xf32>) -> tensor<8x2xf32>
    check.expect_almost_eq_const %rm, dense<[
      [1.0, 2.0],
      [3.0, 4.0],
      [1.0, 2.0],
      [1.0, 2.0],
      [3.0, 4.0],
      [3.0, 4.0],
      [1.0, 2.0],
      [3.0, 4.0]
    ]> : tensor<8x2xf32>

    %v = stablehlo.constant dense<[10.0, 20.0, 30.0, 40.0]> : tensor<4xf32>
    %rv = func.call @dusdus_slice_past_middle_no_fold(%v) : (tensor<4xf32>) -> tensor<4xf32>
    check.expect_almost_eq_const %rv, dense<[30.0, 40.0, 30.0, 30.0]> : tensor<4xf32>
    return
  }
}
