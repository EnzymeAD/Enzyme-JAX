// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  // Test: reduce(max, exp(x) * y) where exp(x) is non-negative scalar
  // Should transform to: exp(x) * reduce(max, y)
  func.func @reduce_max_mul_exp_scalar(%arg0: tensor<f64>, %arg1: tensor<64x64xf64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
    %exp = stablehlo.exponential %arg0 {enzymexla.non_negative = [#enzymexla<guaranteed GUARANTEED>]} : tensor<f64>
    %bcast = stablehlo.broadcast_in_dim %exp, dims = [] : (tensor<f64>) -> tensor<64x64xf64>
    %mul = stablehlo.multiply %arg1, %bcast : tensor<64x64xf64>
    %reduce = stablehlo.reduce(%mul init: %cst) applies stablehlo.maximum across dimensions = [0, 1] : (tensor<64x64xf64>, tensor<f64>) -> tensor<f64>
    return %reduce : tensor<f64>
  }

  // Test: reduce(min, exp(x) * y) where exp(x) is non-negative scalar
  // Should transform to: exp(x) * reduce(min, y)
  func.func @reduce_min_mul_exp_scalar(%arg0: tensor<f64>, %arg1: tensor<64x64xf64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
    %exp = stablehlo.exponential %arg0 {enzymexla.non_negative = [#enzymexla<guaranteed GUARANTEED>]} : tensor<f64>
    %bcast = stablehlo.broadcast_in_dim %exp, dims = [] : (tensor<f64>) -> tensor<64x64xf64>
    %mul = stablehlo.multiply %arg1, %bcast : tensor<64x64xf64>
    %reduce = stablehlo.reduce(%mul init: %cst) applies stablehlo.minimum across dimensions = [0, 1] : (tensor<64x64xf64>, tensor<f64>) -> tensor<f64>
    return %reduce : tensor<f64>
  }

  // Test: reduce(max, scalar * tensor) with scalar on lhs
  func.func @reduce_max_mul_scalar_lhs(%arg0: tensor<f64>, %arg1: tensor<32xf64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
    %exp = stablehlo.exponential %arg0 {enzymexla.non_negative = [#enzymexla<guaranteed GUARANTEED>]} : tensor<f64>
    %bcast = stablehlo.broadcast_in_dim %exp, dims = [] : (tensor<f64>) -> tensor<32xf64>
    %mul = stablehlo.multiply %bcast, %arg1 : tensor<32xf64>
    %reduce = stablehlo.reduce(%mul init: %cst) applies stablehlo.maximum across dimensions = [0] : (tensor<32xf64>, tensor<f64>) -> tensor<f64>
    return %reduce : tensor<f64>
  }

  // Test: reduce(max, a * b) where neither is guaranteed non-negative - should NOT transform
  func.func @reduce_max_mul_no_guarantee(%arg0: tensor<f64>, %arg1: tensor<32xf64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
    %bcast = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f64>) -> tensor<32xf64>
    %mul = stablehlo.multiply %bcast, %arg1 : tensor<32xf64>
    %reduce = stablehlo.reduce(%mul init: %cst) applies stablehlo.maximum across dimensions = [0] : (tensor<32xf64>, tensor<f64>) -> tensor<f64>
    return %reduce : tensor<f64>
  }
}

// CHECK-LABEL: func.func @reduce_max_mul_exp_scalar
// CHECK-SAME: (%[[ARG0:.+]]: tensor<f64>, %[[ARG1:.+]]: tensor<64x64xf64>)
// CHECK: %[[CST:.+]] = stablehlo.constant
// CHECK: %[[EXP:.+]] = stablehlo.exponential %[[ARG0]]
// CHECK: %[[REDUCE:.+]] = stablehlo.reduce(%[[ARG1]] init: %[[CST]]) applies stablehlo.maximum across dimensions = [0, 1]
// CHECK: %[[MUL:.+]] = stablehlo.multiply %[[REDUCE]], %[[EXP]]
// CHECK: return %[[MUL]]

// CHECK-LABEL: func.func @reduce_min_mul_exp_scalar
// CHECK-SAME: (%[[ARG0:.+]]: tensor<f64>, %[[ARG1:.+]]: tensor<64x64xf64>)
// CHECK: %[[CST:.+]] = stablehlo.constant
// CHECK: %[[EXP:.+]] = stablehlo.exponential %[[ARG0]]
// CHECK: %[[REDUCE:.+]] = stablehlo.reduce(%[[ARG1]] init: %[[CST]]) applies stablehlo.minimum across dimensions = [0, 1]
// CHECK: %[[MUL:.+]] = stablehlo.multiply %[[REDUCE]], %[[EXP]]
// CHECK: return %[[MUL]]

// CHECK-LABEL: func.func @reduce_max_mul_scalar_lhs
// CHECK-SAME: (%[[ARG0:.+]]: tensor<f64>, %[[ARG1:.+]]: tensor<32xf64>)
// CHECK: %[[CST:.+]] = stablehlo.constant
// CHECK: %[[EXP:.+]] = stablehlo.exponential %[[ARG0]]
// CHECK: %[[REDUCE:.+]] = stablehlo.reduce(%[[ARG1]] init: %[[CST]]) applies stablehlo.maximum across dimensions = [0]
// CHECK: %[[MUL:.+]] = stablehlo.multiply %[[REDUCE]], %[[EXP]]
// CHECK: return %[[MUL]]

// CHECK-LABEL: func.func @reduce_max_mul_no_guarantee
// Should NOT be transformed - still has multiply before reduce
// CHECK: stablehlo.multiply
// CHECK: stablehlo.reduce
