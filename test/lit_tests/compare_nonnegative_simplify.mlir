// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// ============================================================================
// Tests for non-negative compared to negative constant (SIGNED integers)
// ============================================================================

// abs(x) is always >= 0, so abs(x) < -5 is always false
func.func @nonneg_lt_neg_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<-5> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare LT, %0, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @nonneg_lt_neg_signed
// CHECK-NEXT: %c = stablehlo.constant dense<false> : tensor<4xi1>
// CHECK-NEXT: return %c

// abs(x) is always >= 0, so abs(x) <= -5 is always false
func.func @nonneg_le_neg_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<-5> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare LE, %0, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @nonneg_le_neg_signed
// CHECK-NEXT: %c = stablehlo.constant dense<false> : tensor<4xi1>
// CHECK-NEXT: return %c

// abs(x) is always >= 0, so abs(x) == -5 is always false
func.func @nonneg_eq_neg_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<-5> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare EQ, %0, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @nonneg_eq_neg_signed
// CHECK-NEXT: %c = stablehlo.constant dense<false> : tensor<4xi1>
// CHECK-NEXT: return %c

// abs(x) is always >= 0, so abs(x) > -5 is always true
func.func @nonneg_gt_neg_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<-5> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare GT, %0, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @nonneg_gt_neg_signed
// CHECK-NEXT: %c = stablehlo.constant dense<true> : tensor<4xi1>
// CHECK-NEXT: return %c

// abs(x) is always >= 0, so abs(x) >= -5 is always true
func.func @nonneg_ge_neg_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<-5> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare GE, %0, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @nonneg_ge_neg_signed
// CHECK-NEXT: %c = stablehlo.constant dense<true> : tensor<4xi1>
// CHECK-NEXT: return %c

// abs(x) is always >= 0, so abs(x) != -5 is always true
func.func @nonneg_ne_neg_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<-5> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare NE, %0, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @nonneg_ne_neg_signed
// CHECK-NEXT: %c = stablehlo.constant dense<true> : tensor<4xi1>
// CHECK-NEXT: return %c

// ============================================================================
// Tests for non-negative compared to zero (SIGNED integers)
// ============================================================================

// abs(x) is always >= 0, so abs(x) < 0 is always false
func.func @nonneg_lt_zero_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<0> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare LT, %0, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @nonneg_lt_zero_signed
// CHECK-NEXT: %c = stablehlo.constant dense<false> : tensor<4xi1>
// CHECK-NEXT: return %c

// abs(x) is always >= 0, so abs(x) >= 0 is always true
func.func @nonneg_ge_zero_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<0> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare GE, %0, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @nonneg_ge_zero_signed
// CHECK-NEXT: %c = stablehlo.constant dense<true> : tensor<4xi1>
// CHECK-NEXT: return %c

// ============================================================================
// Tests with different non-negative sources: iota, x*x
// ============================================================================

// iota produces values 0, 1, 2, ... which are always >= 0
func.func @iota_lt_neg_signed() -> tensor<4xi1> {
    %c = stablehlo.constant dense<-3> : tensor<4xi64>
    %iota = stablehlo.iota dim = 0 : tensor<4xi64>
    %cmp = stablehlo.compare LT, %iota, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @iota_lt_neg_signed
// CHECK-NEXT: %c = stablehlo.constant dense<false> : tensor<4xi1>
// CHECK-NEXT: return %c

// x*x is always >= 0 for real numbers
func.func @square_gt_neg_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<-7> : tensor<4xi64>
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare GT, %0, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @square_gt_neg_signed
// CHECK-NEXT: %c = stablehlo.constant dense<true> : tensor<4xi1>
// CHECK-NEXT: return %c

// ============================================================================
// Tests with constant on LHS (tests invertDirection logic)
// ============================================================================

// -5 < abs(x) is equivalent to abs(x) > -5, which is always true
func.func @neg_lt_nonneg_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<-5> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare LT, %c, %0, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @neg_lt_nonneg_signed
// CHECK-NEXT: %c = stablehlo.constant dense<true> : tensor<4xi1>
// CHECK-NEXT: return %c

// -5 >= abs(x) is equivalent to abs(x) <= -5, which is always false
func.func @neg_ge_nonneg_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<-5> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare GE, %c, %0, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @neg_ge_nonneg_signed
// CHECK-NEXT: %c = stablehlo.constant dense<false> : tensor<4xi1>
// CHECK-NEXT: return %c

// 0 > abs(x) is equivalent to abs(x) < 0, which is always false
func.func @zero_gt_nonneg_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<0> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare GT, %c, %0, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @zero_gt_nonneg_signed
// CHECK-NEXT: %c = stablehlo.constant dense<false> : tensor<4xi1>
// CHECK-NEXT: return %c

// 0 <= abs(x) is equivalent to abs(x) >= 0, which is always true
func.func @zero_le_nonneg_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<0> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare LE, %c, %0, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @zero_le_nonneg_signed
// CHECK-NEXT: %c = stablehlo.constant dense<true> : tensor<4xi1>
// CHECK-NEXT: return %c

// ============================================================================
// Tests for floats with non-negative values (abs produces no NaN result)
// ============================================================================

// abs(x) is always >= 0 for floats, so abs(x) < -5.0 is always false
func.func @nonneg_lt_neg_float(%arg0: tensor<4xf64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<-5.0> : tensor<4xf64>
    %0 = stablehlo.abs %arg0 {enzymexla.no_nan = [#enzymexla<guaranteed GUARANTEED>]} : tensor<4xf64>
    %cmp = stablehlo.compare LT, %0, %c, FLOAT : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @nonneg_lt_neg_float
// CHECK-NEXT: %c = stablehlo.constant dense<false> : tensor<4xi1>
// CHECK-NEXT: return %c

// abs(x) >= -5.0 is always true for floats
func.func @nonneg_ge_neg_float(%arg0: tensor<4xf64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<-5.0> : tensor<4xf64>
    %0 = stablehlo.abs %arg0 {enzymexla.no_nan = [#enzymexla<guaranteed GUARANTEED>]} : tensor<4xf64>
    %cmp = stablehlo.compare GE, %0, %c, FLOAT : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @nonneg_ge_neg_float
// CHECK-NEXT: %c = stablehlo.constant dense<true> : tensor<4xi1>
// CHECK-NEXT: return %c

// x*x >= 0.0 is always true for floats (when x is not NaN)
func.func @square_ge_zero_float(%arg0: tensor<4xf64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<0.0> : tensor<4xf64>
    %0 = stablehlo.multiply %arg0, %arg0 {enzymexla.no_nan = [#enzymexla<guaranteed GUARANTEED>]} : tensor<4xf64>
    %cmp = stablehlo.compare GE, %0, %c, FLOAT : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @square_ge_zero_float
// CHECK-NEXT: %c = stablehlo.constant dense<true> : tensor<4xi1>
// CHECK-NEXT: return %c

// ============================================================================
// Negative test cases (should NOT simplify)
// ============================================================================

// Non-negative compared to positive constant - cannot simplify
func.func @nonneg_lt_pos_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<5> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare LT, %0, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @nonneg_lt_pos_signed
// CHECK: stablehlo.compare LT

// Non-splat constant - cannot simplify
func.func @nonneg_lt_nonsplat_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<[-5, -3, -1, 0]> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare LT, %0, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @nonneg_lt_nonsplat_signed
// CHECK: stablehlo.compare LT

// Not guaranteed non-negative - cannot simplify
func.func @not_nonneg_lt_neg_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<-5> : tensor<4xi64>
    %cmp = stablehlo.compare LT, %arg0, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @not_nonneg_lt_neg_signed
// CHECK: stablehlo.compare
