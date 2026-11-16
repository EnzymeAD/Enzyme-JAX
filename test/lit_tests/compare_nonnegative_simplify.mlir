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

func.func @nonneg_lt_neg_default_type(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<-5> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare LT, %0, %c : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @nonneg_lt_neg_default_type
// CHECK-NEXT: %c = stablehlo.constant dense<false> : tensor<4xi1>
// CHECK-NEXT: return %c

func.func @nonneg_lt_neg_notype(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<-5> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare LT, %0, %c, NOTYPE : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @nonneg_lt_neg_notype
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

// Bitwise OR of two non-negative values has sign bit 0, so it is >= 0
func.func @or_nonneg_lt_neg_signed(%arg0: tensor<4xi64>, %arg1: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<-5> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %1 = stablehlo.abs %arg1 : tensor<4xi64>
    %2 = stablehlo.or %0, %1 : tensor<4xi64>
    %cmp = stablehlo.compare LT, %2, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @or_nonneg_lt_neg_signed
// CHECK-NEXT: %c = stablehlo.constant dense<false> : tensor<4xi1>
// CHECK-NEXT: return %c

// Bitwise XOR of two non-negative values has sign bit 0, so it is >= 0
func.func @xor_nonneg_ge_zero_signed(%arg0: tensor<4xi64>, %arg1: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<0> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %1 = stablehlo.abs %arg1 : tensor<4xi64>
    %2 = stablehlo.xor %0, %1 : tensor<4xi64>
    %cmp = stablehlo.compare GE, %2, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @xor_nonneg_ge_zero_signed
// CHECK-NEXT: %c = stablehlo.constant dense<true> : tensor<4xi1>
// CHECK-NEXT: return %c

// Non-wrapping integer iota on i8 (extent <= 128) is guaranteed >= 0
func.func @iota_no_wrap_or_lt_zero_signed(%arg0: tensor<100xi8>) -> tensor<100xi1> {
    %c0 = stablehlo.constant dense<0> : tensor<100xi8>
    %0 = stablehlo.iota dim = 0 : tensor<100xi8>
    %1 = stablehlo.abs %arg0 : tensor<100xi8>
    %2 = stablehlo.or %0, %1 : tensor<100xi8>
    %cmp = stablehlo.compare LT, %2, %c0, SIGNED : (tensor<100xi8>, tensor<100xi8>) -> tensor<100xi1>
    return %cmp : tensor<100xi1>
}
// CHECK-LABEL: func.func @iota_no_wrap_or_lt_zero_signed
// CHECK-NEXT: %c = stablehlo.constant dense<false> : tensor<100xi1>
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
    %0 = stablehlo.abs %arg0 {enzymexla.no_nan = [#enzymexla.guaranteed<GUARANTEED>]} : tensor<4xf64>
    %cmp = stablehlo.compare LT, %0, %c, FLOAT : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @nonneg_lt_neg_float
// CHECK-NEXT: %c = stablehlo.constant dense<false> : tensor<4xi1>
// CHECK-NEXT: return %c

// abs(x) >= -5.0 is always true for floats
func.func @nonneg_ge_neg_float(%arg0: tensor<4xf64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<-5.0> : tensor<4xf64>
    %0 = stablehlo.abs %arg0 {enzymexla.no_nan = [#enzymexla.guaranteed<GUARANTEED>]} : tensor<4xf64>
    %cmp = stablehlo.compare GE, %0, %c, FLOAT : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @nonneg_ge_neg_float
// CHECK-NEXT: %c = stablehlo.constant dense<true> : tensor<4xi1>
// CHECK-NEXT: return %c

// x*x >= 0.0 is always true for floats (when x is not NaN)
func.func @square_ge_zero_float(%arg0: tensor<4xf64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<0.0> : tensor<4xf64>
    %0 = stablehlo.multiply %arg0, %arg0 {enzymexla.no_nan = [#enzymexla.guaranteed<GUARANTEED>]} : tensor<4xf64>
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

// Bitwise NOT flips the sign bit, so ~abs(x) is negative and cannot be assumed >= 0
func.func @bitwise_not_abs_lt_zero_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<0> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %1 = stablehlo.not %0 : tensor<4xi64>
    %cmp = stablehlo.compare LT, %1, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @bitwise_not_abs_lt_zero_signed
// CHECK: stablehlo.compare LT

// Bitwise OR with potentially negative operand is not guaranteed non-negative
func.func @or_unknown_lt_zero_signed(%arg0: tensor<4xi64>, %arg1: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<0> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %1 = stablehlo.or %0, %arg1 : tensor<4xi64>
    %cmp = stablehlo.compare LT, %1, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @or_unknown_lt_zero_signed
// CHECK: stablehlo.compare LT

// Bitwise XOR with potentially negative operand is not guaranteed non-negative
func.func @xor_unknown_lt_zero_signed(%arg0: tensor<4xi64>, %arg1: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<0> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %1 = stablehlo.xor %0, %arg1 : tensor<4xi64>
    %cmp = stablehlo.compare LT, %1, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @xor_unknown_lt_zero_signed
// CHECK: stablehlo.compare LT

// Product of two distinct block arguments is not a square and may be negative
func.func @mul_distinct_block_args_lt_zero_signed(%arg0: tensor<4xi64>, %arg1: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<0> : tensor<4xi64>
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<4xi64>
    %cmp = stablehlo.compare LT, %0, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @mul_distinct_block_args_lt_zero_signed
// CHECK: stablehlo.compare LT

// iota on i8 with extent > 128 wraps into negative signed values
func.func @iota_may_wrap_or_lt_zero_signed(%arg0: tensor<200xi8>) -> tensor<200xi1> {
    %c = stablehlo.constant dense<0> : tensor<200xi8>
    %0 = stablehlo.iota dim = 0 : tensor<200xi8>
    %1 = stablehlo.abs %arg0 : tensor<200xi8>
    %2 = stablehlo.or %0, %1 : tensor<200xi8>
    %cmp = stablehlo.compare LT, %2, %c, SIGNED : (tensor<200xi8>, tensor<200xi8>) -> tensor<200xi1>
    return %cmp : tensor<200xi1>
}
// CHECK-LABEL: func.func @iota_may_wrap_or_lt_zero_signed
// CHECK: stablehlo.compare LT
