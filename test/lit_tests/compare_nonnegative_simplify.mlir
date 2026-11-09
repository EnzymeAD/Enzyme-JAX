// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-opt{no_signed_wrap=true})" | FileCheck %s --check-prefixes=CHECK,NSW
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s --check-prefixes=CHECK,WRAP

// ============================================================================
// Tests for non-negative compared to negative constant (SIGNED integers)
// ============================================================================

// Unbounded abs(x) may wrap on INT_MIN in two's complement, so it folds under
// no_signed_wrap=true (NSW) and stays un-simplified by default (WRAP).
func.func @nonneg_lt_neg_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<-5> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare LT, %0, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @nonneg_lt_neg_signed
// NSW-NEXT: %[[CST:.+]] = stablehlo.constant dense<false> : tensor<4xi1>
// NSW-NEXT: return %[[CST]]
// WRAP: stablehlo.compare LT

func.func @nonneg_lt_neg_default_type(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<-5> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare LT, %0, %c : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @nonneg_lt_neg_default_type
// NSW-NEXT: %[[CST:.+]] = stablehlo.constant dense<false> : tensor<4xi1>
// NSW-NEXT: return %[[CST]]
// WRAP: stablehlo.compare LT

func.func @nonneg_lt_neg_notype(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<-5> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare LT, %0, %c, NOTYPE : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @nonneg_lt_neg_notype
// NSW-NEXT: %[[CST:.+]] = stablehlo.constant dense<false> : tensor<4xi1>
// NSW-NEXT: return %[[CST]]
// WRAP: stablehlo.compare LT

// abs(x) <= -5
func.func @nonneg_le_neg_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<-5> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare LE, %0, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @nonneg_le_neg_signed
// NSW-NEXT: %[[CST:.+]] = stablehlo.constant dense<false> : tensor<4xi1>
// NSW-NEXT: return %[[CST]]
// WRAP: stablehlo.compare LE

// abs(x) == -5
func.func @nonneg_eq_neg_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<-5> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare EQ, %0, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @nonneg_eq_neg_signed
// NSW-NEXT: %[[CST:.+]] = stablehlo.constant dense<false> : tensor<4xi1>
// NSW-NEXT: return %[[CST]]
// WRAP: stablehlo.compare EQ

// abs(x) > -5
func.func @nonneg_gt_neg_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<-5> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare GT, %0, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @nonneg_gt_neg_signed
// NSW-NEXT: %[[CST:.+]] = stablehlo.constant dense<true> : tensor<4xi1>
// NSW-NEXT: return %[[CST]]
// WRAP: stablehlo.compare GT

// abs(x) >= -5
func.func @nonneg_ge_neg_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<-5> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare GE, %0, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @nonneg_ge_neg_signed
// NSW-NEXT: %[[CST:.+]] = stablehlo.constant dense<true> : tensor<4xi1>
// NSW-NEXT: return %[[CST]]
// WRAP: stablehlo.compare GE

// abs(x) != -5
func.func @nonneg_ne_neg_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<-5> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare NE, %0, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @nonneg_ne_neg_signed
// NSW-NEXT: %[[CST:.+]] = stablehlo.constant dense<true> : tensor<4xi1>
// NSW-NEXT: return %[[CST]]
// WRAP: stablehlo.compare NE

// When %arg0 is bounded away from INT_MIN, abs(x) is non-negative even under WRAP
func.func @bounded_abs_lt_neg_signed(%arg0: tensor<4xi64> {enzymexla.bounds = [-100, 100]}) -> tensor<4xi1> {
    %c = stablehlo.constant dense<-5> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare LT, %0, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @bounded_abs_lt_neg_signed
// CHECK-NEXT: %[[CST:.+]] = stablehlo.constant dense<false> : tensor<4xi1>
// CHECK-NEXT: return %[[CST]]

// abs of an already non-negative value (e.g. x & 7) cannot be INT_MIN even under WRAP
func.func @abs_of_masked_lt_neg_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<-5> : tensor<4xi64>
    %mask = stablehlo.constant dense<7> : tensor<4xi64>
    %0 = stablehlo.and %arg0, %mask : tensor<4xi64>
    %1 = stablehlo.abs %0 : tensor<4xi64>
    %cmp = stablehlo.compare LT, %1, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @abs_of_masked_lt_neg_signed
// CHECK-NEXT: %[[CST:.+]] = stablehlo.constant dense<false> : tensor<4xi1>
// CHECK-NEXT: return %[[CST]]

// ============================================================================
// Tests for non-negative compared to zero (SIGNED integers)
// ============================================================================

// abs(x) < 0
func.func @nonneg_lt_zero_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<0> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare LT, %0, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @nonneg_lt_zero_signed
// NSW-NEXT: %[[CST:.+]] = stablehlo.constant dense<false> : tensor<4xi1>
// NSW-NEXT: return %[[CST]]
// WRAP: stablehlo.compare LT

// abs(x) >= 0
func.func @nonneg_ge_zero_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<0> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare GE, %0, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @nonneg_ge_zero_signed
// NSW-NEXT: %[[CST:.+]] = stablehlo.constant dense<true> : tensor<4xi1>
// NSW-NEXT: return %[[CST]]
// WRAP: stablehlo.compare GE

// ============================================================================
// Tests with different non-negative sources: iota, x*x, bitwise, add/mul
// ============================================================================

// Non-wrapping iota produces values 0, 1, 2, 3 which are >= 0 in both modes
func.func @iota_lt_neg_signed() -> tensor<4xi1> {
    %c = stablehlo.constant dense<-3> : tensor<4xi64>
    %iota = stablehlo.iota dim = 0 : tensor<4xi64>
    %cmp = stablehlo.compare LT, %iota, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @iota_lt_neg_signed
// CHECK-NEXT: %c = stablehlo.constant dense<false> : tensor<4xi1>
// CHECK-NEXT: return %c

// Unbounded x*x on signed integers may wrap around INT_MAX unless NSW is enabled
func.func @square_gt_neg_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<-7> : tensor<4xi64>
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare GT, %0, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @square_gt_neg_signed
// NSW-NEXT: %[[CST:.+]] = stablehlo.constant dense<true> : tensor<4xi1>
// NSW-NEXT: return %[[CST]]
// WRAP: stablehlo.compare GT

// Bounded x*x whose product fits in signed i64 folds in both modes
func.func @bounded_square_gt_neg_signed(%arg0: tensor<4xi64> {enzymexla.bounds = [-1000, 1000]}) -> tensor<4xi1> {
    %c = stablehlo.constant dense<-7> : tensor<4xi64>
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare GT, %0, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @bounded_square_gt_neg_signed
// CHECK-NEXT: %[[CST:.+]] = stablehlo.constant dense<true> : tensor<4xi1>
// CHECK-NEXT: return %[[CST]]

// Bitwise OR of two non-negative values has sign bit 0, so it is >= 0
func.func @or_nonneg_lt_neg_signed(%arg0: tensor<4xi64> {enzymexla.bounds = [-100, 100]}, %arg1: tensor<4xi64> {enzymexla.bounds = [-100, 100]}) -> tensor<4xi1> {
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
func.func @xor_nonneg_ge_zero_signed(%arg0: tensor<4xi64> {enzymexla.bounds = [-100, 100]}, %arg1: tensor<4xi64> {enzymexla.bounds = [-100, 100]}) -> tensor<4xi1> {
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
func.func @iota_no_wrap_or_lt_zero_signed(%arg0: tensor<100xi8> {enzymexla.bounds = [-100, 100]}) -> tensor<100xi1> {
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

// Non-wrapping integer add of non-negative values is guaranteed >= 0 in both modes
func.func @add_no_wrap_or_lt_zero_signed(%arg0: tensor<100xi8> {enzymexla.bounds = [-100, 100]}) -> tensor<100xi1> {
    %c0 = stablehlo.constant dense<0> : tensor<100xi8>
    %c10 = stablehlo.constant dense<10> : tensor<100xi8>
    %0 = stablehlo.iota dim = 0 : tensor<100xi8>
    %1 = stablehlo.add %0, %c10 : tensor<100xi8>
    %2 = stablehlo.abs %arg0 : tensor<100xi8>
    %3 = stablehlo.or %1, %2 : tensor<100xi8>
    %cmp = stablehlo.compare LT, %3, %c0, SIGNED : (tensor<100xi8>, tensor<100xi8>) -> tensor<100xi1>
    return %cmp : tensor<100xi1>
}
// CHECK-LABEL: func.func @add_no_wrap_or_lt_zero_signed
// CHECK-NEXT: %[[CST:.+]] = stablehlo.constant dense<false> : tensor<100xi1>
// CHECK-NEXT: return %[[CST]]

// ============================================================================
// Tests with constant on LHS (tests invertDirection logic)
// ============================================================================

// -5 < abs(x)
func.func @neg_lt_nonneg_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<-5> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare LT, %c, %0, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @neg_lt_nonneg_signed
// NSW-NEXT: %[[CST:.+]] = stablehlo.constant dense<true> : tensor<4xi1>
// NSW-NEXT: return %[[CST]]
// WRAP: stablehlo.compare GT

// -5 >= abs(x)
func.func @neg_ge_nonneg_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<-5> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare GE, %c, %0, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @neg_ge_nonneg_signed
// NSW-NEXT: %[[CST:.+]] = stablehlo.constant dense<false> : tensor<4xi1>
// NSW-NEXT: return %[[CST]]
// WRAP: stablehlo.compare LE

// 0 > abs(x)
func.func @zero_gt_nonneg_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<0> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare GT, %c, %0, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @zero_gt_nonneg_signed
// NSW-NEXT: %[[CST:.+]] = stablehlo.constant dense<false> : tensor<4xi1>
// NSW-NEXT: return %[[CST]]
// WRAP: stablehlo.compare LT

// 0 <= abs(x)
func.func @zero_le_nonneg_signed(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<0> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %cmp = stablehlo.compare LE, %c, %0, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
}
// CHECK-LABEL: func.func @zero_le_nonneg_signed
// NSW-NEXT: %[[CST:.+]] = stablehlo.constant dense<true> : tensor<4xi1>
// NSW-NEXT: return %[[CST]]
// WRAP: stablehlo.compare GE

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
// Overflow-sensitive add/mul test cases (fold under NSW, preserved under WRAP)
// ============================================================================

// Integer add that may wrap signed i8: folds under NSW, preserved under WRAP
func.func @add_may_wrap_lt_zero_signed() -> tensor<100xi1> {
    %c0 = stablehlo.constant dense<0> : tensor<100xi8>
    %c50 = stablehlo.constant dense<50> : tensor<100xi8>
    %0 = stablehlo.iota dim = 0 : tensor<100xi8>
    %1 = stablehlo.add %0, %c50 : tensor<100xi8>
    %cmp = stablehlo.compare LT, %1, %c0, SIGNED : (tensor<100xi8>, tensor<100xi8>) -> tensor<100xi1>
    return %cmp : tensor<100xi1>
}
// CHECK-LABEL: func.func @add_may_wrap_lt_zero_signed
// NSW-NEXT: %[[CST:.+]] = stablehlo.constant dense<false> : tensor<100xi1>
// NSW-NEXT: return %[[CST]]
// WRAP: stablehlo.compare LT

// Integer multiply that may wrap signed i8: folds under NSW, preserved under WRAP
func.func @mul_may_wrap_ge_zero_signed() -> tensor<64xi1> {
    %c4 = stablehlo.constant dense<4> : tensor<64xi8>
    %c0 = stablehlo.constant dense<0> : tensor<64xi8>
    %0 = stablehlo.iota dim = 0 : tensor<64xi8>
    %1 = stablehlo.multiply %0, %c4 : tensor<64xi8>
    %cmp = stablehlo.compare GE, %1, %c0, SIGNED : (tensor<64xi8>, tensor<64xi8>) -> tensor<64xi1>
    return %cmp : tensor<64xi1>
}
// CHECK-LABEL: func.func @mul_may_wrap_ge_zero_signed
// NSW-NEXT: %[[CST:.+]] = stablehlo.constant dense<true> : tensor<64xi1>
// NSW-NEXT: return %[[CST]]
// WRAP: stablehlo.compare GE

// ============================================================================
// Negative test cases (should NOT simplify even under NSW)
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

// iota on i8 with extent > 128 has negative elements regardless of arithmetic overflow
func.func @iota_may_wrap_or_lt_zero_signed(%arg0: tensor<200xi8> {enzymexla.bounds = [-100, 100]}) -> tensor<200xi1> {
    %c = stablehlo.constant dense<0> : tensor<200xi8>
    %0 = stablehlo.iota dim = 0 : tensor<200xi8>
    %1 = stablehlo.abs %arg0 : tensor<200xi8>
    %2 = stablehlo.or %0, %1 : tensor<200xi8>
    %cmp = stablehlo.compare LT, %2, %c, SIGNED : (tensor<200xi8>, tensor<200xi8>) -> tensor<200xi1>
    return %cmp : tensor<200xi1>
}
// CHECK-LABEL: func.func @iota_may_wrap_or_lt_zero_signed
// CHECK: stablehlo.compare LT
