// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first expansion-size=2 dot-general-to-reduce=false" %s | FileCheck %s --check-prefix=FIRST
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last expansion-size=2 dot-general-to-reduce=false" %s | FileCheck %s --check-prefix=LAST
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple expansion-size=2 dot-general-to-reduce=false" %s | FileCheck %s --check-prefix=TUPLE
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first expansion-size=2 dot-general-to-reduce=false" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @power_test(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
  // FIRST-LABEL: @power_test
  // FIRST: stablehlo.concatenate
  // FIRST-NOT: stablehlo.power {{.*}} : tensor<f64>

  // LAST-LABEL: @power_test
  // LAST: stablehlo.concatenate
  // LAST-NOT: stablehlo.power {{.*}} : tensor<f64>

  // TUPLE-LABEL: @power_test
  // TUPLE: stablehlo.get_tuple_element
  // TUPLE-NOT: stablehlo.power {{.*}} : tensor<f64>

  %0 = stablehlo.power %arg0, %arg1 : tensor<f64>
  return %0 : tensor<f64>
}

func.func @power_neg_base(%arg0: tensor<f64>) -> tensor<f64> {
  // pow(-2.0, x): integer x gives exact result, non-integer x gives NaN.
  // FIRST-LABEL: @power_neg_base
  // FIRST: stablehlo.concatenate
  // FIRST-NOT: stablehlo.power {{.*}} : tensor<f64>

  // LAST-LABEL: @power_neg_base
  // LAST: stablehlo.concatenate
  // LAST-NOT: stablehlo.power {{.*}} : tensor<f64>

  // TUPLE-LABEL: @power_neg_base
  // TUPLE: stablehlo.get_tuple_element
  // TUPLE-NOT: stablehlo.power {{.*}} : tensor<f64>

  %cst = stablehlo.constant dense<-2.0> : tensor<f64>
  %0 = stablehlo.power %cst, %arg0 : tensor<f64>
  return %0 : tensor<f64>
}

func.func @power_const_exp(%arg0: tensor<f64>) -> tensor<f64> {
  // FIRST-LABEL: @power_const_exp
  // FIRST: stablehlo.concatenate
  // FIRST-NOT: stablehlo.power {{.*}} : tensor<f64>

  // LAST-LABEL: @power_const_exp
  // LAST: stablehlo.concatenate
  // LAST-NOT: stablehlo.power {{.*}} : tensor<f64>

  // TUPLE-LABEL: @power_const_exp
  // TUPLE: stablehlo.get_tuple_element
  // TUPLE-NOT: stablehlo.power {{.*}} : tensor<f64>

  %cst = stablehlo.constant dense<0.1> : tensor<f64>
  %0 = stablehlo.power %arg0, %cst : tensor<f64>
  return %0 : tensor<f64>
}

// Edge case: pow(x, 0.0). Here y is a stablehlo.constant inside the same
// function, so PowOpLowerPattern's integer fast path (n = 0) folds the op
// to constant 1.0 before any log/exp is emitted — the body has no exp/log
// chain at all. The marker we look for is therefore the multifloat
// representation of the constant itself: a concatenate of the split limbs in
// FIRST/LAST modes, and a stablehlo.tuple in TUPLE mode.
// (Note: the @main interpret check exercises pow(0.0, 0.0) via the general
// power_test wrapper, where y is a function argument and the fast path does
// not apply; that path correctly returns 1.0 via the y_is_zero correction.)
func.func @power_zero_exponent(%arg0: tensor<f64>) -> tensor<f64> {
  // FIRST-LABEL: @power_zero_exponent
  // FIRST: stablehlo.concatenate
  // FIRST-NOT: stablehlo.power {{.*}} : tensor<f64>

  // LAST-LABEL: @power_zero_exponent
  // LAST: stablehlo.concatenate
  // LAST-NOT: stablehlo.power {{.*}} : tensor<f64>

  // TUPLE-LABEL: @power_zero_exponent
  // TUPLE: stablehlo.tuple
  // TUPLE-NOT: stablehlo.power {{.*}} : tensor<f64>

  %cst = stablehlo.constant dense<0.0> : tensor<f64>
  %0 = stablehlo.power %arg0, %cst : tensor<f64>
  return %0 : tensor<f64>
}

// Edge case: pow(1.0, y) = 1 via exp(y * log(1)) = exp(0) = 1 for any y.
func.func @power_one_base(%arg0: tensor<f64>) -> tensor<f64> {
  // FIRST-LABEL: @power_one_base
  // FIRST: stablehlo.concatenate
  // FIRST-NOT: stablehlo.power {{.*}} : tensor<f64>

  // LAST-LABEL: @power_one_base
  // LAST: stablehlo.concatenate
  // LAST-NOT: stablehlo.power {{.*}} : tensor<f64>

  // TUPLE-LABEL: @power_one_base
  // TUPLE: stablehlo.get_tuple_element
  // TUPLE-NOT: stablehlo.power {{.*}} : tensor<f64>

  %cst = stablehlo.constant dense<1.0> : tensor<f64>
  %0 = stablehlo.power %cst, %arg0 : tensor<f64>
  return %0 : tensor<f64>
}

// Half-integer exponent: pow(x, 3.5) = x^3 * sqrt(x) via the sqrt fast path
// (no exp/log fallback).
func.func @power_half_integer_exp(%arg0: tensor<f64>) -> tensor<f64> {
  // FIRST-LABEL: @power_half_integer_exp
  // FIRST: stablehlo.rsqrt
  // FIRST: stablehlo.concatenate
  // FIRST-NOT: stablehlo.power {{.*}} : tensor<f64>

  // LAST-LABEL: @power_half_integer_exp
  // LAST: stablehlo.rsqrt
  // LAST: stablehlo.concatenate
  // LAST-NOT: stablehlo.power {{.*}} : tensor<f64>

  // TUPLE-LABEL: @power_half_integer_exp
  // TUPLE: stablehlo.rsqrt
  // TUPLE-NOT: stablehlo.power {{.*}} : tensor<f64>

  %cst = stablehlo.constant dense<3.5> : tensor<f64>
  %0 = stablehlo.power %arg0, %cst : tensor<f64>
  return %0 : tensor<f64>
}

// Negative half-integer exponent: pow(x, -1.5) = 1 / (x * sqrt(x)).
func.func @power_neg_half_integer_exp(%arg0: tensor<f64>) -> tensor<f64> {
  // FIRST-LABEL: @power_neg_half_integer_exp
  // FIRST: stablehlo.rsqrt
  // FIRST: stablehlo.concatenate
  // FIRST-NOT: stablehlo.power {{.*}} : tensor<f64>

  // LAST-LABEL: @power_neg_half_integer_exp
  // LAST: stablehlo.rsqrt
  // LAST: stablehlo.concatenate
  // LAST-NOT: stablehlo.power {{.*}} : tensor<f64>

  // TUPLE-LABEL: @power_neg_half_integer_exp
  // TUPLE: stablehlo.rsqrt
  // TUPLE-NOT: stablehlo.power {{.*}} : tensor<f64>

  %cst = stablehlo.constant dense<-1.5> : tensor<f64>
  %0 = stablehlo.power %arg0, %cst : tensor<f64>
  return %0 : tensor<f64>
}

// Dyadic (quarter) exponent, no integer part: pow(x, 0.75) = (sqrt(sqrt(x)))^3.
func.func @power_quarter_exp(%arg0: tensor<f64>) -> tensor<f64> {
  // FIRST-LABEL: @power_quarter_exp
  // FIRST: stablehlo.rsqrt
  // FIRST-NOT: stablehlo.power {{.*}} : tensor<f64>

  // LAST-LABEL: @power_quarter_exp
  // LAST: stablehlo.rsqrt
  // LAST-NOT: stablehlo.power {{.*}} : tensor<f64>

  // TUPLE-LABEL: @power_quarter_exp
  // TUPLE: stablehlo.rsqrt
  // TUPLE-NOT: stablehlo.power {{.*}} : tensor<f64>

  %cst = stablehlo.constant dense<7.500000e-01> : tensor<f64>
  %0 = stablehlo.power %arg0, %cst : tensor<f64>
  return %0 : tensor<f64>
}

// Dyadic (quarter) exponent with an integer part: pow(x, 2.25) = x^2 * x^0.25.
func.func @power_quarter_gt1_exp(%arg0: tensor<f64>) -> tensor<f64> {
  // FIRST-LABEL: @power_quarter_gt1_exp
  // FIRST: stablehlo.rsqrt
  // FIRST-NOT: stablehlo.power {{.*}} : tensor<f64>

  // LAST-LABEL: @power_quarter_gt1_exp
  // LAST: stablehlo.rsqrt
  // LAST-NOT: stablehlo.power {{.*}} : tensor<f64>

  // TUPLE-LABEL: @power_quarter_gt1_exp
  // TUPLE: stablehlo.rsqrt
  // TUPLE-NOT: stablehlo.power {{.*}} : tensor<f64>

  %cst = stablehlo.constant dense<2.250000e+00> : tensor<f64>
  %0 = stablehlo.power %arg0, %cst : tensor<f64>
  return %0 : tensor<f64>
}

// Cube-root rational exponent: pow(x, 1/3) = cbrt(x). Denominator 3 → one cbrt.
func.func @power_third_exp(%arg0: tensor<f64>) -> tensor<f64> {
  // FIRST-LABEL: @power_third_exp
  // FIRST: stablehlo.cbrt
  // FIRST-NOT: stablehlo.power {{.*}} : tensor<f64>

  // LAST-LABEL: @power_third_exp
  // LAST: stablehlo.cbrt
  // LAST-NOT: stablehlo.power {{.*}} : tensor<f64>

  // TUPLE-LABEL: @power_third_exp
  // TUPLE: stablehlo.cbrt
  // TUPLE-NOT: stablehlo.power {{.*}} : tensor<f64>

  %cst = stablehlo.constant dense<0.3333333333333333> : tensor<f64>
  %0 = stablehlo.power %arg0, %cst : tensor<f64>
  return %0 : tensor<f64>
}

// {2,3}-smooth exponent: pow(x, 1/6) = cbrt(sqrt(x)). Denominator 6 = 2*3 →
// one sqrt (rsqrt kernel) composed with one cbrt.
func.func @power_sixth_exp(%arg0: tensor<f64>) -> tensor<f64> {
  // FIRST-LABEL: @power_sixth_exp
  // FIRST: stablehlo.cbrt
  // FIRST-NOT: stablehlo.power {{.*}} : tensor<f64>

  // LAST-LABEL: @power_sixth_exp
  // LAST: stablehlo.cbrt
  // LAST-NOT: stablehlo.power {{.*}} : tensor<f64>

  // TUPLE-LABEL: @power_sixth_exp
  // TUPLE: stablehlo.cbrt
  // TUPLE-NOT: stablehlo.power {{.*}} : tensor<f64>

  %cst = stablehlo.constant dense<0.16666666666666666> : tensor<f64>
  %0 = stablehlo.power %arg0, %cst : tensor<f64>
  return %0 : tensor<f64>
}

// Constant base, runtime exponent: log(|base|) is folded to a constant on the
// host, so no multifloat log kernel is emitted. The kernel's distinctive
// bit-extraction op (stablehlo.shift_right_logical) must therefore be absent.
func.func @power_const_base(%arg0: tensor<f64>) -> tensor<f64> {
  // FIRST-LABEL: @power_const_base
  // FIRST-NOT: stablehlo.shift_right_logical
  // FIRST-NOT: stablehlo.power {{.*}} : tensor<f64>

  // LAST-LABEL: @power_const_base
  // LAST-NOT: stablehlo.shift_right_logical
  // LAST-NOT: stablehlo.power {{.*}} : tensor<f64>

  // TUPLE-LABEL: @power_const_base
  // TUPLE-NOT: stablehlo.shift_right_logical
  // TUPLE-NOT: stablehlo.power {{.*}} : tensor<f64>

  %cst = stablehlo.constant dense<2.0> : tensor<f64>
  %0 = stablehlo.power %cst, %arg0 : tensor<f64>
  return %0 : tensor<f64>
}

// FIRST-LABEL: func.func @main
// LAST-LABEL: func.func @main
// TUPLE-LABEL: func.func @main
func.func @main() attributes {enzyme.no_multifloat} {
  // pow(2.0, 3.0) = 8.0
  %c2 = stablehlo.constant dense<2.0> : tensor<f64>
  %c3 = stablehlo.constant dense<3.0> : tensor<f64>
  %e8 = stablehlo.constant dense<8.0> : tensor<f64>
  %r1 = func.call @power_test(%c2, %c3) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r1, %e8) {max_ulp_difference = 10 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  // pow(4.0, 0.5) = 2.0
  %c4 = stablehlo.constant dense<4.0> : tensor<f64>
  %c05 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
  %e2 = stablehlo.constant dense<2.0> : tensor<f64>
  %r2 = func.call @power_test(%c4, %c05) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r2, %e2) {max_ulp_difference = 10 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  // pow(1.0, 5.0) = 1.0 exactly (log(1) = 0, exp(0) = 1)
  %c1 = stablehlo.constant dense<1.0> : tensor<f64>
  %c5 = stablehlo.constant dense<5.0> : tensor<f64>
  %r3 = func.call @power_test(%c1, %c5) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r3, %c1) {max_ulp_difference = 0 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  // pow(x, 0.0) = 1.0 for x > 0 (exp(0 * log(x)) = exp(0) = 1)
  %r4 = func.call @power_zero_exponent(%c2) : (tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r4, %c1) {max_ulp_difference = 0 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  // pow(1.0, y) = 1.0 for any y (exp(y * log(1)) = exp(0) = 1)
  %r5 = func.call @power_one_base(%c3) : (tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r5, %c1) {max_ulp_difference = 0 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  // pow(-2.0, 2.0) = 4.0  (negative base, even integer exponent)
  %cn2 = stablehlo.constant dense<-2.0> : tensor<f64>
  %e4 = stablehlo.constant dense<4.0> : tensor<f64>
  %r6 = func.call @power_test(%cn2, %c2) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r6, %e4) {max_ulp_difference = 10 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  // pow(-2.0, 3.0) = -8.0  (negative base, odd integer exponent)
  %en8 = stablehlo.constant dense<-8.0> : tensor<f64>
  %r7 = func.call @power_test(%cn2, %c3) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r7, %en8) {max_ulp_difference = 10 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  // pow(0.0, 0.0) = 1.0  (IEEE 754: pow(x, 0) = 1 for all x)
  %c0 = stablehlo.constant dense<0.0> : tensor<f64>
  %r8 = func.call @power_test(%c0, %c0) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r8, %c1) {max_ulp_difference = 0 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  // pow(+inf, 0.0) = 1.0  (IEEE 754: pow(x, 0) = 1 for all x)
  %cinf = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
  %r9 = func.call @power_test(%cinf, %c0) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r9, %c1) {max_ulp_difference = 0 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  // pow(1.0, +inf) = 1.0  (IEEE 754: pow(1, y) = 1 for all y)
  %r10 = func.call @power_test(%c1, %cinf) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r10, %c1) {max_ulp_difference = 0 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  // pow(-1.0, +inf) = 1.0  (IEEE 754 special case)
  %cn1 = stablehlo.constant dense<-1.0> : tensor<f64>
  %r11 = func.call @power_test(%cn1, %cinf) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r11, %c1) {max_ulp_difference = 0 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  // pow(4.0, 3.5) = 4^3 * sqrt(4) = 64 * 2 = 128.0. Call the constant-exponent
  // function so the half-integer fast path (sqrt + squaring) is exercised;
  // power_test takes the exponent as an argument and would use the fallback.
  %e128 = stablehlo.constant dense<1.280000e+02> : tensor<f64>
  %r12 = func.call @power_half_integer_exp(%c4) : (tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r12, %e128) {max_ulp_difference = 10 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  // pow(4.0, -1.5) = 1 / (4 * sqrt(4)) = 1 / 8 = 0.125  (negative half-integer)
  %e0125 = stablehlo.constant dense<1.250000e-01> : tensor<f64>
  %r13 = func.call @power_neg_half_integer_exp(%c4) : (tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r13, %e0125) {max_ulp_difference = 10 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  // pow(2.0, 3.0) = 8.0 via the constant-base fallback (folded log(2), runtime
  // exponent). Exercises the exp(y * const) path with the host-folded log.
  %r14 = func.call @power_const_base(%c3) : (tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r14, %e8) {max_ulp_difference = 10 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  // pow(16.0, 0.75) = (16^(1/4))^3 = 2^3 = 8.0  (dyadic quarter, no integer part)
  %c16 = stablehlo.constant dense<1.600000e+01> : tensor<f64>
  %r15 = func.call @power_quarter_exp(%c16) : (tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r15, %e8) {max_ulp_difference = 10 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  // pow(16.0, 2.25) = 16^2 * 16^(1/4) = 256 * 2 = 512.0  (dyadic, integer part)
  %e512 = stablehlo.constant dense<5.120000e+02> : tensor<f64>
  %r16 = func.call @power_quarter_gt1_exp(%c16) : (tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r16, %e512) {max_ulp_difference = 10 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  // Infinite / zero base with a runtime exponent: handled directly, not via
  // exp/log (which would give NaN). These exercise the fallback overrides.
  %cninf = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>

  // pow(+inf, 2.0) = +inf
  %r17 = func.call @power_test(%cinf, %c2) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r17, %cinf) {max_ulp_difference = 0 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  // pow(-inf, 3.0) = -inf  (odd integer exponent → negated)
  %r18 = func.call @power_test(%cninf, %c3) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r18, %cninf) {max_ulp_difference = 0 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  // pow(0.0, 2.0) = +0
  %r19 = func.call @power_test(%c0, %c2) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r19, %c0) {max_ulp_difference = 0 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  // pow(+inf, -1.0) = +0
  %r20 = func.call @power_test(%cinf, %cn1) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r20, %c0) {max_ulp_difference = 0 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  // pow(64.0, 1/3) = cbrt(64) = 4.0  ({2,3}-smooth, cube-root fast path). A
  // single cbrt of a perfect cube is (near-)exact. (%e4 = 4.0 defined above.)
  %c64 = stablehlo.constant dense<6.400000e+01> : tensor<f64>
  %r21 = func.call @power_third_exp(%c64) : (tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r21, %e4) {max_ulp_difference = 10 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  // pow(729.0, 1/6) = cbrt(sqrt(729)) = cbrt(27) = 3.0  (sqrt∘cbrt composition).
  // Composing two double-double roots is ~48-bit accurate, i.e. a few tens of
  // f64 ulp — the loose bound still rejects a wrong decomposition (which would
  // be off by a huge margin: 27, 729, or NaN).
  %c729 = stablehlo.constant dense<7.290000e+02> : tensor<f64>
  %e3 = stablehlo.constant dense<3.000000e+00> : tensor<f64>
  %r22 = func.call @power_sixth_exp(%c729) : (tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r22, %e3) {max_ulp_difference = 50 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  return
}
