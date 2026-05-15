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

  return
}
