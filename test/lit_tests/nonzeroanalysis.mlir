// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=non_zero_compare_fold" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

/////////////////////////////////////////////////////////////////////////////
// Pattern gating: comparison direction, comparison type and operand shape.
/////////////////////////////////////////////////////////////////////////////

func.func @gate_ne_zero() -> tensor<2xi1> {
  %cst = stablehlo.constant dense<3.000000e+00> : tensor<2xf32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.compare NE, %cst, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %0 : tensor<2xi1>
}

// CHECK:  func.func @gate_ne_zero() -> tensor<2xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<2xi1>
// CHECK-NEXT:    return %c : tensor<2xi1>
// CHECK-NEXT:  }

func.func @gate_eq_zero() -> tensor<2xi1> {
  %cst = stablehlo.constant dense<3.000000e+00> : tensor<2xf32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.compare EQ, %cst, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %0 : tensor<2xi1>
}

// CHECK:  func.func @gate_eq_zero() -> tensor<2xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<false> : tensor<2xi1>
// CHECK-NEXT:    return %c : tensor<2xi1>
// CHECK-NEXT:  }

// The zero may appear on either side of the comparison.
func.func @gate_zero_on_lhs() -> tensor<2xi1> {
  %cst = stablehlo.constant dense<3.000000e+00> : tensor<2xf32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.compare NE, %zero, %cst, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %0 : tensor<2xi1>
}

// CHECK:  func.func @gate_zero_on_lhs() -> tensor<2xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<2xi1>
// CHECK-NEXT:    return %c : tensor<2xi1>
// CHECK-NEXT:  }

// Only EQ/NE say something about being zero.
func.func @gate_lt_not_folded() -> tensor<2xi1> {
  %cst = stablehlo.constant dense<3.000000e+00> : tensor<2xf32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.compare LT, %cst, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %0 : tensor<2xi1>
}

// CHECK:  func.func @gate_lt_not_folded() -> tensor<2xi1> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<3.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %0 = stablehlo.compare LT, %cst, %cst_0, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT:    return %0 : tensor<2xi1>
// CHECK-NEXT:  }

// TOTALORDER distinguishes +0.0 from -0.0, so the fold does not apply.
func.func @gate_totalorder_not_folded() -> tensor<2xi1> {
  %cst = stablehlo.constant dense<3.000000e+00> : tensor<2xf32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.compare NE, %cst, %zero, TOTALORDER : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %0 : tensor<2xi1>
}

// CHECK:  func.func @gate_totalorder_not_folded() -> tensor<2xi1> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<3.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %0 = stablehlo.compare NE, %cst, %cst_0, TOTALORDER : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT:    return %0 : tensor<2xi1>
// CHECK-NEXT:  }

// Neither side is zero, so there is nothing to fold.
func.func @gate_no_zero_operand() -> tensor<2xi1> {
  %cst = stablehlo.constant dense<3.000000e+00> : tensor<2xf32>
  %one = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
  %0 = stablehlo.compare NE, %cst, %one, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %0 : tensor<2xi1>
}

// CHECK:  func.func @gate_no_zero_operand() -> tensor<2xi1> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<3.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %0 = stablehlo.compare NE, %cst, %cst_0, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT:    return %0 : tensor<2xi1>
// CHECK-NEXT:  }

/////////////////////////////////////////////////////////////////////////////
// Values without a defining operation are never guaranteed.
/////////////////////////////////////////////////////////////////////////////

func.func @block_argument(%arg0: tensor<2xf32>) -> tensor<2xi1> {
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.compare NE, %arg0, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %0 : tensor<2xi1>
}

// CHECK:  func.func @block_argument(%arg0: tensor<2xf32>) -> tensor<2xi1> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %0 = stablehlo.compare NE, %arg0, %cst, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT:    return %0 : tensor<2xi1>
// CHECK-NEXT:  }

// stablehlo.multiply is not modelled by the analysis.
func.func @unhandled_op() -> tensor<2xi1> {
  %cst = stablehlo.constant dense<3.000000e+00> : tensor<2xf32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.multiply %cst, %cst : tensor<2xf32>
  %1 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %1 : tensor<2xi1>
}

// CHECK:  func.func @unhandled_op() -> tensor<2xi1> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<3.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %0 = stablehlo.multiply %cst, %cst {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2xf32>
// CHECK-NEXT:    %1 = stablehlo.compare NE, %0, %cst_0, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT:    return %1 : tensor<2xi1>
// CHECK-NEXT:  }

/////////////////////////////////////////////////////////////////////////////
// Constant element checks.
/////////////////////////////////////////////////////////////////////////////

func.func @const_int_nonzero() -> tensor<3xi1> {
  %cst = stablehlo.constant dense<[1, -2, 3]> : tensor<3xi32>
  %zero = stablehlo.constant dense<0> : tensor<3xi32>
  %0 = stablehlo.compare NE, %cst, %zero, SIGNED : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
  return %0 : tensor<3xi1>
}

// CHECK:  func.func @const_int_nonzero() -> tensor<3xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<3xi1>
// CHECK-NEXT:    return %c : tensor<3xi1>
// CHECK-NEXT:  }

func.func @const_int_has_zero() -> tensor<3xi1> {
  %cst = stablehlo.constant dense<[1, 0, 3]> : tensor<3xi32>
  %zero = stablehlo.constant dense<0> : tensor<3xi32>
  %0 = stablehlo.compare NE, %cst, %zero, SIGNED : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
  return %0 : tensor<3xi1>
}

// CHECK:  func.func @const_int_has_zero() -> tensor<3xi1> {
// CHECK-NEXT:    %c = stablehlo.constant {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} dense<[1, 0, 3]> : tensor<3xi32>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<3xi32>
// CHECK-NEXT:    %0 = stablehlo.compare NE, %c, %c_0, SIGNED : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
// CHECK-NEXT:    return %0 : tensor<3xi1>
// CHECK-NEXT:  }

func.func @const_float_nonzero() -> tensor<3xi1> {
  %cst = stablehlo.constant dense<[1.000000e+00, -2.000000e+00, 3.000000e+00]> : tensor<3xf32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<3xf32>
  %0 = stablehlo.compare NE, %cst, %zero, FLOAT : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
  return %0 : tensor<3xi1>
}

// CHECK:  func.func @const_float_nonzero() -> tensor<3xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<3xi1>
// CHECK-NEXT:    return %c : tensor<3xi1>
// CHECK-NEXT:  }

func.func @const_float_has_zero() -> tensor<3xi1> {
  %cst = stablehlo.constant dense<[1.000000e+00, 0.000000e+00, 3.000000e+00]> : tensor<3xf32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<3xf32>
  %0 = stablehlo.compare NE, %cst, %zero, FLOAT : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
  return %0 : tensor<3xi1>
}

// CHECK:  func.func @const_float_has_zero() -> tensor<3xi1> {
// CHECK-NEXT:    %cst = stablehlo.constant {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} dense<[1.000000e+00, 0.000000e+00, 3.000000e+00]> : tensor<3xf32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<3xf32>
// CHECK-NEXT:    %0 = stablehlo.compare NE, %cst, %cst_0, FLOAT : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
// CHECK-NEXT:    return %0 : tensor<3xi1>
// CHECK-NEXT:  }

// Negative zero is a zero as well.
func.func @const_float_negative_zero() -> tensor<2xi1> {
  %cst = stablehlo.constant dense<-0.000000e+00> : tensor<2xf32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.compare NE, %cst, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %0 : tensor<2xi1>
}

// CHECK:  func.func @const_float_negative_zero() -> tensor<2xi1> {
// CHECK-NEXT:    %cst = stablehlo.constant {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} dense<-0.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %0 = stablehlo.compare NE, %cst, %cst_0, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT:    return %0 : tensor<2xi1>
// CHECK-NEXT:  }

// Denormals may be flushed to zero, so they are conservatively rejected.
func.func @const_float_denormal() -> tensor<2xi1> {
  %cst = stablehlo.constant dense<1.000000e-320> : tensor<2xf64>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf64>
  %0 = stablehlo.compare NE, %cst, %zero, FLOAT : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xi1>
  return %0 : tensor<2xi1>
}

// CHECK:  func.func @const_float_denormal() -> tensor<2xi1> {
// CHECK-NEXT:    %cst = stablehlo.constant {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} dense<9.999880e-321> : tensor<2xf64>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<2xf64>
// CHECK-NEXT:    %0 = stablehlo.compare NE, %cst, %cst_0, FLOAT : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xi1>
// CHECK-NEXT:    return %0 : tensor<2xi1>
// CHECK-NEXT:  }

/////////////////////////////////////////////////////////////////////////////
// Integer range reasoning.
/////////////////////////////////////////////////////////////////////////////

// clamp bounds the result to [1, 10], which excludes zero.
func.func @int_range_positive(%arg0: tensor<4xi32>) -> tensor<4xi1> {
  %lo = stablehlo.constant dense<1> : tensor<4xi32>
  %hi = stablehlo.constant dense<10> : tensor<4xi32>
  %zero = stablehlo.constant dense<0> : tensor<4xi32>
  %0 = stablehlo.clamp %lo, %arg0, %hi : tensor<4xi32>
  %1 = stablehlo.compare NE, %0, %zero, SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %1 : tensor<4xi1>
}

// CHECK:  func.func @int_range_positive(%arg0: tensor<4xi32>) -> tensor<4xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<4xi1>
// CHECK-NEXT:    return %c : tensor<4xi1>
// CHECK-NEXT:  }

// ... and the same for a strictly negative range.
func.func @int_range_negative(%arg0: tensor<4xi32>) -> tensor<4xi1> {
  %lo = stablehlo.constant dense<-10> : tensor<4xi32>
  %hi = stablehlo.constant dense<-1> : tensor<4xi32>
  %zero = stablehlo.constant dense<0> : tensor<4xi32>
  %0 = stablehlo.clamp %lo, %arg0, %hi : tensor<4xi32>
  %1 = stablehlo.compare NE, %0, %zero, SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %1 : tensor<4xi1>
}

// CHECK:  func.func @int_range_negative(%arg0: tensor<4xi32>) -> tensor<4xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<4xi1>
// CHECK-NEXT:    return %c : tensor<4xi1>
// CHECK-NEXT:  }

// A range straddling zero proves nothing.
func.func @int_range_straddles_zero(%arg0: tensor<4xi32>) -> tensor<4xi1> {
  %lo = stablehlo.constant dense<-10> : tensor<4xi32>
  %hi = stablehlo.constant dense<10> : tensor<4xi32>
  %zero = stablehlo.constant dense<0> : tensor<4xi32>
  %0 = stablehlo.clamp %lo, %arg0, %hi : tensor<4xi32>
  %1 = stablehlo.compare NE, %0, %zero, SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %1 : tensor<4xi1>
}

// CHECK:  func.func @int_range_straddles_zero(%arg0: tensor<4xi32>) -> tensor<4xi1> {
// CHECK-NEXT:    %c = stablehlo.constant {enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>], enzymexla.non_zero = [#enzymexla<guaranteed GUARANTEED>]} dense<-10> : tensor<4xi32>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<10> : tensor<4xi32>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<4xi32>
// CHECK-NEXT:    %0 = stablehlo.clamp %c, %arg0, %c_0 {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<4xi32>
// CHECK-NEXT:    %1 = stablehlo.compare NE, %0, %c_1, SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
// CHECK-NEXT:    return %1 : tensor<4xi1>
// CHECK-NEXT:  }

// iota + 1 has a computed range that overflows the element type, so the range
// is discarded as unreliable (the addition may wrap around to zero).
func.func @int_range_may_wrap() -> tensor<8xi1> {
  %one = stablehlo.constant dense<1> : tensor<8xi64>
  %zero = stablehlo.constant dense<0> : tensor<8xi64>
  %0 = stablehlo.iota dim = 0 : tensor<8xi64>
  %1 = stablehlo.add %0, %one : tensor<8xi64>
  %2 = stablehlo.compare NE, %1, %zero, SIGNED : (tensor<8xi64>, tensor<8xi64>) -> tensor<8xi1>
  return %2 : tensor<8xi1>
}

// CHECK:  func.func @int_range_may_wrap() -> tensor<8xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<8xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<8xi64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<8xi64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %c {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<8xi64>
// CHECK-NEXT:    %2 = stablehlo.compare NE, %1, %c_0, SIGNED : (tensor<8xi64>, tensor<8xi64>) -> tensor<8xi1>
// CHECK-NEXT:    return %2 : tensor<8xi1>
// CHECK-NEXT:  }

// Bounds attached to the IR are picked up as well.
func.func @int_range_from_ir_bounds(%arg0: tensor<4xi32>) -> tensor<4xi1> {
  %zero = stablehlo.constant dense<0> : tensor<4xi32>
  %0 = stablehlo.abs %arg0 {enzymexla.bounds = [[5, 9]]} : tensor<4xi32>
  %1 = stablehlo.compare NE, %0, %zero, SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %1 : tensor<4xi1>
}

// CHECK:  func.func @int_range_from_ir_bounds(%arg0: tensor<4xi32>) -> tensor<4xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<4xi1>
// CHECK-NEXT:    return %c : tensor<4xi1>
// CHECK-NEXT:  }

/////////////////////////////////////////////////////////////////////////////
// Transcendental operations.
/////////////////////////////////////////////////////////////////////////////

// exp(x) >= 1 for x >= 0.
func.func @exp_of_nonnegative(%arg0: tensor<2xf32>) -> tensor<2xi1> {
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.abs %arg0 : tensor<2xf32>
  %1 = stablehlo.exponential %0 : tensor<2xf32>
  %2 = stablehlo.compare NE, %1, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %2 : tensor<2xi1>
}

// CHECK:  func.func @exp_of_nonnegative(%arg0: tensor<2xf32>) -> tensor<2xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<2xi1>
// CHECK-NEXT:    return %c : tensor<2xi1>
// CHECK-NEXT:  }

// exp of an unconstrained value can underflow to zero.
func.func @exp_of_unknown(%arg0: tensor<2xf32>) -> tensor<2xi1> {
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.exponential %arg0 : tensor<2xf32>
  %1 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %1 : tensor<2xi1>
}

// CHECK:  func.func @exp_of_unknown(%arg0: tensor<2xf32>) -> tensor<2xi1> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %0 = stablehlo.exponential %arg0 {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2xf32>
// CHECK-NEXT:    %1 = stablehlo.compare NE, %0, %cst, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT:    return %1 : tensor<2xi1>
// CHECK-NEXT:  }

// logistic(x) >= 0.5 for x >= 0.
func.func @logistic_of_nonnegative(%arg0: tensor<2xf32>) -> tensor<2xi1> {
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.abs %arg0 : tensor<2xf32>
  %1 = stablehlo.logistic %0 : tensor<2xf32>
  %2 = stablehlo.compare NE, %1, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %2 : tensor<2xi1>
}

// CHECK:  func.func @logistic_of_nonnegative(%arg0: tensor<2xf32>) -> tensor<2xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<2xi1>
// CHECK-NEXT:    return %c : tensor<2xi1>
// CHECK-NEXT:  }

func.func @logistic_of_unknown(%arg0: tensor<2xf32>) -> tensor<2xi1> {
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.logistic %arg0 : tensor<2xf32>
  %1 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %1 : tensor<2xi1>
}

// CHECK:  func.func @logistic_of_unknown(%arg0: tensor<2xf32>) -> tensor<2xi1> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %0 = stablehlo.logistic %arg0 {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2xf32>
// CHECK-NEXT:    %1 = stablehlo.compare NE, %0, %cst, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT:    return %1 : tensor<2xi1>
// CHECK-NEXT:  }

// cosh(x) >= 1 everywhere.
func.func @cosh_is_always_nonzero(%arg0: tensor<2xf32>) -> tensor<2xi1> {
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = chlo.cosh %arg0 : tensor<2xf32> -> tensor<2xf32>
  %1 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %1 : tensor<2xi1>
}

// CHECK:  func.func @cosh_is_always_nonzero(%arg0: tensor<2xf32>) -> tensor<2xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<2xi1>
// CHECK-NEXT:    return %c : tensor<2xi1>
// CHECK-NEXT:  }

// rsqrt of a finite value is either infinite (rsqrt(0)) or bounded away from
// zero; tanh is guaranteed finite.
func.func @rsqrt_of_finite(%arg0: tensor<2xf32>) -> tensor<2xi1> {
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.tanh %arg0 : tensor<2xf32>
  %1 = stablehlo.rsqrt %0 : tensor<2xf32>
  %2 = stablehlo.compare NE, %1, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %2 : tensor<2xi1>
}

// CHECK:  func.func @rsqrt_of_finite(%arg0: tensor<2xf32>) -> tensor<2xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<2xi1>
// CHECK-NEXT:    return %c : tensor<2xi1>
// CHECK-NEXT:  }

// rsqrt(inf) == 0, so an unconstrained operand proves nothing.
func.func @rsqrt_of_unknown(%arg0: tensor<2xf32>) -> tensor<2xi1> {
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.rsqrt %arg0 : tensor<2xf32>
  %1 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %1 : tensor<2xi1>
}

// CHECK:  func.func @rsqrt_of_unknown(%arg0: tensor<2xf32>) -> tensor<2xi1> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %0 = stablehlo.rsqrt %arg0 {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2xf32>
// CHECK-NEXT:    %1 = stablehlo.compare NE, %0, %cst, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT:    return %1 : tensor<2xi1>
// CHECK-NEXT:  }

/////////////////////////////////////////////////////////////////////////////
// Addition.
/////////////////////////////////////////////////////////////////////////////

// a >= 0 and b > 0 implies a + b > 0.
func.func @add_nonnegative_and_nonzero(%arg0: tensor<2xf32>) -> tensor<2xi1> {
  %one = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.abs %arg0 : tensor<2xf32>
  %1 = stablehlo.add %0, %one : tensor<2xf32>
  %2 = stablehlo.compare NE, %1, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %2 : tensor<2xi1>
}

// CHECK:  func.func @add_nonnegative_and_nonzero(%arg0: tensor<2xf32>) -> tensor<2xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<2xi1>
// CHECK-NEXT:    return %c : tensor<2xi1>
// CHECK-NEXT:  }

// Both operands are non-negative but both may be zero.
func.func @add_both_may_be_zero(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xi1> {
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.abs %arg0 : tensor<2xf32>
  %1 = stablehlo.abs %arg1 : tensor<2xf32>
  %2 = stablehlo.add %0, %1 : tensor<2xf32>
  %3 = stablehlo.compare NE, %2, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %3 : tensor<2xi1>
}

// CHECK:  func.func @add_both_may_be_zero(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xi1> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {enzymexla.non_negative = [#enzymexla<guaranteed GUARANTEED>], enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2xf32>
// CHECK-NEXT:    %1 = stablehlo.abs %arg1 {enzymexla.non_negative = [#enzymexla<guaranteed GUARANTEED>], enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2xf32>
// CHECK-NEXT:    %2 = stablehlo.add %0, %1 {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2xf32>
// CHECK-NEXT:    %3 = stablehlo.compare NE, %2, %cst, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT:    return %3 : tensor<2xi1>
// CHECK-NEXT:  }

// One operand is positive but the other one may be negative.
func.func @add_operand_may_be_negative(%arg0: tensor<2xf32>) -> tensor<2xi1> {
  %one = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.add %arg0, %one : tensor<2xf32>
  %1 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %1 : tensor<2xi1>
}

// CHECK:  func.func @add_operand_may_be_negative(%arg0: tensor<2xf32>) -> tensor<2xi1> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %0 = stablehlo.add %arg0, %cst {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2xf32>
// CHECK-NEXT:    %1 = stablehlo.compare NE, %0, %cst_0, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT:    return %1 : tensor<2xi1>
// CHECK-NEXT:  }

// Integer addition is left to the range analysis; here it may wrap around.
func.func @add_integer_not_folded(%arg0: tensor<4xi32>) -> tensor<4xi1> {
  %lo = stablehlo.constant dense<1> : tensor<4xi32>
  %hi = stablehlo.constant dense<10> : tensor<4xi32>
  %zero = stablehlo.constant dense<0> : tensor<4xi32>
  %0 = stablehlo.clamp %lo, %arg0, %hi : tensor<4xi32>
  %1 = stablehlo.add %0, %arg0 : tensor<4xi32>
  %2 = stablehlo.compare NE, %1, %zero, SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %2 : tensor<4xi1>
}

// CHECK:  func.func @add_integer_not_folded(%arg0: tensor<4xi32>) -> tensor<4xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<4xi32>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<10> : tensor<4xi32>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<4xi32>
// CHECK-NEXT:    %0 = stablehlo.clamp %c, %arg0, %c_0 : tensor<4xi32>
// CHECK-NEXT:    %1 = stablehlo.add %0, %arg0 {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<4xi32>
// CHECK-NEXT:    %2 = stablehlo.compare NE, %1, %c_1, SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
// CHECK-NEXT:    return %2 : tensor<4xi1>
// CHECK-NEXT:  }

/////////////////////////////////////////////////////////////////////////////
// Element-wise operations that preserve nonzero-ness.
/////////////////////////////////////////////////////////////////////////////

func.func @abs_neg_sign(%arg0: tensor<2xf32>) -> (tensor<2xi1>, tensor<2xi1>, tensor<2xi1>) {
  %cst = stablehlo.constant dense<3.000000e+00> : tensor<2xf32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.abs %cst : tensor<2xf32>
  %1 = stablehlo.negate %cst : tensor<2xf32>
  %2 = stablehlo.sign %cst : tensor<2xf32>
  %3 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  %4 = stablehlo.compare NE, %1, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  %5 = stablehlo.compare NE, %2, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %3, %4, %5 : tensor<2xi1>, tensor<2xi1>, tensor<2xi1>
}

// CHECK:  func.func @abs_neg_sign(%arg0: tensor<2xf32>) -> (tensor<2xi1>, tensor<2xi1>, tensor<2xi1>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<2xi1>
// CHECK-NEXT:    return %c, %c, %c : tensor<2xi1>, tensor<2xi1>, tensor<2xi1>
// CHECK-NEXT:  }

func.func @sqrt_cbrt_erf() -> (tensor<2xi1>, tensor<2xi1>, tensor<2xi1>) {
  %cst = stablehlo.constant dense<3.000000e+00> : tensor<2xf32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.sqrt %cst : tensor<2xf32>
  %1 = stablehlo.cbrt %cst : tensor<2xf32>
  %2 = chlo.erf %cst : tensor<2xf32> -> tensor<2xf32>
  %3 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  %4 = stablehlo.compare NE, %1, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  %5 = stablehlo.compare NE, %2, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %3, %4, %5 : tensor<2xi1>, tensor<2xi1>, tensor<2xi1>
}

// CHECK:  func.func @sqrt_cbrt_erf() -> (tensor<2xi1>, tensor<2xi1>, tensor<2xi1>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<2xi1>
// CHECK-NEXT:    return %c, %c, %c : tensor<2xi1>, tensor<2xi1>, tensor<2xi1>
// CHECK-NEXT:  }

func.func @sqrt_of_unknown(%arg0: tensor<2xf32>) -> tensor<2xi1> {
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.sqrt %arg0 : tensor<2xf32>
  %1 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %1 : tensor<2xi1>
}

// CHECK:  func.func @sqrt_of_unknown(%arg0: tensor<2xf32>) -> tensor<2xi1> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %0 = stablehlo.sqrt %arg0 {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2xf32>
// CHECK-NEXT:    %1 = stablehlo.compare NE, %0, %cst, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT:    return %1 : tensor<2xi1>
// CHECK-NEXT:  }

// popcnt(x) >= 1 for a nonzero x.
func.func @population_count(%arg0: tensor<4xi32>) -> tensor<4xi1> {
  %lo = stablehlo.constant dense<1> : tensor<4xi32>
  %hi = stablehlo.constant dense<10> : tensor<4xi32>
  %zero = stablehlo.constant dense<0> : tensor<4xi32>
  %0 = stablehlo.clamp %lo, %arg0, %hi : tensor<4xi32>
  %1 = stablehlo.popcnt %0 : tensor<4xi32>
  %2 = stablehlo.compare NE, %1, %zero, SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %2 : tensor<4xi1>
}

// CHECK:  func.func @population_count(%arg0: tensor<4xi32>) -> tensor<4xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<4xi1>
// CHECK-NEXT:    return %c : tensor<4xi1>
// CHECK-NEXT:  }

/////////////////////////////////////////////////////////////////////////////
// Data movement operations.
/////////////////////////////////////////////////////////////////////////////

func.func @reshape_transpose(%arg0: tensor<2x3xf32>) -> (tensor<6xi1>, tensor<3x2xi1>) {
  %cst = stablehlo.constant dense<3.000000e+00> : tensor<2x3xf32>
  %zero1 = stablehlo.constant dense<0.000000e+00> : tensor<6xf32>
  %zero2 = stablehlo.constant dense<0.000000e+00> : tensor<3x2xf32>
  %0 = stablehlo.reshape %cst : (tensor<2x3xf32>) -> tensor<6xf32>
  %1 = stablehlo.transpose %cst, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
  %2 = stablehlo.compare NE, %0, %zero1, FLOAT : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xi1>
  %3 = stablehlo.compare NE, %1, %zero2, FLOAT : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xi1>
  return %2, %3 : tensor<6xi1>, tensor<3x2xi1>
}

// CHECK:  func.func @reshape_transpose(%arg0: tensor<2x3xf32>) -> (tensor<6xi1>, tensor<3x2xi1>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<6xi1>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<true> : tensor<3x2xi1>
// CHECK-NEXT:    return %c, %c_0 : tensor<6xi1>, tensor<3x2xi1>
// CHECK-NEXT:  }

func.func @slice_reverse_broadcast() -> (tensor<2xi1>, tensor<4xi1>, tensor<3x4xi1>) {
  %cst = stablehlo.constant dense<[1.000000e+00, -2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>
  %zero2 = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %zero4 = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
  %zero34 = stablehlo.constant dense<0.000000e+00> : tensor<3x4xf32>
  %0 = stablehlo.slice %cst [0:2] : (tensor<4xf32>) -> tensor<2xf32>
  %1 = stablehlo.reverse %cst, dims = [0] : tensor<4xf32>
  %2 = stablehlo.broadcast_in_dim %cst, dims = [1] : (tensor<4xf32>) -> tensor<3x4xf32>
  %3 = stablehlo.compare NE, %0, %zero2, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  %4 = stablehlo.compare NE, %1, %zero4, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  %5 = stablehlo.compare NE, %2, %zero34, FLOAT : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xi1>
  return %3, %4, %5 : tensor<2xi1>, tensor<4xi1>, tensor<3x4xi1>
}

// CHECK:  func.func @slice_reverse_broadcast() -> (tensor<2xi1>, tensor<4xi1>, tensor<3x4xi1>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<2xi1>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<true> : tensor<4xi1>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<true> : tensor<3x4xi1>
// CHECK-NEXT:    return %c, %c_0, %c_1 : tensor<2xi1>, tensor<4xi1>, tensor<3x4xi1>
// CHECK-NEXT:  }

func.func @dynamic_slice(%arg0: tensor<i32>) -> tensor<2xi1> {
  %cst = stablehlo.constant dense<[1.000000e+00, -2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.dynamic_slice %cst, %arg0, sizes = [2] : (tensor<4xf32>, tensor<i32>) -> tensor<2xf32>
  %1 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %1 : tensor<2xi1>
}

// CHECK:  func.func @dynamic_slice(%arg0: tensor<i32>) -> tensor<2xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<2xi1>
// CHECK-NEXT:    return %c : tensor<2xi1>
// CHECK-NEXT:  }

// Data movement out of an unknown value proves nothing.
func.func @reshape_of_unknown(%arg0: tensor<2x3xf32>) -> tensor<6xi1> {
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<6xf32>
  %0 = stablehlo.reshape %arg0 : (tensor<2x3xf32>) -> tensor<6xf32>
  %1 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xi1>
  return %1 : tensor<6xi1>
}

// CHECK:  func.func @reshape_of_unknown(%arg0: tensor<2x3xf32>) -> tensor<6xi1> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<6xf32>
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<2x3xf32>) -> tensor<6xf32>
// CHECK-NEXT:    %1 = stablehlo.compare NE, %0, %cst, FLOAT : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xi1>
// CHECK-NEXT:    return %1 : tensor<6xi1>
// CHECK-NEXT:  }

/////////////////////////////////////////////////////////////////////////////
// Padding and concatenation: the introduced elements must be nonzero too.
/////////////////////////////////////////////////////////////////////////////

func.func @pad_nonzero_padding_value() -> tensor<4xi1> {
  %cst = stablehlo.constant dense<3.000000e+00> : tensor<2xf32>
  %pad = stablehlo.constant dense<7.000000e+00> : tensor<f32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
  %0 = stablehlo.pad %cst, %pad, low = [1], high = [1], interior = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<4xf32>
  %1 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %1 : tensor<4xi1>
}

// CHECK:  func.func @pad_nonzero_padding_value() -> tensor<4xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<4xi1>
// CHECK-NEXT:    return %c : tensor<4xi1>
// CHECK-NEXT:  }

func.func @pad_zero_padding_value() -> tensor<4xi1> {
  %cst = stablehlo.constant dense<3.000000e+00> : tensor<2xf32>
  %pad = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
  %0 = stablehlo.pad %cst, %pad, low = [1], high = [1], interior = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<4xf32>
  %1 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %1 : tensor<4xi1>
}

// CHECK:  func.func @pad_zero_padding_value() -> tensor<4xi1> {
// CHECK-NEXT:    %cst = stablehlo.constant {enzymexla.non_zero = [#enzymexla<guaranteed GUARANTEED>]} dense<3.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
// CHECK-NEXT:    %0 = stablehlo.pad %cst, %cst_0, low = [1], high = [1], interior = [0] {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<2xf32>, tensor<f32>) -> tensor<4xf32>
// CHECK-NEXT:    %1 = stablehlo.compare NE, %0, %cst_1, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
// CHECK-NEXT:    return %1 : tensor<4xi1>
// CHECK-NEXT:  }

func.func @dynamic_pad_nonzero_padding_value() -> tensor<4xi1> {
  %cst = stablehlo.constant dense<3.000000e+00> : tensor<2xf32>
  %pad = stablehlo.constant dense<7.000000e+00> : tensor<f32>
  %low = stablehlo.constant dense<1> : tensor<1xi64>
  %high = stablehlo.constant dense<1> : tensor<1xi64>
  %interior = stablehlo.constant dense<0> : tensor<1xi64>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
  %0 = stablehlo.dynamic_pad %cst, %pad, %low, %high, %interior : (tensor<2xf32>, tensor<f32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xf32>
  %1 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %1 : tensor<4xi1>
}

// CHECK:  func.func @dynamic_pad_nonzero_padding_value() -> tensor<4xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<4xi1>
// CHECK-NEXT:    return %c : tensor<4xi1>
// CHECK-NEXT:  }

func.func @dynamic_pad_zero_padding_value() -> tensor<4xi1> {
  %cst = stablehlo.constant dense<3.000000e+00> : tensor<2xf32>
  %pad = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %low = stablehlo.constant dense<1> : tensor<1xi64>
  %high = stablehlo.constant dense<1> : tensor<1xi64>
  %interior = stablehlo.constant dense<0> : tensor<1xi64>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
  %0 = stablehlo.dynamic_pad %cst, %pad, %low, %high, %interior : (tensor<2xf32>, tensor<f32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xf32>
  %1 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %1 : tensor<4xi1>
}

// CHECK:  func.func @dynamic_pad_zero_padding_value() -> tensor<4xi1> {
// CHECK-NEXT:    %cst = stablehlo.constant {enzymexla.non_zero = [#enzymexla<guaranteed GUARANTEED>]} dense<3.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
// CHECK-NEXT:    %0 = stablehlo.dynamic_pad %cst, %cst_0, %c, %c, %c_1 {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<2xf32>, tensor<f32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xf32>
// CHECK-NEXT:    %1 = stablehlo.compare NE, %0, %cst_2, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
// CHECK-NEXT:    return %1 : tensor<4xi1>
// CHECK-NEXT:  }

func.func @concatenate_all_nonzero() -> tensor<4xi1> {
  %a = stablehlo.constant dense<3.000000e+00> : tensor<2xf32>
  %b = stablehlo.constant dense<-1.000000e+00> : tensor<2xf32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
  %0 = stablehlo.concatenate %a, %b, dim = 0 : (tensor<2xf32>, tensor<2xf32>) -> tensor<4xf32>
  %1 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %1 : tensor<4xi1>
}

// CHECK:  func.func @concatenate_all_nonzero() -> tensor<4xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<4xi1>
// CHECK-NEXT:    return %c : tensor<4xi1>
// CHECK-NEXT:  }

func.func @concatenate_one_unknown(%arg0: tensor<2xf32>) -> tensor<4xi1> {
  %a = stablehlo.constant dense<3.000000e+00> : tensor<2xf32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
  %0 = stablehlo.concatenate %a, %arg0, dim = 0 : (tensor<2xf32>, tensor<2xf32>) -> tensor<4xf32>
  %1 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %1 : tensor<4xi1>
}

// CHECK:  func.func @concatenate_one_unknown(%arg0: tensor<2xf32>) -> tensor<4xi1> {
// CHECK-NEXT:    %cst = stablehlo.constant {enzymexla.non_zero = [#enzymexla<guaranteed GUARANTEED>]} dense<3.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
// CHECK-NEXT:    %0 = stablehlo.concatenate %cst, %arg0, dim = 0 {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<2xf32>, tensor<2xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %1 = stablehlo.compare NE, %0, %cst_0, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
// CHECK-NEXT:    return %1 : tensor<4xi1>
// CHECK-NEXT:  }

/////////////////////////////////////////////////////////////////////////////
// max / min / select / clamp.
/////////////////////////////////////////////////////////////////////////////

// max(x, c) >= c > 0 regardless of x.
func.func @max_with_positive_constant(%arg0: tensor<2xf32>) -> tensor<2xi1> {
  %cst = stablehlo.constant dense<2.000000e+00> : tensor<2xf32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.maximum %arg0, %cst : tensor<2xf32>
  %1 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %1 : tensor<2xi1>
}

// CHECK:  func.func @max_with_positive_constant(%arg0: tensor<2xf32>) -> tensor<2xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<2xi1>
// CHECK-NEXT:    return %c : tensor<2xi1>
// CHECK-NEXT:  }

// The result is one of the two operands, and both are nonzero.
func.func @max_both_nonzero() -> tensor<2xi1> {
  %a = stablehlo.constant dense<-3.000000e+00> : tensor<2xf32>
  %b = stablehlo.constant dense<-1.000000e+00> : tensor<2xf32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.maximum %a, %b : tensor<2xf32>
  %1 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %1 : tensor<2xi1>
}

// CHECK:  func.func @max_both_nonzero() -> tensor<2xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<2xi1>
// CHECK-NEXT:    return %c : tensor<2xi1>
// CHECK-NEXT:  }

// max(x, -1) may well be zero.
func.func @max_with_negative_constant(%arg0: tensor<2xf32>) -> tensor<2xi1> {
  %cst = stablehlo.constant dense<-1.000000e+00> : tensor<2xf32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.maximum %arg0, %cst : tensor<2xf32>
  %1 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %1 : tensor<2xi1>
}

// CHECK:  func.func @max_with_negative_constant(%arg0: tensor<2xf32>) -> tensor<2xi1> {
// CHECK-NEXT:    %cst = stablehlo.constant {enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} dense<-1.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %0 = stablehlo.maximum %arg0, %cst {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2xf32>
// CHECK-NEXT:    %1 = stablehlo.compare NE, %0, %cst_0, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT:    return %1 : tensor<2xi1>
// CHECK-NEXT:  }

func.func @min_both_nonzero() -> tensor<2xi1> {
  %a = stablehlo.constant dense<-3.000000e+00> : tensor<2xf32>
  %b = stablehlo.constant dense<2.000000e+00> : tensor<2xf32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.minimum %a, %b : tensor<2xf32>
  %1 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %1 : tensor<2xi1>
}

// CHECK:  func.func @min_both_nonzero() -> tensor<2xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<2xi1>
// CHECK-NEXT:    return %c : tensor<2xi1>
// CHECK-NEXT:  }

func.func @min_one_unknown(%arg0: tensor<2xf32>) -> tensor<2xi1> {
  %a = stablehlo.constant dense<-3.000000e+00> : tensor<2xf32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.minimum %a, %arg0 : tensor<2xf32>
  %1 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %1 : tensor<2xi1>
}

// CHECK:  func.func @min_one_unknown(%arg0: tensor<2xf32>) -> tensor<2xi1> {
// CHECK-NEXT:    %cst = stablehlo.constant {enzymexla.non_zero = [#enzymexla<guaranteed GUARANTEED>]} dense<-3.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %0 = stablehlo.minimum %cst, %arg0 {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2xf32>
// CHECK-NEXT:    %1 = stablehlo.compare NE, %0, %cst_0, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT:    return %1 : tensor<2xi1>
// CHECK-NEXT:  }

// The predicate is irrelevant; both branches are nonzero.
func.func @select_both_nonzero(%pred: tensor<2xi1>) -> tensor<2xi1> {
  %a = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
  %b = stablehlo.constant dense<2.000000e+00> : tensor<2xf32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.select %pred, %a, %b : tensor<2xi1>, tensor<2xf32>
  %1 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %1 : tensor<2xi1>
}

// CHECK:  func.func @select_both_nonzero(%arg0: tensor<2xi1>) -> tensor<2xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<2xi1>
// CHECK-NEXT:    return %c : tensor<2xi1>
// CHECK-NEXT:  }

func.func @select_one_zero(%pred: tensor<2xi1>) -> tensor<2xi1> {
  %a = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
  %b = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.select %pred, %a, %b : tensor<2xi1>, tensor<2xf32>
  %1 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %1 : tensor<2xi1>
}

// CHECK:  func.func @select_one_zero(%arg0: tensor<2xi1>) -> tensor<2xi1> {
// CHECK-NEXT:    %cst = stablehlo.constant {enzymexla.non_zero = [#enzymexla<guaranteed GUARANTEED>]} dense<1.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} dense<0.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %0 = stablehlo.select %arg0, %cst, %cst_0 {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2xi1>, tensor<2xf32>
// CHECK-NEXT:    %1 = stablehlo.compare NE, %0, %cst_0, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT:    return %1 : tensor<2xi1>
// CHECK-NEXT:  }

// lo > 0 and hi != 0: the result is at least lo, or hi if the bounds cross.
func.func @clamp_positive_lower_bound(%arg0: tensor<2xf32>) -> tensor<2xi1> {
  %lo = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
  %hi = stablehlo.constant dense<5.000000e+00> : tensor<2xf32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.clamp %lo, %arg0, %hi : tensor<2xf32>
  %1 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %1 : tensor<2xi1>
}

// CHECK:  func.func @clamp_positive_lower_bound(%arg0: tensor<2xf32>) -> tensor<2xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<2xi1>
// CHECK-NEXT:    return %c : tensor<2xi1>
// CHECK-NEXT:  }

// A negative lower bound needs all three operands to be nonzero.
func.func @clamp_all_operands_nonzero(%arg0: tensor<2xf32>) -> tensor<2xi1> {
  %lo = stablehlo.constant dense<-5.000000e+00> : tensor<2xf32>
  %hi = stablehlo.constant dense<-1.000000e+00> : tensor<2xf32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = chlo.cosh %arg0 : tensor<2xf32> -> tensor<2xf32>
  %1 = stablehlo.clamp %lo, %0, %hi : tensor<2xf32>
  %2 = stablehlo.compare NE, %1, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %2 : tensor<2xi1>
}

// CHECK:  func.func @clamp_all_operands_nonzero(%arg0: tensor<2xf32>) -> tensor<2xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<2xi1>
// CHECK-NEXT:    return %c : tensor<2xi1>
// CHECK-NEXT:  }

func.func @clamp_operand_unknown(%arg0: tensor<2xf32>) -> tensor<2xi1> {
  %lo = stablehlo.constant dense<-5.000000e+00> : tensor<2xf32>
  %hi = stablehlo.constant dense<5.000000e+00> : tensor<2xf32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.clamp %lo, %arg0, %hi : tensor<2xf32>
  %1 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %1 : tensor<2xi1>
}

// CHECK:  func.func @clamp_operand_unknown(%arg0: tensor<2xf32>) -> tensor<2xi1> {
// CHECK-NEXT:    %cst = stablehlo.constant {enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>], enzymexla.non_zero = [#enzymexla<guaranteed GUARANTEED>]} dense<-5.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<5.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %0 = stablehlo.clamp %cst, %arg0, %cst_0 {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2xf32>
// CHECK-NEXT:    %1 = stablehlo.compare NE, %0, %cst_1, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT:    return %1 : tensor<2xi1>
// CHECK-NEXT:  }

/////////////////////////////////////////////////////////////////////////////
// stablehlo.convert.
/////////////////////////////////////////////////////////////////////////////

// A conversion to i1 is exactly (x != 0).
func.func @convert_int_to_i1() -> tensor<2xi1> {
  %cst = stablehlo.constant dense<[1, -2]> : tensor<2xi32>
  %zero = stablehlo.constant dense<false> : tensor<2xi1>
  %0 = stablehlo.convert %cst : (tensor<2xi32>) -> tensor<2xi1>
  %1 = stablehlo.compare NE, %0, %zero, UNSIGNED : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  return %1 : tensor<2xi1>
}

// CHECK:  func.func @convert_int_to_i1() -> tensor<2xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<2xi1>
// CHECK-NEXT:    return %c : tensor<2xi1>
// CHECK-NEXT:  }

func.func @convert_int_widening() -> tensor<2xi1> {
  %cst = stablehlo.constant dense<[1, -2]> : tensor<2xi8>
  %zero = stablehlo.constant dense<0> : tensor<2xi32>
  %0 = stablehlo.convert %cst : (tensor<2xi8>) -> tensor<2xi32>
  %1 = stablehlo.compare NE, %0, %zero, SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  return %1 : tensor<2xi1>
}

// CHECK:  func.func @convert_int_widening() -> tensor<2xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<2xi1>
// CHECK-NEXT:    return %c : tensor<2xi1>
// CHECK-NEXT:  }

// Truncation can drop all the set bits (256 -> 0).
func.func @convert_int_narrowing() -> tensor<2xi1> {
  %cst = stablehlo.constant dense<[1, -2]> : tensor<2xi32>
  %zero = stablehlo.constant dense<0> : tensor<2xi8>
  %0 = stablehlo.convert %cst : (tensor<2xi32>) -> tensor<2xi8>
  %1 = stablehlo.compare NE, %0, %zero, SIGNED : (tensor<2xi8>, tensor<2xi8>) -> tensor<2xi1>
  return %1 : tensor<2xi1>
}

// CHECK:  func.func @convert_int_narrowing() -> tensor<2xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<[1, -2]> : tensor<2xi32>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<2xi8>
// CHECK-NEXT:    %0 = stablehlo.convert %c {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<2xi32>) -> tensor<2xi8>
// CHECK-NEXT:    %1 = stablehlo.compare NE, %0, %c_0, SIGNED : (tensor<2xi8>, tensor<2xi8>) -> tensor<2xi1>
// CHECK-NEXT:    return %1 : tensor<2xi1>
// CHECK-NEXT:  }

// A nonzero integer maps to a float of magnitude at least one.
func.func @convert_int_to_float() -> tensor<2xi1> {
  %cst = stablehlo.constant dense<[1, -2]> : tensor<2xi32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.convert %cst : (tensor<2xi32>) -> tensor<2xf32>
  %1 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %1 : tensor<2xi1>
}

// CHECK:  func.func @convert_int_to_float() -> tensor<2xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<2xi1>
// CHECK-NEXT:    return %c : tensor<2xi1>
// CHECK-NEXT:  }

// Widening the exponent range cannot underflow.
func.func @convert_float_widening() -> tensor<2xi1> {
  %cst = stablehlo.constant dense<[1.000000e+00, -2.000000e+00]> : tensor<2xf32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf64>
  %0 = stablehlo.convert %cst : (tensor<2xf32>) -> tensor<2xf64>
  %1 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xi1>
  return %1 : tensor<2xi1>
}

// CHECK:  func.func @convert_float_widening() -> tensor<2xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<2xi1>
// CHECK-NEXT:    return %c : tensor<2xi1>
// CHECK-NEXT:  }

// Narrowing the exponent range may underflow to zero (1e-300 in f64 -> f32).
func.func @convert_float_narrowing() -> tensor<2xi1> {
  %cst = stablehlo.constant dense<[1.000000e+00, -2.000000e+00]> : tensor<2xf64>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.convert %cst : (tensor<2xf64>) -> tensor<2xf32>
  %1 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %1 : tensor<2xi1>
}

// CHECK:  func.func @convert_float_narrowing() -> tensor<2xi1> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<[1.000000e+00, -2.000000e+00]> : tensor<2xf64>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %0 = stablehlo.convert %cst {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<2xf64>) -> tensor<2xf32>
// CHECK-NEXT:    %1 = stablehlo.compare NE, %0, %cst_0, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT:    return %1 : tensor<2xi1>
// CHECK-NEXT:  }

// Truncation towards zero: 0.5 converts to 0.
func.func @convert_float_to_int() -> tensor<2xi1> {
  %cst = stablehlo.constant dense<[5.000000e-01, -2.000000e+00]> : tensor<2xf32>
  %zero = stablehlo.constant dense<0> : tensor<2xi32>
  %0 = stablehlo.convert %cst : (tensor<2xf32>) -> tensor<2xi32>
  %1 = stablehlo.compare NE, %0, %zero, SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  return %1 : tensor<2xi1>
}

// CHECK:  func.func @convert_float_to_int() -> tensor<2xi1> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<[5.000000e-01, -2.000000e+00]> : tensor<2xf32>
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<2xi32>
// CHECK-NEXT:    %0 = stablehlo.convert %cst {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<2xf32>) -> tensor<2xi32>
// CHECK-NEXT:    %1 = stablehlo.compare NE, %0, %c, SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
// CHECK-NEXT:    return %1 : tensor<2xi1>
// CHECK-NEXT:  }

/////////////////////////////////////////////////////////////////////////////
// Complex element types bail out early.
/////////////////////////////////////////////////////////////////////////////

func.func @complex_constant() -> tensor<2xi1> {
  %cst = stablehlo.constant dense<(1.000000e+00,2.000000e+00)> : tensor<2xcomplex<f32>>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.abs %cst : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
  %1 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %1 : tensor<2xi1>
}

// CHECK:  func.func @complex_constant() -> tensor<2xi1> {
// CHECK-NEXT:    %cst = stablehlo.constant {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} dense<(1.000000e+00,2.000000e+00)> : tensor<2xcomplex<f32>>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %0 = stablehlo.abs %cst {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// CHECK-NEXT:    %1 = stablehlo.compare NE, %0, %cst_0, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT:    return %1 : tensor<2xi1>
// CHECK-NEXT:  }

func.func @complex_bails_out(%arg0: tensor<2xcomplex<f32>>) -> tensor<2xi1> {
  %re = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
  %zerof = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %cst = stablehlo.complex %re, %zerof : tensor<2xcomplex<f32>>
  %0 = stablehlo.abs %cst : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
  %1 = stablehlo.compare NE, %0, %zerof, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %1 : tensor<2xi1>
}

// CHECK:  func.func @complex_bails_out(%arg0: tensor<2xcomplex<f32>>) -> tensor<2xi1> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %0 = stablehlo.complex %cst, %cst_0 {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2xcomplex<f32>>
// CHECK-NEXT:    %1 = stablehlo.abs %0 {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// CHECK-NEXT:    %2 = stablehlo.compare NE, %1, %cst_0, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT:    return %2 : tensor<2xi1>
// CHECK-NEXT:  }

/////////////////////////////////////////////////////////////////////////////
// Results already recorded in the IR are trusted as-is.
/////////////////////////////////////////////////////////////////////////////

func.func @preexisting_guaranteed_attr(%arg0: tensor<2xf32>) -> tensor<2xi1> {
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = stablehlo.multiply %arg0, %arg0 {enzymexla.non_zero = [#enzymexla<guaranteed GUARANTEED>]} : tensor<2xf32>
  %1 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %1 : tensor<2xi1>
}

// CHECK:  func.func @preexisting_guaranteed_attr(%arg0: tensor<2xf32>) -> tensor<2xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<2xi1>
// CHECK-NEXT:    return %c : tensor<2xi1>
// CHECK-NEXT:  }

func.func @preexisting_notguaranteed_attr(%arg0: tensor<2xf32>) -> tensor<2xi1> {
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = chlo.cosh %arg0 {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2xf32> -> tensor<2xf32>
  %1 = stablehlo.compare NE, %0, %zero, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %1 : tensor<2xi1>
}

// CHECK:  func.func @preexisting_notguaranteed_attr(%arg0: tensor<2xf32>) -> tensor<2xi1> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %0 = chlo.cosh %arg0 {enzymexla.non_zero = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2xf32> -> tensor<2xf32>
// CHECK-NEXT:    %1 = stablehlo.compare NE, %0, %cst, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT:    return %1 : tensor<2xi1>
// CHECK-NEXT:  }

/////////////////////////////////////////////////////////////////////////////
// A deeper chain, exercising the pending/worklist bookkeeping.
/////////////////////////////////////////////////////////////////////////////

func.func @deep_chain(%arg0: tensor<2xf32>, %pred: tensor<4xi1>) -> tensor<4xi1> {
  %one = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
  %two = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
  %0 = stablehlo.abs %arg0 : tensor<2xf32>
  %1 = stablehlo.add %0, %one : tensor<2xf32>
  %2 = stablehlo.concatenate %1, %1, dim = 0 : (tensor<2xf32>, tensor<2xf32>) -> tensor<4xf32>
  %3 = stablehlo.select %pred, %2, %two : tensor<4xi1>, tensor<4xf32>
  %4 = stablehlo.negate %3 : tensor<4xf32>
  %5 = stablehlo.compare NE, %4, %zero, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %5 : tensor<4xi1>
}

// CHECK:  func.func @deep_chain(%arg0: tensor<2xf32>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<4xi1>
// CHECK-NEXT:    return %c : tensor<4xi1>
// CHECK-NEXT:  }
