// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=log_simplify;log_const_prop},transform-interpreter,enzyme-hlo-remove-transform)" | FileCheck %s

func.func @main1(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.exponential %arg0 : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @main1(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     return %arg0 : tensor<f64>
// CHECK-NEXT: }

// A zero constant must not split: for a negative argument, the original is
// log(-0) = -inf while log(0) + log(argument) is NaN.
func.func @main_zero_mul(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.multiply %cst, %arg0 : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @main_zero_mul(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.multiply %cst, %arg0 : tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.log %0 : tensor<f64>
// CHECK-NEXT:     return %1 : tensor<f64>
// CHECK-NEXT: }

// A zero numerator must not split for the same signed-zero reason.
func.func @main_zero_div(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.divide %cst, %arg0 : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @main_zero_div(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.divide %cst, %arg0 : tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.log %0 : tensor<f64>
// CHECK-NEXT:     return %1 : tensor<f64>
// CHECK-NEXT: }

// The power rewrite is disabled: zero bases and infinite exponents can change
// finite results into NaNs (e.g. pow(0, 0) and pow(1, inf)).
func.func @main_pow_zero(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.power %cst, %arg0 : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @main_pow_zero(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.power %cst, %arg0 : tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.log %0 : tensor<f64>
// CHECK-NEXT:     return %1 : tensor<f64>
// CHECK-NEXT: }

func.func @main_pow_positive(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %0 = stablehlo.power %cst, %arg0 : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @main_pow_positive(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.power %cst, %arg0 : tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.log %0 : tensor<f64>
// CHECK-NEXT:     return %1 : tensor<f64>
// CHECK-NEXT: }

func.func @main2(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
    %1 = stablehlo.multiply %0, %0 : tensor<f64>
    %2 = stablehlo.log %1 : tensor<f64>
    return %2 : tensor<f64>
}

// arg0 * arg0 is provably non-negative, so log(mul(sq, sq)) folds to
// 2 * log(sq) without inserting an abs.
// CHECK: func.func @main2(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.multiply %arg0, %arg0 {enzymexla.non_negative = [#enzymexla<guaranteed GUARANTEED>]} : tensor<f64>
// CHECK-NEXT:    %1 = stablehlo.log %0 : tensor<f64>
// CHECK-NEXT:    %2 = stablehlo.multiply %cst, %1 : tensor<f64>
// CHECK-NEXT:    return %2 : tensor<f64>
// CHECK-NEXT:  }

func.func @main3(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<4.000000e+00> : tensor<f64>
    %0 = stablehlo.multiply %arg0, %cst : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @main3(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.3862943611198906> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.log %arg0 : tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.add %0, %cst : tensor<f64>
// CHECK-NEXT:     return %1 : tensor<f64>
// CHECK-NEXT: }

func.func @main4(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<4.000000e+00> : tensor<f64>
    %0 = stablehlo.multiply %cst, %arg0 : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @main4(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.3862943611198906> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.log %arg0 : tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.add %cst, %0 : tensor<f64>
// CHECK-NEXT:     return %1 : tensor<f64>
// CHECK-NEXT: }

func.func @main5(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.add %arg0, %arg0 : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @main5(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.69314718055994529> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.log %arg0 : tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.add %cst, %0 : tensor<f64>
// CHECK-NEXT:     return %1 : tensor<f64>
// CHECK-NEXT: }

func.func @main6(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// arg0 is not provably non-negative, so log(arg0 * arg0) is left untouched
// rather than folded to a NaN-producing 2 * log(arg0).
// CHECK: func.func @main6(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %0 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.log %0 : tensor<f64>
// CHECK-NEXT:     return %1 : tensor<f64>
// CHECK-NEXT: }

func.func @main7(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<4.000000e+00> : tensor<f64>
    %0 = stablehlo.divide %arg0, %cst : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @main7(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.3862943611198906> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.log %arg0 : tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.subtract %0, %cst : tensor<f64>
// CHECK-NEXT:     return %1 : tensor<f64>
// CHECK-NEXT: }

func.func @main8(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<4.000000e+00> : tensor<f64>
    %0 = stablehlo.divide %cst, %arg0 : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @main8(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.3862943611198906> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.log %arg0 : tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.subtract %cst, %0 : tensor<f64>
// CHECK-NEXT:     return %1 : tensor<f64>
// CHECK-NEXT: }

func.func @main9(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.sqrt %arg0 : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @main9(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.log %arg0 : tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.divide %0, %cst : tensor<f64>
// CHECK-NEXT:     return %1 : tensor<f64>
// CHECK-NEXT: }

func.func @main10(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.rsqrt %arg0 : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @main10(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<-2.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.log %arg0 : tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.divide %0, %cst : tensor<f64>
// CHECK-NEXT:     return %1 : tensor<f64>
// CHECK-NEXT: }

func.func @main11(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.cbrt %arg0 : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @main11(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<3.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.log %arg0 : tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.divide %0, %cst : tensor<f64>
// CHECK-NEXT:     return %1 : tensor<f64>
// CHECK-NEXT: }

// Domain guards (issue #2570): the split rewrites must NOT fire when they would
// narrow the domain and manufacture a NaN on inputs the original handles.

// log(a * c) with a negative constant must stay as-is: splitting to
// log(a) + log(c) makes log(c) NaN where log(a*c) is a finite real for a < 0.
func.func @main_neg_mul(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<-4.000000e+00> : tensor<f64>
    %0 = stablehlo.multiply %arg0, %cst : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @main_neg_mul(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<-4.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.multiply %arg0, %cst : tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.log %0 : tensor<f64>
// CHECK-NEXT:     return %1 : tensor<f64>
// CHECK-NEXT: }

// log(a / c) with a negative constant must stay as-is for the same reason.
func.func @main_neg_div(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<-4.000000e+00> : tensor<f64>
    %0 = stablehlo.divide %arg0, %cst : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @main_neg_div(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<-4.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.divide %arg0, %cst : tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.log %0 : tensor<f64>
// CHECK-NEXT:     return %1 : tensor<f64>
// CHECK-NEXT: }

// log(a * c) with an infinite constant must stay as-is: log(x * inf) is a
// finite -inf/NaN that splitting to log(x) + log(inf) would not preserve.
func.func @main_inf_mul(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
    %0 = stablehlo.multiply %arg0, %cst : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @main_inf_mul(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.multiply %arg0, %cst : tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.log %0 : tensor<f64>
// CHECK-NEXT:     return %1 : tensor<f64>
// CHECK-NEXT: }

// log(pow(x, y)) must stay as-is when x is not provably non-negative: for x < 0,
// log(pow(x, y)) can be a finite real while y * log(x) is NaN.
func.func @main_pow_unknown(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.power %arg0, %arg1 : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @main_pow_unknown(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %0 = stablehlo.power %arg0, %arg1 : tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.log %0 : tensor<f64>
// CHECK-NEXT:     return %1 : tensor<f64>
// CHECK-NEXT: }

// abs(arg0) is provably non-negative, so log(mul(a, a)) folds to 2 * log(a)
// directly, with no extra abs inserted.
func.func @main_square_nonneg(%arg0: tensor<f64>) -> tensor<f64> {
    %a = stablehlo.abs %arg0 : tensor<f64>
    %0 = stablehlo.multiply %a, %a : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @main_square_nonneg(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.abs %arg0 : tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.log %0 : tensor<f64>
// CHECK-NEXT:     %2 = stablehlo.multiply %cst, %1 : tensor<f64>
// CHECK-NEXT:     return %2 : tensor<f64>
// CHECK-NEXT: }
