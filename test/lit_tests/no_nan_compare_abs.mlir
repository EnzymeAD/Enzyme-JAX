// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{no_nan=true})" %s | FileCheck %s --check-prefix=NONAN
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{no_nan=false})" %s | FileCheck %s --check-prefix=NAN

module {
  // 0 < abs(x), i.e. abs(x) > 0 -> x != 0. Wrong for a NaN x, where the
  // original is false and x != 0 is true, so this needs no_nan.
  func.func @main(%2861:  tensor<3056xf64>) -> tensor<3056xi1> {
    %cst_116 = stablehlo.constant dense<0.000000e+00> : tensor<3056xf64>
    %2863 = stablehlo.abs %2861 : tensor<3056xf64>
    %2864 = stablehlo.compare  LT, %cst_116, %2863,  FLOAT : (tensor<3056xf64>, tensor<3056xf64>) -> tensor<3056xi1>
    return %2864 : tensor<3056xi1>
  }

  // abs(x) >= 0 -> true. Wrong for a NaN x, where it is false, so this needs
  // no_nan as well.
  func.func @ge(%arg0: tensor<4xf64>) -> tensor<4xi1> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
    %0 = stablehlo.abs %arg0 : tensor<4xf64>
    %1 = stablehlo.compare  GE, %0, %cst,  FLOAT : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
    return %1 : tensor<4xi1>
  }

  // The remaining directions agree with NaN semantics and fold unconditionally.

  // abs(x) < 0 -> false, and abs(NaN) < 0 is false too.
  func.func @lt(%arg0: tensor<4xf64>) -> tensor<4xi1> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
    %0 = stablehlo.abs %arg0 : tensor<4xf64>
    %1 = stablehlo.compare  LT, %0, %cst,  FLOAT : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
    return %1 : tensor<4xi1>
  }

  // abs(x) <= 0 -> x == 0; both sides are false for a NaN x.
  func.func @le(%arg0: tensor<4xf64>) -> tensor<4xi1> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
    %0 = stablehlo.abs %arg0 : tensor<4xf64>
    %1 = stablehlo.compare  LE, %0, %cst,  FLOAT : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
    return %1 : tensor<4xi1>
  }

  // abs(x) == 0 -> x == 0; both sides are false for a NaN x.
  func.func @eq(%arg0: tensor<4xf64>) -> tensor<4xi1> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
    %0 = stablehlo.abs %arg0 : tensor<4xf64>
    %1 = stablehlo.compare  EQ, %0, %cst,  FLOAT : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
    return %1 : tensor<4xi1>
  }

  // abs(x) != 0 -> x != 0; both sides are true for a NaN x.
  func.func @ne(%arg0: tensor<4xf64>) -> tensor<4xi1> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
    %0 = stablehlo.abs %arg0 : tensor<4xf64>
    %1 = stablehlo.compare  NE, %0, %cst,  FLOAT : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
    return %1 : tensor<4xi1>
  }

  // Integers have no NaN, so even the gated directions fold unconditionally.
  func.func @int(%arg0: tensor<4xi64>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<0> : tensor<4xi64>
    %0 = stablehlo.abs %arg0 : tensor<4xi64>
    %1 = stablehlo.compare  GT, %0, %c,  SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %1 : tensor<4xi1>
  }
}

// The two NaN-sensitive directions: folded under no_nan, left alone otherwise.

// NONAN:  func.func @main(%arg0: tensor<3056xf64>) -> tensor<3056xi1> {
// NONAN-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<3056xf64>
// NONAN-NEXT:    %0 = stablehlo.compare NE, %arg0, %cst : (tensor<3056xf64>, tensor<3056xf64>) -> tensor<3056xi1>
// NONAN-NEXT:    return %0 : tensor<3056xi1>
// NONAN-NEXT:  }
// NAN:  func.func @main(%arg0: tensor<3056xf64>) -> tensor<3056xi1> {
// NAN-NEXT:    %cst = stablehlo.constant {enzymexla.no_nan = [#enzymexla<guaranteed GUARANTEED>]} dense<0.000000e+00> : tensor<3056xf64>
// NAN-NEXT:    %0 = stablehlo.abs %arg0 {enzymexla.no_nan = [#enzymexla<guaranteed NOTGUARANTEED>], enzymexla.non_negative = [#enzymexla<guaranteed GUARANTEED>]} : tensor<3056xf64>
// NAN-NEXT:    %1 = stablehlo.compare LT, %cst, %0, FLOAT {enzymexla.no_nan = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<3056xf64>, tensor<3056xf64>) -> tensor<3056xi1>
// NAN-NEXT:    return %1 : tensor<3056xi1>
// NAN-NEXT:  }

// NONAN:  func.func @ge(%arg0: tensor<4xf64>) -> tensor<4xi1> {
// NONAN-NEXT:    %c = stablehlo.constant dense<true> : tensor<4xi1>
// NONAN-NEXT:    return %c : tensor<4xi1>
// NONAN-NEXT:  }
// NAN:  func.func @ge(%arg0: tensor<4xf64>) -> tensor<4xi1> {
// NAN-NEXT:    %cst = stablehlo.constant {enzymexla.no_nan = [#enzymexla<guaranteed GUARANTEED>]} dense<0.000000e+00> : tensor<4xf64>
// NAN-NEXT:    %0 = stablehlo.abs %arg0 {enzymexla.no_nan = [#enzymexla<guaranteed NOTGUARANTEED>], enzymexla.non_negative = [#enzymexla<guaranteed GUARANTEED>]} : tensor<4xf64>
// NAN-NEXT:    %1 = stablehlo.compare GE, %0, %cst, FLOAT {enzymexla.no_nan = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
// NAN-NEXT:    return %1 : tensor<4xi1>
// NAN-NEXT:  }

// The NaN-safe directions: identical output in both modes.

// NONAN:  func.func @lt(%arg0: tensor<4xf64>) -> tensor<4xi1> {
// NONAN-NEXT:    %c = stablehlo.constant dense<false> : tensor<4xi1>
// NONAN-NEXT:    return %c : tensor<4xi1>
// NONAN-NEXT:  }
// NAN:  func.func @lt(%arg0: tensor<4xf64>) -> tensor<4xi1> {
// NAN-NEXT:    %c = stablehlo.constant dense<false> : tensor<4xi1>
// NAN-NEXT:    return %c : tensor<4xi1>
// NAN-NEXT:  }

// NONAN:  func.func @le(%arg0: tensor<4xf64>) -> tensor<4xi1> {
// NONAN-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
// NONAN-NEXT:    %0 = stablehlo.compare EQ, %arg0, %cst : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
// NONAN-NEXT:    return %0 : tensor<4xi1>
// NONAN-NEXT:  }
// NAN:  func.func @le(%arg0: tensor<4xf64>) -> tensor<4xi1> {
// NAN-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
// NAN-NEXT:    %0 = stablehlo.compare EQ, %arg0, %cst : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
// NAN-NEXT:    return %0 : tensor<4xi1>
// NAN-NEXT:  }

// NONAN:  func.func @eq(%arg0: tensor<4xf64>) -> tensor<4xi1> {
// NONAN-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
// NONAN-NEXT:    %0 = stablehlo.compare EQ, %arg0, %cst : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
// NONAN-NEXT:    return %0 : tensor<4xi1>
// NONAN-NEXT:  }
// NAN:  func.func @eq(%arg0: tensor<4xf64>) -> tensor<4xi1> {
// NAN-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
// NAN-NEXT:    %0 = stablehlo.compare EQ, %arg0, %cst : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
// NAN-NEXT:    return %0 : tensor<4xi1>
// NAN-NEXT:  }

// NONAN:  func.func @ne(%arg0: tensor<4xf64>) -> tensor<4xi1> {
// NONAN-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
// NONAN-NEXT:    %0 = stablehlo.compare NE, %arg0, %cst : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
// NONAN-NEXT:    return %0 : tensor<4xi1>
// NONAN-NEXT:  }
// NAN:  func.func @ne(%arg0: tensor<4xf64>) -> tensor<4xi1> {
// NAN-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
// NAN-NEXT:    %0 = stablehlo.compare NE, %arg0, %cst : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
// NAN-NEXT:    return %0 : tensor<4xi1>
// NAN-NEXT:  }

// NONAN:  func.func @int(%arg0: tensor<4xi64>) -> tensor<4xi1> {
// NONAN-NEXT:    %c = stablehlo.constant dense<0> : tensor<4xi64>
// NONAN-NEXT:    %0 = stablehlo.compare NE, %arg0, %c : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
// NONAN-NEXT:    return %0 : tensor<4xi1>
// NONAN-NEXT:  }
// NAN:  func.func @int(%arg0: tensor<4xi64>) -> tensor<4xi1> {
// NAN-NEXT:    %c = stablehlo.constant dense<0> : tensor<4xi64>
// NAN-NEXT:    %0 = stablehlo.compare NE, %arg0, %c : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
// NAN-NEXT:    return %0 : tensor<4xi1>
// NAN-NEXT:  }
