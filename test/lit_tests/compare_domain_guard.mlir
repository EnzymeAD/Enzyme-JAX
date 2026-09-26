// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

// Domain-guard regression for the CompareOpCanon folds that trust an analysis
// fact (non_negative / no_nan). These lock the *guard*: an operand of unknown
// sign that is not provably no-NaN must NOT let `compare(x, const)` collapse to
// a boolean constant. If NonNegativeResultAnalysis / NoNanResultAnalysis ever
// wrongly proves one of these operands (the #2648 / #2651 failure shape), the
// fold fires and the compare disappears -- these CHECKs then fail.
//
// The assertion is deliberately robust: it only requires that a `stablehlo.compare`
// survive (other canonicalizations may rewrite its operands, which is fine).
// The single thing it forbids is the fold to a constant.
//
// No local build available (heavy LLVM/MLIR/XLA); CHECK lines are reasoned
// against the pass logic, not run. CI is the verifier.

// A float operand of unknown sign and unknown NaN-status: `x - y` is neither
// guaranteed non-negative nor guaranteed no-NaN, so `(x - y) < 0` must be kept.
func.func @compare_unknown_float_kept(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> tensor<4xi1> {
  %0 = stablehlo.subtract %arg0, %arg1 : tensor<4xf64>
  %c = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
  %1 = stablehlo.compare LT, %0, %c : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
  return %1 : tensor<4xi1>
}

// CHECK-LABEL: func.func @compare_unknown_float_kept
// CHECK: stablehlo.compare

// A signed integer operand of unknown sign: `x < -1` must be kept.
func.func @compare_unknown_signed_kept(%arg0: tensor<4xi32>) -> tensor<4xi1> {
  %c = stablehlo.constant dense<-1> : tensor<4xi32>
  %0 = stablehlo.compare LT, %arg0, %c, SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// CHECK-LABEL: func.func @compare_unknown_signed_kept
// CHECK: stablehlo.compare
