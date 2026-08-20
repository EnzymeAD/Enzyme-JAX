// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=cse_batch_norm_training<16>;cse_exp<16>;cse_convolution<16>" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file %s | FileCheck %s

// The `enzymexla.{non_negative,no_nan,finite,...}` attributes are memoized analysis
// results: GuaranteedResultAnalysisBase::setGuaranteedInIR writes the conclusion back onto
// the defining op so later queries can read it via lookupGuaranteedFromIR. They are
// discardable attributes, so MLIR's own CSE -- which compares the full discardable
// attribute dictionary -- treats two otherwise identical ops as inequivalent once one of
// them has been queried by an analysis.
//
// CSE<T> passes OperationEquivalence::IgnoreDiscardableAttrs and so is immune, but it is
// registered for an explicit list of op types. Ops off that list had no working CSE at all.
//
// Reduced from grad(DGCNN) in Reactant.jl's benchmark/nn, where this left three copies of
// an 8.4M-element batch_norm_training in the derivative.

func.func @bn_differs_by_cached_analysis(%x: tensor<64x128xf32>, %s: tensor<64xf32>, %o: tensor<64xf32>)
    -> (tensor<64x128xf32>, tensor<64x128xf32>) {
  %a:3 = "stablehlo.batch_norm_training"(%x, %s, %o) <{epsilon = 9.99999974E-6 : f32, feature_index = 0 : i64}> : (tensor<64x128xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<64x128xf32>, tensor<64xf32>, tensor<64xf32>)
  %b:3 = "stablehlo.batch_norm_training"(%x, %s, %o) <{epsilon = 9.99999974E-6 : f32, feature_index = 0 : i64}> {enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>]} : (tensor<64x128xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<64x128xf32>, tensor<64xf32>, tensor<64xf32>)
  return %a#0, %b#0 : tensor<64x128xf32>, tensor<64x128xf32>
}

// CHECK-LABEL:   func.func @bn_differs_by_cached_analysis(
// CHECK:           %[[BN:.*]], %{{.*}}, %{{.*}} = "stablehlo.batch_norm_training"
// CHECK-NOT:       stablehlo.batch_norm_training
// CHECK:           return %[[BN]], %[[BN]]

// -----

// Differing epsilon must still block CSE.
func.func @bn_differs_by_epsilon(%x: tensor<64x128xf32>, %s: tensor<64xf32>, %o: tensor<64xf32>)
    -> (tensor<64x128xf32>, tensor<64x128xf32>) {
  %a:3 = "stablehlo.batch_norm_training"(%x, %s, %o) <{epsilon = 9.99999974E-6 : f32, feature_index = 0 : i64}> : (tensor<64x128xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<64x128xf32>, tensor<64xf32>, tensor<64xf32>)
  %b:3 = "stablehlo.batch_norm_training"(%x, %s, %o) <{epsilon = 1.000000e-03 : f32, feature_index = 0 : i64}> : (tensor<64x128xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<64x128xf32>, tensor<64xf32>, tensor<64xf32>)
  return %a#0, %b#0 : tensor<64x128xf32>, tensor<64x128xf32>
}

// CHECK-LABEL:   func.func @bn_differs_by_epsilon(
// CHECK:           "stablehlo.batch_norm_training"
// CHECK:           "stablehlo.batch_norm_training"

// -----

// Same story for unary math, which was likewise uncovered.
func.func @exp_differs_by_cached_analysis(%x: tensor<64x128xf32>) -> (tensor<64x128xf32>, tensor<64x128xf32>) {
  %a = stablehlo.exponential %x : tensor<64x128xf32>
  %b = stablehlo.exponential %x {enzymexla.no_nan = [#enzymexla<guaranteed GUARANTEED>]} : tensor<64x128xf32>
  return %a, %b : tensor<64x128xf32>, tensor<64x128xf32>
}

// CHECK-LABEL:   func.func @exp_differs_by_cached_analysis(
// CHECK-NEXT:      %[[E:.*]] = stablehlo.exponential
// CHECK-NOT:       stablehlo.exponential
// CHECK:           return %[[E]], %[[E]]
