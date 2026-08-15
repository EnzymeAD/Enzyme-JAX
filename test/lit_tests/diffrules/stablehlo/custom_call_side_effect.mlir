// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

// Two properties of the motivating kernel at once: the forward call is marked
// has_side_effect (the primal is cloned as-is and the caches are pushed around
// it, so this is fine), and the reverse rule is not itself pure StableHLO -- it
// contains custom calls of its own, one feeding the next, exactly like
// _bwd_q producing the `delta` that _bwd_kv consumes. Nothing may try to
// differentiate the rule's own body.

func.func @scale_rev(%x: tensor<4xf32>, %y: tensor<4xf32>,
                     %dy: tensor<4xf32>) -> tensor<4xf32> {
  %delta = stablehlo.custom_call @bwd_first(%x, %dy) {
    has_side_effect = true
  } : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %dx = stablehlo.custom_call @bwd_second(%delta, %y) {
    has_side_effect = true
  } : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %dx : tensor<4xf32>
}

func.func @main(%x : tensor<4xf32>) -> tensor<4xf32> {
  %y = stablehlo.custom_call @scale(%x) {
    enzyme.reverse = @scale_rev,
    enzyme.active_operands = array<i64: 0>,
    has_side_effect = true
  } : (tensor<4xf32>) -> tensor<4xf32>
  func.return %y : tensor<4xf32>
}

// The rule's body is untouched, and the ordering between its two calls stays a
// dataflow edge rather than a scheduling assumption.
// REVERSE: func.func @scale_rev(
// REVERSE:   %[[DELTA:.+]] = stablehlo.custom_call @bwd_first(
// REVERSE:   stablehlo.custom_call @bwd_second(%[[DELTA]],

// REVERSE: func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
// REVERSE:   stablehlo.custom_call @scale(%arg0)
// REVERSE:   call @scale_rev(
// REVERSE:   return
