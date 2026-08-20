// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

// enzyme.nondiff marks the call as a gradient stop: no reverse function is
// needed and none is called, and no cotangent flows back to the operand.

func.func @main(%x : tensor<4xf32>) -> tensor<4xf32> {
  %y = stablehlo.custom_call @round_to_int(%x) {
    enzyme.nondiff
  } : (tensor<4xf32>) -> tensor<4xf32>
  func.return %y : tensor<4xf32>
}

// REVERSE: func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
// REVERSE-NOT: = call @
// REVERSE: return
