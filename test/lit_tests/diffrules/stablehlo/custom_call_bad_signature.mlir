// RUN: not enzymexlamlir-opt %s --split-input-file --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" 2>&1 | FileCheck %s

// The convention is checked before it is used: getting it wrong would produce a
// silently wrong gradient rather than a failure.

// @scale_rev forgets the primal result, so it takes one argument too few.
func.func @scale_rev(%x: tensor<4xf32>, %dy: tensor<4xf32>) -> tensor<4xf32> {
  %three = stablehlo.constant dense<3.000000e+00> : tensor<4xf32>
  %res = stablehlo.multiply %dy, %three : tensor<4xf32>
  func.return %res : tensor<4xf32>
}

func.func @main(%x : tensor<4xf32>) -> tensor<4xf32> {
  %y = stablehlo.custom_call @scale(%x) {
    enzyme.reverse = @scale_rev,
    enzyme.active_operands = array<i64: 0>
  } : (tensor<4xf32>) -> tensor<4xf32>
  func.return %y : tensor<4xf32>
}

// CHECK: 'scale_rev' takes 2 argument(s), expected 3

// -----

// @scale_rev returns a cotangent of the wrong shape for operand 0.
func.func @scale_rev(%x: tensor<4xf32>, %y: tensor<4xf32>,
                     %dy: tensor<4xf32>) -> tensor<8xf32> {
  %res = stablehlo.constant dense<0.000000e+00> : tensor<8xf32>
  func.return %res : tensor<8xf32>
}

func.func @main(%x : tensor<4xf32>) -> tensor<4xf32> {
  %y = stablehlo.custom_call @scale(%x) {
    enzyme.reverse = @scale_rev,
    enzyme.active_operands = array<i64: 0>
  } : (tensor<4xf32>) -> tensor<4xf32>
  func.return %y : tensor<4xf32>
}

// CHECK: 'scale_rev' result 0 has type tensor<8xf32>, expected the cotangent type tensor<4xf32> of operand 0
