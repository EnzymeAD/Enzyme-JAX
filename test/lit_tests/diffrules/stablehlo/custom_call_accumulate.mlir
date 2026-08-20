// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

// The custom call's result feeds two consumers and its operand feeds a third,
// so cotangents have to accumulate on both sides rather than overwrite: the
// cotangent handed to @scale_rev is the sum of the two uses of %y, and the
// cotangent @scale_rev returns is added onto the one %x already has.

func.func @scale_rev(%x: tensor<4xf32>,
                     %y: tensor<4xf32>,
                     %dy: tensor<4xf32>) -> tensor<4xf32> {
  %three = stablehlo.constant dense<3.000000e+00> : tensor<4xf32>
  %res = stablehlo.multiply %dy, %three : tensor<4xf32>
  func.return %res : tensor<4xf32>
}

func.func @main(%x : tensor<4xf32>) -> tensor<4xf32> {
  %y = stablehlo.custom_call @scale(%x) {
    enzyme.reverse = @scale_rev,
    enzyme.active_operands = array<i64: 0>
  } : (tensor<4xf32>) -> tensor<4xf32>
  %a = stablehlo.sine %y : tensor<4xf32>
  %b = stablehlo.cosine %y : tensor<4xf32>
  %c = stablehlo.add %a, %b : tensor<4xf32>
  %d = stablehlo.add %c, %x : tensor<4xf32>
  func.return %d : tensor<4xf32>
}

// REVERSE: func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
// REVERSE:   stablehlo.custom_call @scale(%arg0)
// The two uses of %y are summed into a single cotangent before the call...
// REVERSE:   arith.addf
// REVERSE:   call @scale_rev(%{{.+}}, %{{.+}}, %{{.+}}) : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// ... and what it returns is added onto the cotangent %x already has.
// REVERSE:   arith.addf
// REVERSE:   return
