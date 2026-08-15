// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

// enzyme.nondiff wins over enzyme.reverse: the call is a gradient stop and the
// reverse function is not called, even though one is named. Pinning the
// precedence down means a frontend that emits both by accident gets the safe
// answer rather than an arbitrary one.

func.func @scale_rev(%x: tensor<4xf32>, %y: tensor<4xf32>,
                     %dy: tensor<4xf32>) -> tensor<4xf32> {
  %three = stablehlo.constant dense<3.000000e+00> : tensor<4xf32>
  %res = stablehlo.multiply %dy, %three : tensor<4xf32>
  func.return %res : tensor<4xf32>
}

func.func @main(%x : tensor<4xf32>) -> tensor<4xf32> {
  %y = stablehlo.custom_call @scale(%x) {
    enzyme.nondiff,
    enzyme.reverse = @scale_rev,
    enzyme.active_operands = array<i64: 0>
  } : (tensor<4xf32>) -> tensor<4xf32>
  func.return %y : tensor<4xf32>
}

// REVERSE: func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
// REVERSE-NOT: = call @
// REVERSE: return
