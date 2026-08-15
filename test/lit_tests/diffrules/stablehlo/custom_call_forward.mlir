// RUN: not enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" 2>&1 | FileCheck %s

// enzyme.reverse is reverse-mode only. Forward mode has no rule for a custom
// call and must keep refusing, rather than quietly using the reverse function
// or producing a zero tangent.

func.func @scale_rev(%x: tensor<4xf32>, %y: tensor<4xf32>,
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
  func.return %y : tensor<4xf32>
}

// CHECK: could not compute the adjoint for this operation
// CHECK-SAME: stablehlo.custom_call
