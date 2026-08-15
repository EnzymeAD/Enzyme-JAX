// RUN: not enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" 2>&1 | FileCheck %s

// A custom call with no enzyme.reverse attribute is still refused, rather than
// silently differentiating to zero.

func.func @main(%x : tensor<4xf32>) -> tensor<4xf32> {
  %y = stablehlo.custom_call @scale(%x) : (tensor<4xf32>) -> tensor<4xf32>
  func.return %y : tensor<4xf32>
}

// CHECK: could not compute the adjoint for this operation
// CHECK-SAME: stablehlo.custom_call
