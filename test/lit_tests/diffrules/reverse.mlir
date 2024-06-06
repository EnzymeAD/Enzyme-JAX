// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%x : tensor<2x3xf32>) -> tensor<2x3xf32> {
  %y = "stablehlo.reverse"(%x) { dimensions = array<i64 : 1> } : (tensor<2x3xf32>) -> tensor<2x3xf32>
  func.return %y : tensor<2x3xf32>
}
