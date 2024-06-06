// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%pred : tensor<2xi1>, %on_true : tensor<2xf32>, %on_false : tensor<2xf32>) -> tensor<2xf32> {
  %res = stablehlo.select %pred, %on_true, %on_false : (tensor<2xi1>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %res : tensor<2xf32>
}
