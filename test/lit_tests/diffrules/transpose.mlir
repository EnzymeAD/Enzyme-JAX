// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%x : tensor<2x3x4x5xf32>) -> tensor<5x3x2x4xf32> {
  %y = "stablehlo.transpose"(%x) { permutation = array<i64 : 3, 1, 0, 2> } : (tensor<2x3x4x5xf32>) -> tensor<5x3x2x4xf32>
  func.return %y : tensor<5x3x2x4xf32>
}
