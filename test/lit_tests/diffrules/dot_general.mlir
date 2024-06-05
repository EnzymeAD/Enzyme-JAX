// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%a : tensor<2x3xf32>, %b : tensor<4x2xf32>) -> tensor<3x4xf32> {
  %c = stablehlo.dot_general %a, %b {
    lhs_batching_dimensions = [1],
    rhs_batching_dimensions = [0],
    lhs_contracting_dimensions = [0],
    rhs_contracting_dimensions = [1],
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<2x3xf32>, tensor<4x2xf32>) -> tensor<3x4xf32>
  func.return %c : tensor<3x4xf32>
}
