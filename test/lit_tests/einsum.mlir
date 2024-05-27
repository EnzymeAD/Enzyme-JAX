// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

module {
  func.func @main(%a : tensor<2x3xf32>, %b : tensor<4x3x5xf32>) -> tensor<4x2x5xf32> {
    %c = "stablehlo.einsum"(%a,%b) {einsum_config = "ab,cbd->cad"} : (tensor<2x3xf32>, tensor<4x3x5xf32>) -> tensor<4x2x5xf32>
    func.return %c : tensor<4x2x5xf32>
  }
}

// TODO complex version
// FORWARD:  func.func @main(%a: tensor<2x3xf32>, %b: tensor<4x3x5xf32>, %da: tensor<2x3xf32>, %db : tensor<4x3x5xf32>) -> (tensor<4x2x5xf32>, tensor<4x2x5xf32>) {
// FORWARD-NEXT:    %[[c:.+]] = "stablehlo.einsum"(%a, %b) {einsum_config = "ab,cbd->cad"} : (tensor<2x3xf32>, tensor<4x3x5xf32>) -> tensor<4x2x5xf32>
// FORWARD-NEXT:    %[[dc:.+]] = "stablehlo.einsum"(%da, %db) {einsum_config = "ab,cbd->cad"} : (tensor<2x3xf32>, tensor<4x3x5xf32>) -> tensor<4x2x5xf32>
// FORWARD-NEXT:    return %[[c]], %[[dc]] : tensor<4x2x5xf32>, tensor<4x2x5xf32>
// FORWARD-NEXT:  }

// TODO complex version
// REVERSE:  func.func @main(%a: tensor<2x3xf32>, %b: tensor<4x3x5xf32>, %dc: tensor<4x2x5xf32>) -> (tensor<2x3xf32>, tensor<4x3x5xf32>) {
// REVERSE-NEXT:    %[[da:.+]] = "stablehlo.einsum"(%dc, %b) {einsum_config = "cad,cbd->ab"} : (tensor<4x2x5xf32>, tensor<4x3x5xf32>) -> tensor<2x3xf32>
// REVERSE-NEXT:    %[[db:.+]] = "stablehlo.einsum"(%a, %dc) {einsum_config = "ab,cad->cbd"} : (tensor<2x3xf32>, tensor<4x2x5xf32>) -> tensor<4x3x5xf32>
// REVERSE-NEXT:    return %[[da]], %[[db]] : tensor<2x3xf32>, tensor<4x3x5xf32>
// REVERSE-NEXT:  }

