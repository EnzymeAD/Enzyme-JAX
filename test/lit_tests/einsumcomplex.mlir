// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE
// XFAIL: *

module {
  func.func @main(%a : tensor<2x3xcomplex<f32>>, %b : tensor<4x3x5xcomplex<f32>>) -> tensor<4x2x5xcomplex<f32>> {
    %c = "stablehlo.einsum"(%a,%b) {einsum_config = "ab,cbd->cad"} : (tensor<2x3xcomplex<f32>>, tensor<4x3x5xcomplex<f32>>) -> tensor<4x2x5xcomplex<f32>>
    func.return %c : tensor<4x2x5xcomplex<f32>>
  }
}

// FORWARD:  func.func @main(%arg0: tensor<2x3xcomplex<f32>>, %arg1: tensor<2x3xcomplex<f32>>, %arg2: tensor<4x3x5xcomplex<f32>>, %arg3: tensor<4x3x5xcomplex<f32>>) -> (tensor<4x2x5xcomplex<f32>>, tensor<4x2x5xcomplex<f32>>) {
// FORWARD-NEXT:    %0 = complex.conj %arg0 : tensor<2x3xcomplex<f32>>
// FORWARD-NEXT:    %1 = complex.conj %arg2 : tensor<4x3x5xcomplex<f32>>
// FORWARD-NEXT:    %2 = stablehlo.einsum %arg1, %1, config = "ab,cbd->cad" : (tensor<2x3xcomplex<f32>>, tensor<4x3x5xcomplex<f32>>) -> tensor<4x2x5xcomplex<f32>>
// FORWARD-NEXT:    %3 = stablehlo.einsum %0, %arg3, config = "ab,cbd->cad" : (tensor<2x3xcomplex<f32>>, tensor<4x3x5xcomplex<f32>>) -> tensor<4x2x5xcomplex<f32>>
// FORWARD-NEXT:    %4 = stablehlo.add %2, %3 : tensor<4x2x5xcomplex<f32>>
// FORWARD-NEXT:    %3 = stablehlo.einsum %arg0, %arg2, config = "ab,cbd->cad" : (tensor<2x3xcomplex<f32>>, tensor<4x3x5xcomplex<f32>>) -> tensor<4x2x5xcomplex<f32>>
// FORWARD-NEXT:    return %3, %4 : tensor<4x2x5xcomplex<f32>>, tensor<4x2x5xcomplex<f32>>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<2x3xcomplex<f32>>, %arg1: tensor<4x3x5xcomplex<f32>>, %arg2: tensor<4x2x5xcomplex<f32>>) -> (tensor<2x3xcomplex<f32>>, tensor<4x3x5xcomplex<f32>>) {
// REVERSE-NEXT:    %0 = complex.conj %arg0 : tensor<2x3xcomplex<f32>>
// REVERSE-NEXT:    %1 = complex.conj %arg1 : tensor<4x2x5xcomplex<f32>>
// REVERSE-NEXT:    %2 = stablehlo.einsum %arg2, %0, config = "cad,ab->cbd" : (tensor<4x2x5xcomplex<f32>>, tensor<2x3xcomplex<f32>>) -> tensor<4x3x5xcomplex<f32>>
// REVERSE-NEXT:    %3 = stablehlo.einsum %arg2, %1, config = "cad,cbd->ab" : (tensor<4x2x5xcomplex<f32>>, tensor<4x3x5xcomplex<f32>>) -> tensor<2x3xcomplex<f32>>
// REVERSE-NEXT:    return %2, %3 : tensor<2x3xcomplex<f32>>, tensor<4x3x5xcomplex<f32>>
// REVERSE-NEXT:  }
