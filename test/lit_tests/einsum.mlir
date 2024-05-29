// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

module {
  func.func @main(%a : tensor<2x3xf32>, %b : tensor<4x3x5xf32>) -> tensor<4x2x5xf32> {
    %c = "stablehlo.einsum"(%a,%b) {einsum_config = "ab,cbd->cad"} : (tensor<2x3xf32>, tensor<4x3x5xf32>) -> tensor<4x2x5xf32>
    func.return %c : tensor<4x2x5xf32>
  }
}

// TODO complex version
// FORWARD:  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<4x3x5xf32>, %arg3: tensor<4x3x5xf32>) -> (tensor<4x2x5xf32>, tensor<4x2x5xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.einsum %arg1, %arg2, config = "ab,cbd->cad" : (tensor<2x3xf32>, tensor<4x3x5xf32>) -> tensor<4x2x5xf32>
// FORWARD-NEXT:    %1 = stablehlo.einsum %arg0, %arg3, config = "ab,cbd->cad" : (tensor<2x3xf32>, tensor<4x3x5xf32>) -> tensor<4x2x5xf32>
// FORWARD-NEXT:    %2 = stablehlo.add %0, %1 : tensor<4x2x5xf32>
// FORWARD-NEXT:    %3 = stablehlo.einsum %arg0, %arg2, config = "ab,cbd->cad" : (tensor<2x3xf32>, tensor<4x3x5xf32>) -> tensor<4x2x5xf32>
// FORWARD-NEXT:    return %3, %2 : tensor<4x2x5xf32>, tensor<4x2x5xf32>
// FORWARD-NEXT:  }

// TODO complex version
// REVERSE:  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<4x3x5xf32>, %arg2: tensor<4x2x5xf32>) -> (tensor<2x3xf32>, tensor<4x3x5xf32>) {
// REVERSE-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<4x2x5xf32>
// REVERSE-NEXT:    %cst_0 = arith.constant dense<0.000000e+00> : tensor<2x3xf32>
// REVERSE-NEXT:    %cst_1 = arith.constant dense<0.000000e+00> : tensor<4x3x5xf32>
// REVERSE-NEXT:    %0 = arith.addf %arg2, %cst : tensor<4x2x5xf32>
// REVERSE-NEXT:    %1 = stablehlo.einsum %0, %arg1, config = "cad,cbd->ab" : (tensor<4x2x5xf32>, tensor<4x3x5xf32>) -> tensor<2x3xf32>
// REVERSE-NEXT:    %2 = arith.addf %1, %cst_0 : tensor<2x3xf32>
// REVERSE-NEXT:    %3 = stablehlo.einsum %0, %arg0, config = "cad,ab->cbd" : (tensor<4x2x5xf32>, tensor<2x3xf32>) -> tensor<4x3x5xf32>
// REVERSE-NEXT:    %4 = arith.addf %3, %cst_1 : tensor<4x3x5xf32>
// REVERSE-NEXT:    return %2, %4 : tensor<2x3xf32>, tensor<4x3x5xf32>
// REVERSE-NEXT:  }
