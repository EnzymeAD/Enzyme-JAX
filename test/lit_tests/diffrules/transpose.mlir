// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%x : tensor<2x3x4x5xf32>) -> tensor<5x3x2x4xf32> {
  %y = "stablehlo.transpose"(%x) { permutation = array<i64 : 3, 1, 0, 2> } : (tensor<2x3x4x5xf32>) -> tensor<5x3x2x4xf32>
  func.return %y : tensor<5x3x2x4xf32>
}

// REVERSE:  func.func @main(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<5x3x2x4xf32>) -> tensor<2x3x4x5xf32> {
// REVERSE-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<5x3x2x4xf32>
// REVERSE-NEXT:    %cst_0 = arith.constant dense<0.000000e+00> : tensor<2x3x4x5xf32>
// REVERSE-NEXT:    %0 = arith.addf %arg1, %cst : tensor<5x3x2x4xf32>
// REVERSE-NEXT:    %1 = stablehlo.transpose %0, dims = [2, 1, 3, 0] : (tensor<5x3x2x4xf32>) -> tensor<2x3x4x5xf32>
// REVERSE-NEXT:    %2 = arith.addf %1, %cst_0 : tensor<2x3x4x5xf32>
// REVERSE-NEXT:    return %2 : tensor<2x3x4x5xf32>
// REVERSE-NEXT:  }
