// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%x : tensor<2xf32>) -> tensor<2xf32> {
  %y = stablehlo.exponential_minus_one %x : (tensor<2xf32>) -> tensor<2xf32>
  func.return %y : tensor<2xf32>
}

// REVERSE:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
// REVERSE-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %0 = arith.addf %arg1, %cst : tensor<2xf32>
// REVERSE-NEXT:    %1 = stablehlo.exponential %arg0 : tensor<2xf32>
// REVERSE-NEXT:    %2 = stablehlo.multiply %0, %1 : tensor<2xf32>
// REVERSE-NEXT:    %3 = arith.addf %2, %cst : tensor<2xf32>
// REVERSE-NEXT:    return %3 : tensor<2xf32>
// REVERSE-NEXT:  }
