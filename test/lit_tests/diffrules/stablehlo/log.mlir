// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%x : tensor<2xf32>) -> tensor<2xf32> {
  %y = stablehlo.log %x : (tensor<2xf32>) -> tensor<2xf32>
  func.return %y : tensor<2xf32>
}

// FORWARD:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.divide %arg1, %arg0 : tensor<2xf32>
// FORWARD-NEXT:    %1 = stablehlo.log %arg0 : tensor<2xf32>
// FORWARD-NEXT:    return %1, %0 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %cst_0 = arith.constant dense<0.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %0 = arith.addf %arg1, %cst_0 : tensor<2xf32>
// REVERSE-NEXT:    %1 = stablehlo.add %arg0, %cst : tensor<2xf32>
// REVERSE-NEXT:    %2 = stablehlo.divide %0, %1 : tensor<2xf32>
// REVERSE-NEXT:    %3 = arith.addf %2, %cst_0 : tensor<2xf32>
// REVERSE-NEXT:    return %3 : tensor<2xf32>
// REVERSE-NEXT:  }
