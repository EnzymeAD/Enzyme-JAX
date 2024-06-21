// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%x : tensor<2xf32>) -> tensor<2xf32> {
  %y = chlo.atanh %x : tensor<2xf32> -> tensor<2xf32>
  func.return %y : tensor<2xf32>
}

// FORWARD:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.multiply %arg0, %arg0 : tensor<2xf32>
// FORWARD-NEXT:    %1 = chlo.constant dense<1.000000e+00> : tensor<2xf32>
// FORWARD-NEXT:    %2 = stablehlo.subtract %0, %1 : tensor<2xf32>
// FORWARD-NEXT:    %3 = stablehlo.divide %arg1, %2 : tensor<2xf32>
// FORWARD-NEXT:    %4 = chlo.atanh %arg0 : tensor<2xf32> -> tensor<2xf32>
// FORWARD-NEXT:    return %4, %3 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
// REVERSE-NEXT:    %0 = chlo.constant dense<1.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %1 = arith.addf %arg1, %cst : tensor<2xf32>
// REVERSE-NEXT:    %2 = stablehlo.multiply %arg0, %arg0 : tensor<2xf32>
// REVERSE-NEXT:    %3 = stablehlo.subtract %2, %0 : tensor<2xf32>
// REVERSE-NEXT:    %4 = stablehlo.divide %1, %3 : tensor<2xf32>
// REVERSE-NEXT:    %5 = arith.addf %4, %cst : tensor<2xf32>
// REVERSE-NEXT:    return %5 : tensor<2xf32>
// REVERSE-NEXT:  }
