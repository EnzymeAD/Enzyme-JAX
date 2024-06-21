// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%x : tensor<2xf32>) -> tensor<2xf32> {
  %y = stablehlo.tanh %x : tensor<2xf32>
  func.return %y : tensor<2xf32>
}

// FORWARD:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
// FORWARD-NEXT:    %0 = stablehlo.tanh %arg0 : tensor<2xf32>
// FORWARD-NEXT:    %1 = stablehlo.multiply %0, %0 : tensor<2xf32>
// FORWARD-NEXT:    %2 = stablehlo.subtract %cst, %1 : tensor<2xf32>
// FORWARD-NEXT:    %3 = stablehlo.multiply %arg1, %2 : tensor<2xf32>
// FORWARD-NEXT:    return %0, %3 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %cst_0 = arith.constant dense<0.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %0 = arith.addf %arg1, %cst_0 : tensor<2xf32>
// REVERSE-NEXT:    %1 = stablehlo.tanh %arg0 : tensor<2xf32>
// REVERSE-NEXT:    %2 = stablehlo.tanh %arg0 : tensor<2xf32>
// REVERSE-NEXT:    %3 = stablehlo.multiply %1, %2 : tensor<2xf32>
// REVERSE-NEXT:    %4 = stablehlo.subtract %cst, %3 : tensor<2xf32>
// REVERSE-NEXT:    %5 = stablehlo.multiply %0, %4 : tensor<2xf32>
// REVERSE-NEXT:    %6 = arith.addf %5, %cst_0 : tensor<2xf32>
// REVERSE-NEXT:    return %6 : tensor<2xf32>
// REVERSE-NEXT:  }
