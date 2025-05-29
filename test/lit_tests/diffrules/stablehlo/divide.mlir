// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%a : tensor<2xf32>, %b : tensor<2xf32>) -> tensor<2xf32> {
  %c = stablehlo.divide %a, %b : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %c : tensor<2xf32>
}

// FORWARD:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.multiply %arg1, %arg2 : tensor<2xf32>
// FORWARD-NEXT:    %1 = stablehlo.multiply %arg3, %arg0 : tensor<2xf32>
// FORWARD-NEXT:    %2 = stablehlo.subtract %0, %1 : tensor<2xf32>
// FORWARD-NEXT:    %3 = stablehlo.multiply %arg2, %arg2 : tensor<2xf32>
// FORWARD-NEXT:    %4 = stablehlo.divide %2, %3 : tensor<2xf32>
// FORWARD-NEXT:    %5 = stablehlo.divide %arg0, %arg2 : tensor<2xf32>
// FORWARD-NEXT:    return %5, %4 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// REVERSE-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %0 = arith.addf %arg2, %cst : tensor<2xf32>
// REVERSE-NEXT:    %1 = stablehlo.divide %0, %arg1 : tensor<2xf32>
// REVERSE-NEXT:    %2 = arith.addf %1, %cst : tensor<2xf32>
// REVERSE-NEXT:    %3 = stablehlo.divide %0, %arg1 : tensor<2xf32>
// REVERSE-NEXT:    %4 = stablehlo.divide %arg0, %arg1 : tensor<2xf32>
// REVERSE-NEXT:    %5 = stablehlo.multiply %3, %4 : tensor<2xf32>
// REVERSE-NEXT:    %6 = stablehlo.negate %5 : tensor<2xf32>
// REVERSE-NEXT:    %7 = arith.addf %6, %cst : tensor<2xf32>
// REVERSE-NEXT:    return %2, %7 : tensor<2xf32>, tensor<2xf32>
// REVERSE-NEXT:  }
