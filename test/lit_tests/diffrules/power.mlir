// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%a : tensor<2xf32>, %b : tensor<2xf32>) -> tensor<2xf32> {
  %c = stablehlo.power %a, %b : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %c : tensor<2xf32>
}

// FORWARD:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
// FORWARD-NEXT:    %0 = stablehlo.subtract %arg2, %cst : tensor<2xf32>
// FORWARD-NEXT:    %1 = stablehlo.power %arg0, %0 : tensor<2xf32>
// FORWARD-NEXT:    %2 = stablehlo.multiply %arg2, %1 : tensor<2xf32>
// FORWARD-NEXT:    %3 = stablehlo.multiply %arg1, %2 : tensor<2xf32>
// FORWARD-NEXT:    %4 = stablehlo.power %arg0, %arg2 : tensor<2xf32>
// FORWARD-NEXT:    %5 = stablehlo.log %arg0 : tensor<2xf32>
// FORWARD-NEXT:    %6 = stablehlo.multiply %4, %5 : tensor<2xf32>
// FORWARD-NEXT:    %7 = stablehlo.multiply %arg3, %6 : tensor<2xf32>
// FORWARD-NEXT:    %8 = arith.addf %3, %7 : tensor<2xf32>
// FORWARD-NEXT:    %9 = stablehlo.power %arg0, %arg2 : tensor<2xf32>
// FORWARD-NEXT:    return %9, %8 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %cst_0 = arith.constant dense<0.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %0 = arith.addf %arg2, %cst_0 : tensor<2xf32>
// REVERSE-NEXT:    %1 = stablehlo.subtract %arg1, %cst : tensor<2xf32>
// REVERSE-NEXT:    %2 = stablehlo.power %arg0, %1 : tensor<2xf32>
// REVERSE-NEXT:    %3 = stablehlo.multiply %arg1, %2 : tensor<2xf32>
// REVERSE-NEXT:    %4 = stablehlo.multiply %0, %3 : tensor<2xf32>
// REVERSE-NEXT:    %5 = arith.addf %4, %cst_0 : tensor<2xf32>
// REVERSE-NEXT:    %6 = stablehlo.power %arg0, %arg1 : tensor<2xf32>
// REVERSE-NEXT:    %7 = stablehlo.log %arg0 : tensor<2xf32>
// REVERSE-NEXT:    %8 = stablehlo.multiply %6, %7 : tensor<2xf32>
// REVERSE-NEXT:    %9 = stablehlo.multiply %0, %8 : tensor<2xf32>
// REVERSE-NEXT:    %10 = arith.addf %9, %cst_0 : tensor<2xf32>
// REVERSE-NEXT:    return %5, %10 : tensor<2xf32>, tensor<2xf32>
// REVERSE-NEXT:  }
