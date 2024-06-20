// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%a : tensor<2xf32>, %b : tensor<2xf32>) -> tensor<2xf32> {
  %c = stablehlo.atan2 %a, %b : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %c : tensor<2xf32>
}

// FORWARD:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.multiply %arg0, %arg3 : tensor<2xf32>
// FORWARD-NEXT:    %1 = stablehlo.multiply %arg2, %arg1 : tensor<2xf32>
// FORWARD-NEXT:    %2 = stablehlo.subtract %0, %1 : tensor<2xf32>
// FORWARD-NEXT:    %cst = stablehlo.constant dense<2.000000e+00> : tensor<2xf32>
// FORWARD-NEXT:    %3 = stablehlo.power %arg0, %cst : tensor<2xf32>
// FORWARD-NEXT:    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<2xf32>
// FORWARD-NEXT:    %4 = stablehlo.power %arg2, %cst_0 : tensor<2xf32>
// FORWARD-NEXT:    %5 = stablehlo.add %3, %4 : tensor<2xf32>
// FORWARD-NEXT:    %6 = stablehlo.divide %2, %5 : tensor<2xf32>
// FORWARD-NEXT:    %7 = stablehlo.atan2 %arg0, %arg2 : tensor<2xf32>
// FORWARD-NEXT:    return %7, %6 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<2.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %cst_0 = arith.constant dense<0.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %0 = arith.addf %arg2, %cst_0 : tensor<2xf32>
// REVERSE-NEXT:    %1 = stablehlo.negate %arg1 : tensor<2xf32>
// REVERSE-NEXT:    %2 = stablehlo.power %arg0, %cst : tensor<2xf32>
// REVERSE-NEXT:    %3 = stablehlo.power %arg1, %cst : tensor<2xf32>
// REVERSE-NEXT:    %4 = stablehlo.add %2, %3 : tensor<2xf32>
// REVERSE-NEXT:    %5 = stablehlo.divide %1, %4 : tensor<2xf32>
// REVERSE-NEXT:    %6 = stablehlo.multiply %0, %5 : tensor<2xf32>
// REVERSE-NEXT:    %7 = arith.addf %6, %cst_0 : tensor<2xf32>
// REVERSE-NEXT:    %8 = stablehlo.power %arg0, %cst : tensor<2xf32>
// REVERSE-NEXT:    %9 = stablehlo.power %arg1, %cst : tensor<2xf32>
// REVERSE-NEXT:    %10 = stablehlo.add %8, %9 : tensor<2xf32>
// REVERSE-NEXT:    %11 = stablehlo.divide %arg0, %10 : tensor<2xf32>
// REVERSE-NEXT:    %12 = stablehlo.multiply %0, %11 : tensor<2xf32>
// REVERSE-NEXT:    %13 = arith.addf %12, %cst_0 : tensor<2xf32>
// REVERSE-NEXT:    return %7, %13 : tensor<2xf32>, tensor<2xf32>
// REVERSE-NEXT:  }
