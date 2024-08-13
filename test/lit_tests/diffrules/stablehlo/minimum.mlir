// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%a : tensor<2xf32>, %b : tensor<2xf32>) -> tensor<2xf32> {
  %c = stablehlo.minimum %a, %b : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %c : tensor<2xf32>
}

// FORWARD:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.compare  LT, %arg2, %arg0 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// FORWARD-NEXT:    %1 = stablehlo.select %0, %arg3, %arg1 : tensor<2xi1>, tensor<2xf32>
// FORWARD-NEXT:    %2 = stablehlo.minimum %arg0, %arg2 : tensor<2xf32>
// FORWARD-NEXT:    return %2, %1 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %cst_0 = arith.constant dense<0.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %0 = arith.addf %arg2, %cst_0 : tensor<2xf32>
// REVERSE-NEXT:    %1 = stablehlo.compare  LT, %arg1, %arg0 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// REVERSE-NEXT:    %2 = stablehlo.select %1, %cst, %0 : tensor<2xi1>, tensor<2xf32>
// REVERSE-NEXT:    %3 = arith.addf %2, %cst_0 : tensor<2xf32>
// REVERSE-NEXT:    %4 = stablehlo.compare  LT, %arg1, %arg0 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// REVERSE-NEXT:    %5 = stablehlo.select %4, %0, %cst : tensor<2xi1>, tensor<2xf32>
// REVERSE-NEXT:    %6 = arith.addf %5, %cst_0 : tensor<2xf32>
// REVERSE-NEXT:    return %3, %6 : tensor<2xf32>, tensor<2xf32>
// REVERSE-NEXT:  }
