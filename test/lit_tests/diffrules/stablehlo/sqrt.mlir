// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%x : tensor<2xf32>) -> tensor<2xf32> {
  %y = stablehlo.sqrt %x : (tensor<2xf32>) -> tensor<2xf32>
  func.return %y : tensor<2xf32>
}

// FORWARD:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// FORWARD-NEXT:    %0 = stablehlo.compare  EQ, %arg0, %cst : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// FORWARD-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// FORWARD-NEXT:    %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<2xf32>
// FORWARD-NEXT:    %1 = stablehlo.sqrt %arg0 : tensor<2xf32>
// FORWARD-NEXT:    %2 = stablehlo.multiply %cst_1, %1 : tensor<2xf32>
// FORWARD-NEXT:    %3 = stablehlo.divide %arg1, %2 : tensor<2xf32>
// FORWARD-NEXT:    %4 = stablehlo.select %0, %cst_0, %3 : tensor<2xi1>, tensor<2xf32>
// FORWARD-NEXT:    %5 = stablehlo.sqrt %arg0 : tensor<2xf32>
// FORWARD-NEXT:    return %5, %4 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<2.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %cst_1 = arith.constant dense<0.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %0 = arith.addf %arg1, %cst_1 : tensor<2xf32>
// REVERSE-NEXT:    %1 = stablehlo.compare  EQ, %arg0, %cst_0 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// REVERSE-NEXT:    %2 = stablehlo.sqrt %arg0 : tensor<2xf32>
// REVERSE-NEXT:    %3 = stablehlo.multiply %cst, %2 : tensor<2xf32>
// REVERSE-NEXT:    %4 = stablehlo.divide %0, %3 : tensor<2xf32>
// REVERSE-NEXT:    %5 = stablehlo.select %1, %cst_0, %4 : tensor<2xi1>, tensor<2xf32>
// REVERSE-NEXT:    %6 = arith.addf %5, %cst_1 : tensor<2xf32>
// REVERSE-NEXT:    return %6 : tensor<2xf32>
// REVERSE-NEXT:  }