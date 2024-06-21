// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup,enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_const,enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%pred : tensor<2xi1>, %on_true : tensor<2xf32>, %on_false : tensor<2xf32>) -> tensor<2xf32> {
  %res = stablehlo.select %pred, %on_true, %on_false : (tensor<2xi1>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %res : tensor<2xf32>
}

// FORWARD:  func.func @main(%arg0: tensor<2xi1>, %arg1: tensor<2xi1>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>, %arg4: tensor<2xf32>, %arg5: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.select %arg0, %arg3, %arg5 : tensor<2xi1>, tensor<2xf32>
// FORWARD-NEXT:    %1 = stablehlo.select %arg0, %arg2, %arg4 : tensor<2xi1>, tensor<2xf32>
// FORWARD-NEXT:    return %1, %0 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<2xi1>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %cst_0 = arith.constant dense<0.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %0 = arith.addf %arg3, %cst_0 : tensor<2xf32>
// REVERSE-NEXT:    %1 = stablehlo.select %arg0, %0, %cst : tensor<2xi1>, tensor<2xf32>
// REVERSE-NEXT:    %2 = arith.addf %1, %cst_0 : tensor<2xf32>
// REVERSE-NEXT:    %3 = stablehlo.select %arg0, %cst, %0 : tensor<2xi1>, tensor<2xf32>
// REVERSE-NEXT:    %4 = arith.addf %3, %cst_0 : tensor<2xf32>
// REVERSE-NEXT:    return %2, %4 : tensor<2xf32>, tensor<2xf32>
// REVERSE-NEXT:  }
