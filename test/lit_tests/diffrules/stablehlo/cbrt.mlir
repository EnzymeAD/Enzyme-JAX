// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%x : tensor<2xf32>) -> tensor<2xf32> {
  %y = stablehlo.cbrt %x : (tensor<2xf32>) -> tensor<2xf32>
  func.return %y : tensor<2xf32>
}

// FORWARD:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %cst = stablehlo.constant dense<-2.000000e+00> : tensor<2xf32>
// FORWARD-NEXT:    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<2xf32>
// FORWARD-NEXT:    %0 = stablehlo.divide %cst, %cst_0 : tensor<2xf32>
// FORWARD-NEXT:    %1 = stablehlo.power %arg0, %0 : tensor<2xf32>
// FORWARD-NEXT:    %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<2xf32>
// FORWARD-NEXT:    %2 = stablehlo.divide %1, %cst_1 : tensor<2xf32>
// FORWARD-NEXT:    %3 = stablehlo.multiply %arg1, %2 : tensor<2xf32>
// FORWARD-NEXT:    %4 = stablehlo.cbrt %arg0 : tensor<2xf32>
// FORWARD-NEXT:    return %4, %3 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<3.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %cst_0 = stablehlo.constant dense<-2.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %cst_1 = arith.constant dense<0.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %0 = arith.addf %arg1, %cst_1 : tensor<2xf32>
// REVERSE-NEXT:    %1 = stablehlo.divide %cst_0, %cst : tensor<2xf32>
// REVERSE-NEXT:    %2 = stablehlo.power %arg0, %1 : tensor<2xf32>
// REVERSE-NEXT:    %3 = stablehlo.divide %2, %cst : tensor<2xf32>
// REVERSE-NEXT:    %4 = stablehlo.multiply %0, %3 : tensor<2xf32>
// REVERSE-NEXT:    %5 = arith.addf %4, %cst_1 : tensor<2xf32>
// REVERSE-NEXT:    return %5 : tensor<2xf32>
// REVERSE-NEXT:  }
