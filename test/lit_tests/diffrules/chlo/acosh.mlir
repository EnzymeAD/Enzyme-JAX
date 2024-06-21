// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%x : tensor<2xf32>) -> tensor<2xf32> {
  %y = chlo.acosh %x : tensor<2xf32> -> tensor<2xf32>
  func.return %y : tensor<2xf32>
}

// FORWARD:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.multiply %arg0, %arg0 : tensor<2xf32>
// FORWARD-NEXT:    %1 = chlo.constant dense<1.000000e+00> : tensor<2xf32>
// FORWARD-NEXT:    %2 = stablehlo.subtract %0, %1 : tensor<2xf32>
// FORWARD-NEXT:    %3 = stablehlo.sqrt %2 : tensor<2xf32>
// FORWARD-NEXT:    %4 = stablehlo.multiply %arg0, %arg0 : tensor<2xf32>
// FORWARD-NEXT:    %5 = chlo.constant dense<1.000000e+00> : tensor<2xf32>
// FORWARD-NEXT:    %6 = stablehlo.add %4, %5 : tensor<2xf32>
// FORWARD-NEXT:    %7 = stablehlo.sqrt %6 : tensor<2xf32>
// FORWARD-NEXT:    %8 = stablehlo.multiply %3, %7 : tensor<2xf32>
// FORWARD-NEXT:    %9 = stablehlo.divide %arg1, %8 : tensor<2xf32>
// FORWARD-NEXT:    %10 = chlo.acosh %arg0 : tensor<2xf32> -> tensor<2xf32>
// FORWARD-NEXT:    return %10, %9 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
// REVERSE-NEXT:    %0 = chlo.constant dense<1.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %1 = arith.addf %arg1, %cst : tensor<2xf32>
// REVERSE-NEXT:    %2 = stablehlo.multiply %arg0, %arg0 : tensor<2xf32>
// REVERSE-NEXT:    %3 = stablehlo.subtract %2, %0 : tensor<2xf32>
// REVERSE-NEXT:    %4 = stablehlo.sqrt %3 : tensor<2xf32>
// REVERSE-NEXT:    %5 = stablehlo.multiply %arg0, %arg0 : tensor<2xf32>
// REVERSE-NEXT:    %6 = stablehlo.add %5, %0 : tensor<2xf32>
// REVERSE-NEXT:    %7 = stablehlo.sqrt %6 : tensor<2xf32>
// REVERSE-NEXT:    %8 = stablehlo.multiply %4, %7 : tensor<2xf32>
// REVERSE-NEXT:    %9 = stablehlo.divide %1, %8 : tensor<2xf32>
// REVERSE-NEXT:    %10 = arith.addf %9, %cst : tensor<2xf32>
// REVERSE-NEXT:    return %10 : tensor<2xf32>
// REVERSE-NEXT:  }
