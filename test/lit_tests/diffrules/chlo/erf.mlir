// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%x : tensor<2xf32>) -> tensor<2xf32> {
  %y = chlo.erf %x : tensor<2xf32> -> tensor<2xf32>
  func.return %y : tensor<2xf32>
}

// FORWARD:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %0 = chlo.constant dense<{{1\.128379[0-9]*}}> : tensor<2xf32>
// FORWARD-NEXT:    %1 = stablehlo.multiply %arg0, %arg0 : tensor<2xf32>
// FORWARD-NEXT:    %2 = stablehlo.negate %1 : tensor<2xf32>
// FORWARD-NEXT:    %3 = stablehlo.exponential %2 : tensor<2xf32>
// FORWARD-NEXT:    %4 = stablehlo.multiply %0, %3 : tensor<2xf32>
// FORWARD-NEXT:    %5 = stablehlo.multiply %arg1, %4 : tensor<2xf32>
// FORWARD-NEXT:    %6 = chlo.erf %arg0 : tensor<2xf32> -> tensor<2xf32>
// FORWARD-NEXT:    return %6, %5 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
// REVERSE-NEXT:    %0 = chlo.constant dense<{{1\.128379[0-9]*}}> : tensor<2xf32>
// REVERSE-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %1 = arith.addf %arg1, %cst : tensor<2xf32>
// REVERSE-NEXT:    %2 = stablehlo.multiply %arg0, %arg0 : tensor<2xf32>
// REVERSE-NEXT:    %3 = stablehlo.negate %2 : tensor<2xf32>
// REVERSE-NEXT:    %4 = stablehlo.exponential %3 : tensor<2xf32>
// REVERSE-NEXT:    %5 = stablehlo.multiply %0, %4 : tensor<2xf32>
// REVERSE-NEXT:    %6 = stablehlo.multiply %1, %5 : tensor<2xf32>
// REVERSE-NEXT:    %7 = arith.addf %6, %cst : tensor<2xf32>
// REVERSE-NEXT:    return %7 : tensor<2xf32>
// REVERSE-NEXT:  }
