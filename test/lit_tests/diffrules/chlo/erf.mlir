// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined strongZero=true" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=STRONG

func.func @main(%x : tensor<2xf32>) -> tensor<2xf32> {
  %y = chlo.erf %x : tensor<2xf32> -> tensor<2xf32>
  func.return %y : tensor<2xf32>
}

// FORWARD:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %0 = chlo.constant dense<1.12837923> : tensor<2xf32>
// FORWARD-NEXT:    %1 = stablehlo.multiply %arg0, %arg0 : tensor<2xf32>
// FORWARD-NEXT:    %2 = stablehlo.negate %1 : tensor<2xf32>
// FORWARD-NEXT:    %3 = stablehlo.exponential %2 : tensor<2xf32>
// FORWARD-NEXT:    %4 = stablehlo.multiply %0, %3 : tensor<2xf32>
// FORWARD-NEXT:    %5 = stablehlo.multiply %arg1, %4 : tensor<2xf32>
// FORWARD-NEXT:    %6 = chlo.erf %arg0 : tensor<2xf32> -> tensor<2xf32>
// FORWARD-NEXT:    return %6, %5 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
// REVERSE-NEXT:    %0 = chlo.constant dense<1.12837923> : tensor<2xf32>
// REVERSE-NEXT:    %1 = stablehlo.multiply %arg0, %arg0 : tensor<2xf32>
// REVERSE-NEXT:    %2 = stablehlo.negate %1 : tensor<2xf32>
// REVERSE-NEXT:    %3 = stablehlo.exponential %2 : tensor<2xf32>
// REVERSE-NEXT:    %4 = stablehlo.multiply %0, %3 : tensor<2xf32>
// REVERSE-NEXT:    %5 = stablehlo.multiply %arg1, %4 : tensor<2xf32>
// REVERSE-NEXT:    return %5 : tensor<2xf32>
// REVERSE-NEXT:  }

// STRONG:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
// STRONG-NEXT:    %[[ZERO:.+]] = chlo.constant dense<0.000000e+00> : tensor<2xf32>
// STRONG-NEXT:    %[[C:.+]] = chlo.constant dense<1.12837923> : tensor<2xf32>
// STRONG-NEXT:    %[[SQ:.+]] = stablehlo.multiply %arg0, %arg0 : tensor<2xf32>
// STRONG-NEXT:    %[[NEG:.+]] = stablehlo.negate %[[SQ]] : tensor<2xf32>
// STRONG-NEXT:    %[[EXP:.+]] = stablehlo.exponential %[[NEG]] : tensor<2xf32>
// STRONG-NEXT:    %[[DER:.+]] = stablehlo.multiply %[[C]], %[[EXP]] : tensor<2xf32>
// STRONG-NEXT:    %[[CMP:.+]] = stablehlo.compare EQ, %arg1, %[[ZERO]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// STRONG-NEXT:    %[[MUL:.+]] = stablehlo.multiply %arg1, %[[DER]] : tensor<2xf32>
// STRONG-NEXT:    %[[SEL:.+]] = stablehlo.select %[[CMP]], %[[ZERO]], %[[MUL]] : tensor<2xi1>, tensor<2xf32>
// STRONG-NEXT:    return %[[SEL]] : tensor<2xf32>
// STRONG-NEXT:  }
