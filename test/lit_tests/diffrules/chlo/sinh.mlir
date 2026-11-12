// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined strongZero=true" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=STRONG

func.func @main(%x : tensor<2xf32>) -> tensor<2xf32> {
  %y = chlo.sinh %x : tensor<2xf32> -> tensor<2xf32>
  func.return %y : tensor<2xf32>
}

// FORWARD:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %0 = chlo.cosh %arg0 : tensor<2xf32> -> tensor<2xf32>
// FORWARD-NEXT:    %1 = stablehlo.multiply %arg1, %0 : tensor<2xf32>
// FORWARD-NEXT:    %2 = chlo.sinh %arg0 : tensor<2xf32> -> tensor<2xf32>
// FORWARD-NEXT:    return %2, %1 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
// REVERSE-NEXT:    %0 = chlo.cosh %arg0 : tensor<2xf32> -> tensor<2xf32>
// REVERSE-NEXT:    %1 = stablehlo.multiply %arg1, %0 : tensor<2xf32>
// REVERSE-NEXT:    return %1 : tensor<2xf32>
// REVERSE-NEXT:  }

// STRONG:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
// STRONG-NEXT:    %[[ZERO:.+]] = chlo.constant dense<0.000000e+00> : tensor<2xf32>
// STRONG-NEXT:    %[[COSH:.+]] = chlo.cosh %arg0 : tensor<2xf32> -> tensor<2xf32>
// STRONG-NEXT:    %[[CMP:.+]] = stablehlo.compare EQ, %arg1, %[[ZERO]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// STRONG-NEXT:    %[[MUL:.+]] = stablehlo.multiply %arg1, %[[COSH]] : tensor<2xf32>
// STRONG-NEXT:    %[[SEL:.+]] = stablehlo.select %[[CMP]], %[[ZERO]], %[[MUL]] : tensor<2xi1>, tensor<2xf32>
// STRONG-NEXT:    return %[[SEL]] : tensor<2xf32>
// STRONG-NEXT:  }
