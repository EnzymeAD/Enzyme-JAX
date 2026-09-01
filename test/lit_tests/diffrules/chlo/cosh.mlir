// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%x : tensor<2xf32>) -> tensor<2xf32> {
  %y = chlo.cosh %x : tensor<2xf32> -> tensor<2xf32>
  func.return %y : tensor<2xf32>
}

// FORWARD:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %0 = chlo.sinh %arg0 : tensor<2xf32> -> tensor<2xf32>
// FORWARD-NEXT:    %1 = stablehlo.multiply %arg1, %0 : tensor<2xf32>
// FORWARD-NEXT:    %2 = chlo.cosh %arg0 : tensor<2xf32> -> tensor<2xf32>
// FORWARD-NEXT:    return %2, %1 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
// REVERSE-NEXT:    %0 = chlo.sinh %arg0 : tensor<2xf32> -> tensor<2xf32>
// REVERSE-NEXT:    %1 = stablehlo.multiply %arg1, %0 : tensor<2xf32>
// REVERSE-NEXT:    return %1 : tensor<2xf32>
// REVERSE-NEXT:  }
