// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%x : tensor<2x3xf32>) -> tensor<2x3xf32> {
  %y = "stablehlo.reverse"(%x) { dimensions = array<i64 : 1> } : (tensor<2x3xf32>) -> tensor<2x3xf32>
  func.return %y : tensor<2x3xf32>
}

// FORWARD:  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.reverse %arg1, dims = [1] : tensor<2x3xf32>
// FORWARD-NEXT:    %1 = stablehlo.reverse %arg0, dims = [1] : tensor<2x3xf32>
// FORWARD-NEXT:    return %1, %0 : tensor<2x3xf32>, tensor<2x3xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
// REVERSE-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<2x3xf32>
// REVERSE-NEXT:    %0 = arith.addf %arg1, %cst : tensor<2x3xf32>
// REVERSE-NEXT:    %1 = stablehlo.reverse %0, dims = [1] : tensor<2x3xf32>
// REVERSE-NEXT:    %2 = arith.addf %1, %cst : tensor<2x3xf32>
// REVERSE-NEXT:    return %2 : tensor<2x3xf32>
// REVERSE-NEXT:  }
