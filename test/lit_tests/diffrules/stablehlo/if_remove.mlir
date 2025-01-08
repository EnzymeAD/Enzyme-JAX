// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | FileCheck %s --check-prefix=REVERSE

module {
  func.func @main(%arg0: tensor<10xf32>, %pred: tensor<i1>) -> tensor<10xf32> {
    %cst = stablehlo.constant dense<1.0> : tensor<10xf32>

    %0 = "stablehlo.if"(%pred) ({
      %1 = stablehlo.multiply %arg0, %cst : tensor<10xf32>
      %2 = stablehlo.multiply %1, %1 : tensor<10xf32>
      "stablehlo.return"(%2) : (tensor<10xf32>) -> ()
    }, {
      "stablehlo.return"(%cst) : (tensor<10xf32>) -> ()
    }) : (tensor<i1>) -> tensor<10xf32>

    return %0 : tensor<10xf32>
  }
}

// REVERSE:  func.func @main(%arg0: tensor<10xf32>, %arg1: tensor<i1>, %arg2: tensor<10xf32>) -> tensor<10xf32> {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<10xf32>
// REVERSE-NEXT:    %0 = stablehlo.select %arg1, %arg0, %cst : tensor<i1>, tensor<10xf32>
// REVERSE-NEXT:    %1 = "stablehlo.if"(%arg1) ({
// REVERSE-NEXT:      %2 = stablehlo.multiply %arg2, %0 : tensor<10xf32>
// REVERSE-NEXT:      %3 = stablehlo.add %2, %2 : tensor<10xf32>
// REVERSE-NEXT:      stablehlo.return %3 : tensor<10xf32>
// REVERSE-NEXT:    }, {
// REVERSE-NEXT:      stablehlo.return %cst : tensor<10xf32>
// REVERSE-NEXT:    }) : (tensor<i1>) -> tensor<10xf32>
// REVERSE-NEXT:    return %1 : tensor<10xf32>
// REVERSE-NEXT:  }
