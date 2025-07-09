// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --verify-each=0 | FileCheck %s --check-prefix=REVERSE

func.func @main(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    %0 = enzymexla.ml.relu %arg0 : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
}

// FORWARD: func.func @main(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x3x4xf32>) -> (tensor<2x3x4xf32>, tensor<2x3x4xf32>) {
// FORWARD-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<2x3x4xf32>
// FORWARD-NEXT:     %0 = stablehlo.compare  LT, %arg0, %cst : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xi1>
// FORWARD-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<2x3x4xf32>
// FORWARD-NEXT:     %1 = stablehlo.select %0, %cst_0, %arg1 : tensor<2x3x4xi1>, tensor<2x3x4xf32>
// FORWARD-NEXT:     %2 = enzymexla.ml.relu %arg0 : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
// FORWARD-NEXT:     return %2, %1 : tensor<2x3x4xf32>, tensor<2x3x4xf32>
// FORWARD-NEXT: }

// REVERSE: func.func @main(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
// REVERSE-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<2x3x4xf32>
// REVERSE-NEXT:     %0 = stablehlo.add %cst, %arg1 : tensor<2x3x4xf32>
// REVERSE-NEXT:     %1 = stablehlo.compare  LT, %arg0, %cst : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xi1>
// REVERSE-NEXT:     %2 = stablehlo.select %1, %cst, %0 : tensor<2x3x4xi1>, tensor<2x3x4xf32>
// REVERSE-NEXT:     %3 = stablehlo.add %cst, %2 : tensor<2x3x4xf32>
// REVERSE-NEXT:     return %3 : tensor<2x3x4xf32>
// REVERSE-NEXT: }
