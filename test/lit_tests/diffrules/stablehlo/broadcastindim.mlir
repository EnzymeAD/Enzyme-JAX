// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup,enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active,enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%arg1: tensor<5x5x3xf32>, %arg2: tensor<10xf32>) -> (tensor<1x3x5x5xf32>, tensor<10x3xf32>) {
  %1 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<10xf32>) -> tensor<10x3xf32>
  %2 = stablehlo.broadcast_in_dim %arg1, dims = [3, 2, 1] : (tensor<5x5x3xf32>) -> tensor<1x3x5x5xf32>
  return %2, %1 : tensor<1x3x5x5xf32>, tensor<10x3xf32>
}

// FORWARD: func.func @main(%arg0: tensor<5x5x3xf32>, %arg1: tensor<5x5x3xf32>, %arg2: tensor<10xf32>, %arg3: tensor<10xf32>) -> (tensor<1x3x5x5xf32>, tensor<1x3x5x5xf32>, tensor<10x3xf32>, tensor<10x3xf32>) {
// FORWARD-NEXT:   %0 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<10xf32>) -> tensor<10x3xf32>
// FORWARD-NEXT:   %1 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<10xf32>) -> tensor<10x3xf32>
// FORWARD-NEXT:   %2 = stablehlo.broadcast_in_dim %arg1, dims = [3, 2, 1] : (tensor<5x5x3xf32>) -> tensor<1x3x5x5xf32>
// FORWARD-NEXT:   %3 = stablehlo.broadcast_in_dim %arg0, dims = [3, 2, 1] : (tensor<5x5x3xf32>) -> tensor<1x3x5x5xf32>
// FORWARD-NEXT:   return %3, %2, %1, %0 : tensor<1x3x5x5xf32>, tensor<1x3x5x5xf32>, tensor<10x3xf32>, tensor<10x3xf32>
// FORWARD-NEXT: }

// REVERSE: func.func @main(%arg0: tensor<5x5x3xf32>, %arg1: tensor<10xf32>, %arg2: tensor<1x3x5x5xf32>, %arg3: tensor<10x3xf32>) -> (tensor<5x5x3xf32>, tensor<10xf32>) {
// REVERSE-NEXT:   %cst = arith.constant dense<0.000000e+00> : tensor<f32>
// REVERSE-NEXT:   %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x3x5x5xf32>
// REVERSE-NEXT:   %cst_1 = arith.constant dense<0.000000e+00> : tensor<10x3xf32>
// REVERSE-NEXT:   %cst_2 = arith.constant dense<0.000000e+00> : tensor<5x5x3xf32>
// REVERSE-NEXT:   %cst_3 = arith.constant dense<0.000000e+00> : tensor<10xf32>
// REVERSE-NEXT:   %0 = arith.addf %arg2, %cst_0 : tensor<1x3x5x5xf32>
// REVERSE-NEXT:   %1 = arith.addf %arg3, %cst_1 : tensor<10x3xf32>
// REVERSE-NEXT:   %2 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<1x3x5x5xf32>, tensor<f32>) -> tensor<3x5x5xf32>
// REVERSE-NEXT:   %3 = stablehlo.transpose %2, dims = [2, 1, 0] : (tensor<3x5x5xf32>) -> tensor<5x5x3xf32>
// REVERSE-NEXT:   %4 = arith.addf %3, %cst_2 : tensor<5x5x3xf32>
// REVERSE-NEXT:   %5 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<10x3xf32>, tensor<f32>) -> tensor<10xf32>
// REVERSE-NEXT:   %6 = stablehlo.transpose %5, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
// REVERSE-NEXT:   %7 = arith.addf %6, %cst_3 : tensor<10xf32>
// REVERSE-NEXT:   return %4, %7 : tensor<5x5x3xf32>, tensor<10xf32>
// REVERSE-NEXT: }
