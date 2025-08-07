// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup,enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" --arith-raise --enzyme-hlo-opt | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active,enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | FileCheck %s --check-prefix=REVERSE

func.func @main(%arg1: tensor<5x4x3xf32>, %arg2: tensor<10xf32>) -> (tensor<3x4x10x5xf32>, tensor<10x3xf32>) {
  %1 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<10xf32>) -> tensor<10x3xf32>
  %2 = stablehlo.broadcast_in_dim %arg1, dims = [3, 1, 0] : (tensor<5x4x3xf32>) -> tensor<3x4x10x5xf32>
  return %2, %1 : tensor<3x4x10x5xf32>, tensor<10x3xf32>
}

// FORWARD: func.func @main(%arg0: tensor<5x4x3xf32>, %arg1: tensor<5x4x3xf32>, %arg2: tensor<10xf32>, %arg3: tensor<10xf32>) -> (tensor<3x4x10x5xf32>, tensor<3x4x10x5xf32>, tensor<10x3xf32>, tensor<10x3xf32>) {
// FORWARD-NEXT:     %0 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<10xf32>) -> tensor<10x3xf32>
// FORWARD-NEXT:     %1 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<10xf32>) -> tensor<10x3xf32>
// FORWARD-NEXT:     %2 = stablehlo.broadcast_in_dim %arg1, dims = [3, 1, 0] : (tensor<5x4x3xf32>) -> tensor<3x4x10x5xf32>
// FORWARD-NEXT:     %3 = stablehlo.broadcast_in_dim %arg0, dims = [3, 1, 0] : (tensor<5x4x3xf32>) -> tensor<3x4x10x5xf32>
// FORWARD-NEXT:     return %3, %2, %1, %0 : tensor<3x4x10x5xf32>, tensor<3x4x10x5xf32>, tensor<10x3xf32>, tensor<10x3xf32>
// FORWARD-NEXT: }

// REVERSE:  func.func @main(%arg0: tensor<5x4x3xf32>, %arg1: tensor<10xf32>, %arg2: tensor<3x4x10x5xf32>, %arg3: tensor<10x3xf32>) -> (tensor<5x4x3xf32>, tensor<10xf32>) {
// REVERSE-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// REVERSE-NEXT:     %0 = stablehlo.reduce(%arg2 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<3x4x10x5xf32>, tensor<f32>) -> tensor<3x4x5xf32>
// REVERSE-NEXT:     %1 = stablehlo.transpose %0, dims = [2, 1, 0] : (tensor<3x4x5xf32>) -> tensor<5x4x3xf32>
// REVERSE-NEXT:     %2 = stablehlo.reduce(%arg3 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<10x3xf32>, tensor<f32>) -> tensor<10xf32>
// REVERSE-NEXT:     return %1, %2 : tensor<5x4x3xf32>, tensor<10xf32>
// REVERSE-NEXT: }
