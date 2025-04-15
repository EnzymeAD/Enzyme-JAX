// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%arg2: tensor<10xf32>) -> (tensor<10x3xf32>) {
  %1 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<10xf32>) -> tensor<10x3xf32>
  return %1 : tensor<10x3xf32>
}

// FORWARD:  func.func @main(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> (tensor<10x3xf32>, tensor<10x3xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<10xf32>) -> tensor<10x3xf32>
// FORWARD-NEXT:    %1 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<10xf32>) -> tensor<10x3xf32>
// FORWARD-NEXT:    return %1, %0 : tensor<10x3xf32>, tensor<10x3xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<10xf32>, %arg1: tensor<10x3xf32>) -> tensor<10xf32> {
// REVERSE-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
// REVERSE-NEXT:    %cst_0 = arith.constant dense<0.000000e+00> : tensor<10x3xf32>
// REVERSE-NEXT:    %cst_1 = arith.constant dense<0.000000e+00> : tensor<10xf32>
// REVERSE-NEXT:    %0 = arith.addf %arg1, %cst_0 : tensor<10x3xf32>
// REVERSE-NEXT:    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<10x3xf32>, tensor<f32>) -> tensor<10xf32>
// REVERSE-NEXT:    %2 = stablehlo.reshape %1 : (tensor<10xf32>) -> tensor<10x1xf32>
// REVERSE-NEXT:    %3 = stablehlo.transpose %2, dims = [0, 1] : (tensor<10x1xf32>) -> tensor<10x1xf32>
// REVERSE-NEXT:    %4 = stablehlo.reshape %3 : (tensor<10x1xf32>) -> tensor<10xf32>
// REVERSE-NEXT:    %5 = arith.addf %4, %cst_1 : tensor<10xf32>
// REVERSE-NEXT:    return %5 : tensor<10xf32>
// REVERSE-NEXT:  }
