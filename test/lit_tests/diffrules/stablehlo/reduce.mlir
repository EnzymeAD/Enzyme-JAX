// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%x : tensor<3680000x3xf32>) -> tensor<3680000xf32> {
  %cst_76 = stablehlo.constant dense<2.7> : tensor<f32>
  %res = stablehlo.reduce(%x init: %cst_76) applies stablehlo.add across dimensions = [1] : (tensor<3680000x3xf32>, tensor<f32>) -> tensor<3680000xf32>
  func.return %res : tensor<3680000xf32>
}

// FORWARD:  func.func @main(%arg0: tensor<3680000x3xf32>, %arg1: tensor<3680000x3xf32>) -> (tensor<3680000xf32>, tensor<3680000xf32>) {
// FORWARD-NEXT:    %cst = stablehlo.constant dense<2.700000e+00> : tensor<f32>
// FORWARD-NEXT:    %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
// FORWARD-NEXT:    %0 = stablehlo.reduce(%arg1 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<3680000x3xf32>, tensor<f32>) -> tensor<3680000xf32>
// FORWARD-NEXT:    %1 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<3680000x3xf32>, tensor<f32>) -> tensor<3680000xf32>
// FORWARD-NEXT:    return %1, %0 : tensor<3680000xf32>, tensor<3680000xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<3680000x3xf32>, %arg1: tensor<3680000xf32>) -> tensor<3680000x3xf32> {
// REVERSE-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<3680000xf32>
// REVERSE-NEXT:    %cst_0 = arith.constant dense<0.000000e+00> : tensor<3680000x3xf32>
// REVERSE-NEXT:    %0 = arith.addf %arg1, %cst : tensor<3680000xf32>
// REVERSE-NEXT:    %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<3680000xf32>) -> tensor<3680000x3xf32>
// REVERSE-NEXT:    %2 = arith.addf %1, %cst_0 : tensor<3680000x3xf32>
// REVERSE-NEXT:    return %2 : tensor<3680000x3xf32>
// REVERSE-NEXT:  }
