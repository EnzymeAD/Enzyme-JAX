// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_out argTys=enzyme_out mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

module {

  func.func @main(%a : tensor<2x3xf32>) -> tensor<18x27xf32> {
    %0 = stablehlo.constant dense<3.140000e+00> : tensor<f32>
    %2 = stablehlo.pad %a, %0, low = [5, 7], high = [11, 13], interior = [0, 2] : (tensor<2x3xf32>, tensor<f32>) -> tensor<18x27xf32>
    return %2 : tensor<18x27xf32>
  }
}

// FORWARD:  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> (tensor<18x27xf32>, tensor<18x27xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.constant dense<3.140000e+00> : tensor<f32>
// FORWARD-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
// FORWARD-NEXT:    %1 = stablehlo.pad %arg1, %cst, low = [5, 7], high = [11, 13], interior = [0, 2] : (tensor<2x3xf32>, tensor<f32>) -> tensor<18x27xf32>
// FORWARD-NEXT:    %2 = stablehlo.pad %arg0, %0, low = [5, 7], high = [11, 13], interior = [0, 2] : (tensor<2x3xf32>, tensor<f32>) -> tensor<18x27xf32>
// FORWARD-NEXT:    return %2, %1 : tensor<18x27xf32>, tensor<18x27xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<18x27xf32>) -> tensor<2x3xf32> {
// REVERSE-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<18x27xf32>
// REVERSE-NEXT:    %cst_0 = arith.constant dense<0.000000e+00> : tensor<2x3xf32>
// REVERSE-NEXT:    %0 = arith.addf %arg1, %cst : tensor<18x27xf32>
// REVERSE-NEXT:    %1 = stablehlo.slice %0 [5:7, 7:14:3] : (tensor<18x27xf32>) -> tensor<2x3xf32>
// REVERSE-NEXT:    %2 = arith.addf %1, %cst_0 : tensor<2x3xf32>
// REVERSE-NEXT:    return %2 : tensor<2x3xf32>
// REVERSE-NEXT:  }
