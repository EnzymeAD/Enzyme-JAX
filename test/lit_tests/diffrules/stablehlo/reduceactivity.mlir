// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

module {
  func.func public @main(%arg0: tensor<64000x3xf32>) -> (tensor<i32>) {
    %c_39 = stablehlo.constant dense<64000> : tensor<i32>
    %cst_76 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.reduce(%arg0 init: %cst_76) applies stablehlo.add across dimensions = [1] : (tensor<64000x3xf32>, tensor<f32>) -> tensor<64000xf32>
    return %c_39 : tensor<i32>
  }
}

