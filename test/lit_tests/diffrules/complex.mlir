// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%a : tensor<2xf32>, %b : tensor<2xf32>) -> tensor<2xcomplex<f32>> {
  %c = stablehlo.complex %a, %b : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xcomplex<f32>>
  func.return %c : tensor<2xcomplex<f32>>
}

// FORWARD:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>) -> (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) {
// FORWARD-NEXT:    %0 = stablehlo.complex %arg1, %arg3 : tensor<2xcomplex<f32>>
// FORWARD-NEXT:    %1 = stablehlo.complex %arg0, %arg2 : tensor<2xcomplex<f32>>
// FORWARD-NEXT:    return %1, %0 : tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>
// FORWARD-NEXT:  }
