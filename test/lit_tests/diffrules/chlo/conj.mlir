// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --arith-raise --verify-each=0 --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%x : tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>> {
  %y = chlo.conj %x : tensor<2xcomplex<f32>> -> tensor<2xcomplex<f32>>
  func.return %y : tensor<2xcomplex<f32>>
}

// FORWARD:  func.func @main(%arg0: tensor<2xcomplex<f32>>, %arg1: tensor<2xcomplex<f32>>) -> (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) {
// FORWARD-NEXT:    %0 = chlo.conj %arg1 : tensor<2xcomplex<f32>> -> tensor<2xcomplex<f32>>
// FORWARD-NEXT:    %1 = chlo.conj %arg0 : tensor<2xcomplex<f32>> -> tensor<2xcomplex<f32>>
// FORWARD-NEXT:    return %1, %0 : tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<2xcomplex<f32>>, %arg1: tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>> {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<2xcomplex<f32>>
// REVERSE-NEXT:    %0 = stablehlo.add %cst, %arg1 : tensor<2xcomplex<f32>>
// REVERSE-NEXT:    %1 = chlo.conj %0 : tensor<2xcomplex<f32>> -> tensor<2xcomplex<f32>>
// REVERSE-NEXT:    %2 = chlo.conj %1 : tensor<2xcomplex<f32>> -> tensor<2xcomplex<f32>>
// REVERSE-NEXT:    %3 = chlo.conj %2 : tensor<2xcomplex<f32>> -> tensor<2xcomplex<f32>>
// REVERSE-NEXT:    %4 = stablehlo.add %cst, %3 : tensor<2xcomplex<f32>>
// REVERSE-NEXT:    return %4 : tensor<2xcomplex<f32>>
// REVERSE-NEXT:  }
