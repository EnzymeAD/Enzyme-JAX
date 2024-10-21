// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --enzyme-simplify-math | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --verify-each=0 | FileCheck %s --check-prefix=REVERSE

func.func @main(%operand : tensor<2xcomplex<f32>>) -> tensor<2xf32> {
  %result = "stablehlo.imag"(%operand) : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
  return %result : tensor<2xf32>
}

// FORWARD:  func.func @main(%arg0: tensor<2xcomplex<f32>>, %arg1: tensor<2xcomplex<f32>>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.imag %arg1 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// FORWARD-NEXT:    %1 = stablehlo.imag %arg0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// FORWARD-NEXT:    return %1, %0 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<2xcomplex<f32>>, %arg1: tensor<2xf32>) -> tensor<2xcomplex<f32>> {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %cst_0 = arith.constant dense<0.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %cst_1 = arith.constant dense<(0.000000e+00,0.000000e+00)> : tensor<2xcomplex<f32>>
// REVERSE-NEXT:    %0 = stablehlo.add %cst_0, %arg1 : tensor<2xf32>
// REVERSE-NEXT:    %1 = stablehlo.negate %0 : tensor<2xf32>
// REVERSE-NEXT:    %2 = stablehlo.complex %cst, %1 : tensor<2xcomplex<f32>>
// REVERSE-NEXT:    %3 = chlo.conj %2 : tensor<2xcomplex<f32>> -> tensor<2xcomplex<f32>>
// REVERSE-NEXT:    %4 = stablehlo.add %cst_1, %3 : tensor<2xcomplex<f32>>
// REVERSE-NEXT:    return %4 : tensor<2xcomplex<f32>>
// REVERSE-NEXT: }
