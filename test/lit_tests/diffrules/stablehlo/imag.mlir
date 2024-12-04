// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup,enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active,enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --verify-each=0 | FileCheck %s --check-prefix=REVERSE

func.func @main(%arg0 : tensor<2xcomplex<f32>>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %0 = stablehlo.imag %arg0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
  %1 = stablehlo.imag %arg1 : (tensor<2xf32>) -> tensor<2xf32>
  return %0, %1 : tensor<2xf32>, tensor<2xf32>
}

// FORWARD:  func.func @main(%arg0: tensor<2xcomplex<f32>>, %arg1: tensor<2xcomplex<f32>>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.imag %arg1 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// FORWARD-NEXT:    %1 = stablehlo.imag %arg0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// FORWARD-NEXT:    %2 = stablehlo.imag %arg3 : tensor<2xf32>
// FORWARD-NEXT:    %3 = stablehlo.imag %arg2 : tensor<2xf32>
// FORWARD-NEXT:    return %1, %0, %3, %2 : tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<2xcomplex<f32>>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>) -> (tensor<2xcomplex<f32>>, tensor<2xf32>) {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %cst_0 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<2xcomplex<f32>>
// REVERSE-NEXT:    %0 = stablehlo.add %cst, %arg2 : tensor<2xf32>
// REVERSE-NEXT:    %1 = stablehlo.add %cst, %cst : tensor<2xf32>
// REVERSE-NEXT:    %2 = stablehlo.negate %0 : tensor<2xf32>
// REVERSE-NEXT:    %3 = stablehlo.complex %cst, %2 : tensor<2xcomplex<f32>>
// REVERSE-NEXT:    %4 = chlo.conj %3 : tensor<2xcomplex<f32>> -> tensor<2xcomplex<f32>>
// REVERSE-NEXT:    %5 = stablehlo.add %cst_0, %4 : tensor<2xcomplex<f32>>
// REVERSE-NEXT:    return %5, %1 : tensor<2xcomplex<f32>>, tensor<2xf32>
// REVERSE-NEXT:  }
