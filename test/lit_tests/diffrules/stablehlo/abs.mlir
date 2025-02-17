// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --verify-each=0 | FileCheck %s --check-prefix=REVERSE

module {
  func.func @main(%arg0: tensor<2xcomplex<f32>>) -> tensor<2xf32> {
    %0 = stablehlo.abs %arg0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }
}

// FORWARD:  func.func @main(%arg0: tensor<2xcomplex<f32>>, %arg1: tensor<2xcomplex<f32>>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.multiply %arg1, %arg0 : tensor<2xcomplex<f32>>
// FORWARD-NEXT:    %1 = stablehlo.real %0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// FORWARD-NEXT:    %2 = stablehlo.abs %arg0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// FORWARD-NEXT:    %3 = stablehlo.divide %1, %2 : tensor<2xf32>
// FORWARD-NEXT:    %4 = stablehlo.abs %arg0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// FORWARD-NEXT:    return %4, %3 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<2xcomplex<f32>>, %arg1: tensor<2xf32>) -> tensor<2xcomplex<f32>> {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<2xcomplex<f32>>
// REVERSE-NEXT:    %0 = stablehlo.real %arg0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// REVERSE-NEXT:    %1 = stablehlo.abs %arg0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// REVERSE-NEXT:    %2 = stablehlo.divide %0, %1 : tensor<2xf32>
// REVERSE-NEXT:    %3 = stablehlo.imag %arg0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// REVERSE-NEXT:    %4 = stablehlo.abs %arg0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// REVERSE-NEXT:    %5 = stablehlo.divide %3, %4 : tensor<2xf32>
// REVERSE-NEXT:    %6 = stablehlo.negate %5 : tensor<2xf32>
// REVERSE-NEXT:    %7 = stablehlo.complex %2, %6 : tensor<2xcomplex<f32>>
// REVERSE-NEXT:    %8 = chlo.conj %7 : tensor<2xcomplex<f32>> -> tensor<2xcomplex<f32>>
// REVERSE-NEXT:    %9 = stablehlo.add %cst, %8 : tensor<2xcomplex<f32>>
// REVERSE-NEXT:    return %9 : tensor<2xcomplex<f32>>
// REVERSE-NEXT:  }
