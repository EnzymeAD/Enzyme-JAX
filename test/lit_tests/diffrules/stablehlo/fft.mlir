// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%x : tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>> {
  %y = "stablehlo.fft"(%x) {
    fft_type = #stablehlo<fft_type FFT>,
    fft_length = array<i64 : 2>
  } : (tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  func.return %y : tensor<2xcomplex<f32>>
}

// FORWARD:  func.func @main(%arg0: tensor<2xcomplex<f32>>, %arg1: tensor<2xcomplex<f32>>) -> (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) {
// FORWARD-NEXT:       %0 = stablehlo.fft %arg1, type =  FFT, length = [2] : (tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
// FORWARD-NEXT:       %1 = stablehlo.fft %arg0, type =  FFT, length = [2] : (tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
// FORWARD-NEXT:       return %1, %0 : tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>
// FORWARD-NEXT:   }

