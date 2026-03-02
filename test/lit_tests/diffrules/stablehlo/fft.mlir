// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=fft outfn=fft_fwddiff retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD-FFT
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=fft outfn=fft_revdiff retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --arith-raise --verify-each=0 --canonicalize --remove-unnecessary-enzyme-ops --enzyme-hlo-opt | FileCheck %s --check-prefix=REVERSE-FFT

func.func @fft(%x : tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %0 = "stablehlo.fft"(%x) {
    fft_type = #stablehlo<fft_type FFT>,
    fft_length = array<i64 : 4>
  } : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
  func.return %0 : tensor<4xcomplex<f32>>
}

// FORWARD-FFT:  func.func private @fft_fwddiff(%arg0: tensor<4xcomplex<f32>>, %arg1: tensor<4xcomplex<f32>>) -> (tensor<4xcomplex<f32>>, tensor<4xcomplex<f32>>) {
// FORWARD-FFT-NEXT:       %0 = stablehlo.fft %arg1, type =  FFT, length = [4] : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
// FORWARD-FFT-NEXT:       %1 = stablehlo.fft %arg0, type =  FFT, length = [4] : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
// FORWARD-FFT-NEXT:       return %1, %0 : tensor<4xcomplex<f32>>, tensor<4xcomplex<f32>>
// FORWARD-FFT-NEXT:   }

// REVERSE-FFT:  func.func private @fft_revdiff(%arg0: tensor<4xcomplex<f32>>, %arg1: tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
// REVERSE-FFT-NEXT:    %0 = chlo.conj %arg1 : tensor<4xcomplex<f32>> -> tensor<4xcomplex<f32>>
// REVERSE-FFT-NEXT:    %1 = stablehlo.fft %0, type = FFT, length = [4] : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
// REVERSE-FFT-NEXT:    return %1 : tensor<4xcomplex<f32>>
// REVERSE-FFT-NEXT:  }
