// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=rfft outfn=rfft_fwddiff retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD-RFFT
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=rfft outfn=rfft_revdiff retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --arith-raise --verify-each=0 --canonicalize --remove-unnecessary-enzyme-ops --enzyme-hlo-opt | FileCheck %s --check-prefix=REVERSE-RFFT

func.func @rfft(%x : tensor<4xf32>) -> tensor<3xcomplex<f32>> {
  %y = "stablehlo.fft"(%x) {
    fft_type = #stablehlo<fft_type RFFT>,
    fft_length = array<i64 : 4>
  } : (tensor<4xf32>) -> tensor<3xcomplex<f32>>
  func.return %y : tensor<3xcomplex<f32>>
}

// FORWARD-RFFT:  func.func private @rfft_fwddiff(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<3xcomplex<f32>>, tensor<3xcomplex<f32>>) {
// FORWARD-RFFT-NEXT:       %0 = stablehlo.fft %arg1, type =  RFFT, length = [4] : (tensor<4xf32>) -> tensor<3xcomplex<f32>>
// FORWARD-RFFT-NEXT:       %1 = stablehlo.fft %arg0, type =  RFFT, length = [4] : (tensor<4xf32>) -> tensor<3xcomplex<f32>>
// FORWARD-RFFT-NEXT:       return %1, %0 : tensor<3xcomplex<f32>>, tensor<3xcomplex<f32>>
// FORWARD-RFFT-NEXT:   }

// REVERSE-RFFT:  func.func private @rfft_revdiff(%arg0: tensor<4xf32>, %arg1: tensor<3xcomplex<f32>>) -> tensor<4xf32> {
// REVERSE-RFFT-NEXT:    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
// REVERSE-RFFT-NEXT:    %0 = chlo.conj %arg1 : tensor<3xcomplex<f32>> -> tensor<3xcomplex<f32>>
// REVERSE-RFFT-NEXT:    %1 = stablehlo.pad %0, %cst, low = [0], high = [1], interior = [0] : (tensor<3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<4xcomplex<f32>>
// REVERSE-RFFT-NEXT:    %2 = stablehlo.fft %1, type = FFT, length = [4] {enzymexla.complex_is_purely_imaginary = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
// REVERSE-RFFT-NEXT:    %3 = stablehlo.real %2 : (tensor<4xcomplex<f32>>) -> tensor<4xf32>
// REVERSE-RFFT-NEXT:    return %3 : tensor<4xf32>
// REVERSE-RFFT-NEXT:  }
