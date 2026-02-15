// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=irfft outfn=irfft_fwddiff retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD-IRFFT
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=irfft outfn=irfft_revdiff retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --arith-raise --verify-each=0 --canonicalize --remove-unnecessary-enzyme-ops --enzyme-hlo-opt | FileCheck %s --check-prefix=REVERSE-IRFFT

func.func @irfft(%x : tensor<3xcomplex<f32>>) -> tensor<4xf32> {
  %y = "stablehlo.fft"(%x) {
    fft_type = #stablehlo<fft_type IRFFT>,
    fft_length = array<i64 : 4>
  } : (tensor<3xcomplex<f32>>) -> tensor<4xf32>
  func.return %y : tensor<4xf32>
}

// FORWARD-IRFFT:  func.func private @irfft_fwddiff(%arg0: tensor<3xcomplex<f32>>, %arg1: tensor<3xcomplex<f32>>) -> (tensor<4xf32>, tensor<4xf32>) {
// FORWARD-IRFFT-NEXT:       %0 = stablehlo.fft %arg1, type =  IRFFT, length = [4] : (tensor<3xcomplex<f32>>) -> tensor<4xf32>
// FORWARD-IRFFT-NEXT:       %1 = stablehlo.fft %arg0, type =  IRFFT, length = [4] : (tensor<3xcomplex<f32>>) -> tensor<4xf32>
// FORWARD-IRFFT-NEXT:       return %1, %0 : tensor<4xf32>, tensor<4xf32>
// FORWARD-IRFFT-NEXT:   }

// REVERSE-IRFFT: func.func private @irfft_revdiff(%arg0: tensor<3xcomplex<f32>>, %arg1: tensor<4xf32>) -> tensor<3xcomplex<f32>> {
// REVERSE-IRFFT-NEXT:      %cst = stablehlo.constant dense<[(2.500000e-01,0.000000e+00), (5.000000e-01,0.000000e+00), (2.500000e-01,0.000000e+00)]> : tensor<3xcomplex<f32>>
// REVERSE-IRFFT-NEXT:      %0 = stablehlo.fft %arg1, type =  RFFT, length = [4] : (tensor<4xf32>) -> tensor<3xcomplex<f32>>
// REVERSE-IRFFT-NEXT:      %1 = stablehlo.multiply %0, %cst : tensor<3xcomplex<f32>>
// REVERSE-IRFFT-NEXT:      %2 = chlo.conj %1 : tensor<3xcomplex<f32>> -> tensor<3xcomplex<f32>>
// REVERSE-IRFFT-NEXT:      return %2 : tensor<3xcomplex<f32>>
// REVERSE-IRFFT-NEXT:    }
