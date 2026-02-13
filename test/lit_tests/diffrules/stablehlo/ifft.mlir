// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=ifft outfn=ifft_fwddiff retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD-IFFT
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=ifft outfn=ifft_revdiff retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --arith-raise --verify-each=0 --canonicalize --remove-unnecessary-enzyme-ops --enzyme-hlo-opt | FileCheck %s --check-prefix=REVERSE-IFFT

func.func @ifft(%x : tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %y = "stablehlo.fft"(%x) {
    fft_type = #stablehlo<fft_type IFFT>,
    fft_length = array<i64 : 4>
  } : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
  func.return %y : tensor<4xcomplex<f32>>
}

// FORWARD-IFFT:  func.func private @ifft_fwddiff(%arg0: tensor<4xcomplex<f32>>, %arg1: tensor<4xcomplex<f32>>) -> (tensor<4xcomplex<f32>>, tensor<4xcomplex<f32>>) {
// FORWARD-IFFT-NEXT:       %0 = stablehlo.fft %arg1, type =  IFFT, length = [4] : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
// FORWARD-IFFT-NEXT:       %1 = stablehlo.fft %arg0, type =  IFFT, length = [4] : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
// FORWARD-IFFT-NEXT:       return %1, %0 : tensor<4xcomplex<f32>>, tensor<4xcomplex<f32>>
// FORWARD-IFFT-NEXT:   }

// REVERSE-IFFT:  func.func private @ifft_revdiff(%arg0: tensor<4xcomplex<f32>>, %arg1: tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
// REVERSE-IFFT-NEXT:    %0 = chlo.conj %arg1 : tensor<4xcomplex<f32>> -> tensor<4xcomplex<f32>>
// REVERSE-IFFT-NEXT:    %1 = stablehlo.fft %0, type = IFFT, length = [4] : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
// REVERSE-IFFT-NEXT:    return %1 : tensor<4xcomplex<f32>>
// REVERSE-IFFT-NEXT:  }
