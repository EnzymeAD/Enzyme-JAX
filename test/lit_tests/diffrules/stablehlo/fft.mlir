// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=fft outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD-FFT
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=fft outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --arith-raise --verify-each=0 --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE-FFT

func.func @fft(%x : tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %0 = "stablehlo.fft"(%x) {
    fft_type = #stablehlo<fft_type FFT>,
    fft_length = array<i64 : 4>
  } : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
  func.return %0 : tensor<4xcomplex<f32>>
}

// FORWARD-FFT:  func.func @fft(%arg0: tensor<4xcomplex<f32>>, %arg1: tensor<4xcomplex<f32>>) -> (tensor<4xcomplex<f32>>, tensor<4xcomplex<f32>>) {
// FORWARD-FFT-NEXT:       %0 = stablehlo.fft %arg1, type =  FFT, length = [4] : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
// FORWARD-FFT-NEXT:       %1 = stablehlo.fft %arg0, type =  FFT, length = [4] : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
// FORWARD-FFT-NEXT:       return %1, %0 : tensor<4xcomplex<f32>>, tensor<4xcomplex<f32>>
// FORWARD-FFT-NEXT:   }

// REVERSE-FFT:  func.func @fft(%arg0: tensor<4xcomplex<f32>>, %arg1: tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
// REVERSE-FFT-NEXT:    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<4xcomplex<f32>>
// REVERSE-FFT-NEXT:    %0 = stablehlo.add %cst, %arg1 : tensor<4xcomplex<f32>>
// REVERSE-FFT-NEXT:    %1 = chlo.conj %0 : tensor<4xcomplex<f32>> -> tensor<4xcomplex<f32>>
// REVERSE-FFT-NEXT:    %2 = stablehlo.fft %1, type =  FFT, length = [4] : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
// REVERSE-FFT-NEXT:    %3 = chlo.conj %2 : tensor<4xcomplex<f32>> -> tensor<4xcomplex<f32>>
// REVERSE-FFT-NEXT:    %4 = stablehlo.add %cst, %3 : tensor<4xcomplex<f32>>
// REVERSE-FFT-NEXT:    return %4 : tensor<4xcomplex<f32>>
// REVERSE-FFT-NEXT:  }

// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=ifft outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD-IFFT
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=ifft outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --arith-raise --verify-each=0 --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE-IFFT

func.func @ifft(%x : tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %y = "stablehlo.fft"(%x) {
    fft_type = #stablehlo<fft_type IFFT>,
    fft_length = array<i64 : 4>
  } : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
  func.return %y : tensor<4xcomplex<f32>>
}
// FORWARD-IFFT:  func.func @ifft(%arg0: tensor<4xcomplex<f32>>, %arg1: tensor<4xcomplex<f32>>) -> (tensor<4xcomplex<f32>>, tensor<4xcomplex<f32>>) {
// FORWARD-IFFT-NEXT:       %0 = stablehlo.fft %arg1, type =  IFFT, length = [4] : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
// FORWARD-IFFT-NEXT:       %1 = stablehlo.fft %arg0, type =  IFFT, length = [4] : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
// FORWARD-IFFT-NEXT:       return %1, %0 : tensor<4xcomplex<f32>>, tensor<4xcomplex<f32>>
// FORWARD-IFFT-NEXT:   }

// REVERSE-IFFT:  func.func @ifft(%arg0: tensor<4xcomplex<f32>>, %arg1: tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
// REVERSE-IFFT-NEXT:    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<4xcomplex<f32>>
// REVERSE-IFFT-NEXT:    %0 = stablehlo.add %cst, %arg1 : tensor<4xcomplex<f32>>
// REVERSE-IFFT-NEXT:    %1 = chlo.conj %0 : tensor<4xcomplex<f32>> -> tensor<4xcomplex<f32>>
// REVERSE-IFFT-NEXT:    %2 = stablehlo.fft %1, type =  IFFT, length = [4] : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
// REVERSE-IFFT-NEXT:    %3 = chlo.conj %2 : tensor<4xcomplex<f32>> -> tensor<4xcomplex<f32>>
// REVERSE-IFFT-NEXT:    %4 = stablehlo.add %cst, %3 : tensor<4xcomplex<f32>>
// REVERSE-IFFT-NEXT:    return %4 : tensor<4xcomplex<f32>>
// REVERSE-IFFT-NEXT:  }

// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=rfft outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD-RFFT
// TODO: enzymexlamlir-opt %s --enzyme-wrap="infn=rfft outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE-RFFT

func.func @rfft(%x : tensor<4xf32>) -> tensor<3xcomplex<f32>> {
  %y = "stablehlo.fft"(%x) {
    fft_type = #stablehlo<fft_type RFFT>,
    fft_length = array<i64 : 4>
  } : (tensor<4xf32>) -> tensor<3xcomplex<f32>>
  func.return %y : tensor<3xcomplex<f32>>
}

// FORWARD-RFFT:  func.func @rfft(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<3xcomplex<f32>>, tensor<3xcomplex<f32>>) {
// FORWARD-RFFT-NEXT:       %0 = stablehlo.fft %arg1, type =  RFFT, length = [4] : (tensor<4xf32>) -> tensor<3xcomplex<f32>>
// FORWARD-RFFT-NEXT:       %1 = stablehlo.fft %arg0, type =  RFFT, length = [4] : (tensor<4xf32>) -> tensor<3xcomplex<f32>>
// FORWARD-RFFT-NEXT:       return %1, %0 : tensor<3xcomplex<f32>>, tensor<3xcomplex<f32>>
// FORWARD-RFFT-NEXT:   }

// REVERSE-RFFT: func.func @rfft(%arg0: tensor<4xf32>, %arg1: tensor<3xcomplex<f32>>) -> tensor<4xf32> {
// REVERSE-RFFT-NEXT:     %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
// REVERSE-RFFT-NEXT:     %cst_0 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<3xcomplex<f32>>
// REVERSE-RFFT-NEXT:     %0 = stablehlo.add %cst_0, %arg1 : tensor<3xcomplex<f32>>
// REVERSE-RFFT-NEXT:     %1 = chlo.conj %0 : tensor<3xcomplex<f32>> -> tensor<3xcomplex<f32>>
// REVERSE-RFFT-NEXT:     %2 = stablehlo.pad %1, %cst, low = [0], high = [1], interior = [0] : (tensor<3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<4xcomplex<f32>>
// REVERSE-RFFT-NEXT:     %3 = stablehlo.fft %2, type =  FFT, length = [4] : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
// REVERSE-RFFT-NEXT:     %4 = stablehlo.real %3 : (tensor<4xcomplex<f32>>) -> tensor<4xf32>
// REVERSE-RFFT-NEXT:     return %4 : tensor<4xf32>
// REVERSE-RFFT-NEXT: }

// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=irfft outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD-IRFFT
// TODO: enzymexlamlir-opt %s --enzyme-wrap="infn=irfft outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE-IRFFT

func.func @irfft(%x : tensor<3xcomplex<f32>>) -> tensor<4xf32> {
  %y = "stablehlo.fft"(%x) {
    fft_type = #stablehlo<fft_type IRFFT>,
    fft_length = array<i64 : 4>
  } : (tensor<3xcomplex<f32>>) -> tensor<4xf32>
  func.return %y : tensor<4xf32>
}

// FORWARD-IRFFT:  func.func @irfft(%arg0: tensor<3xcomplex<f32>>, %arg1: tensor<3xcomplex<f32>>) -> (tensor<4xf32>, tensor<4xf32>) {
// FORWARD-IRFFT-NEXT:       %0 = stablehlo.fft %arg1, type =  IRFFT, length = [4] : (tensor<3xcomplex<f32>>) -> tensor<4xf32>
// FORWARD-IRFFT-NEXT:       %1 = stablehlo.fft %arg0, type =  IRFFT, length = [4] : (tensor<3xcomplex<f32>>) -> tensor<4xf32>
// FORWARD-IRFFT-NEXT:       return %1, %0 : tensor<4xf32>, tensor<4xf32>
// FORWARD-IRFFT-NEXT:   }

// REVERSE_IRFFT: func.func @irfft(%arg0: tensor<3xcomplex<f32>>, %arg1: tensor<4xf32>) -> tensor<3xcomplex<f32>> {
// REVERSE_IRFFT-NEXT:     %cst = stablehlo.constant dense<[2.500000e-01, 5.000000e-01, 2.500000e-01]> : tensor<3xf32>
// REVERSE_IRFFT-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<3xf32>
// REVERSE_IRFFT-NEXT:     %cst_1 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<3xcomplex<f32>>
// REVERSE_IRFFT-NEXT:     %0 = stablehlo.fft %arg1, type =  RFFT, length = [4] : (tensor<4xf32>) -> tensor<3xcomplex<f32>>
// REVERSE_IRFFT-NEXT:     %1 = stablehlo.complex %cst, %cst_0 : tensor<3xcomplex<f32>>
// REVERSE_IRFFT-NEXT:     %2 = stablehlo.multiply %0, %1 : tensor<3xcomplex<f32>>
// REVERSE_IRFFT-NEXT:     %3 = chlo.conj %2 : tensor<3xcomplex<f32>> -> tensor<3xcomplex<f32>>
// REVERSE_IRFFT-NEXT:     %4 = stablehlo.add %cst_1, %3 : tensor<3xcomplex<f32>>
// REVERSE_IRFFT-NEXT:     return %4 : tensor<3xcomplex<f32>>
// REVERSE_IRFFT-NEXT: }
