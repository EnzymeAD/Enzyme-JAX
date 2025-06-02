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

// RUN: enzymexlamlir-opt --pass-pipeline='builtin.module(enzyme{postpasses="arith-raise{stablehlo=true},canonicalize,cse,canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math,canonicalize,cse,canonicalize"},remove-unnecessary-enzyme-ops,inline,enzyme-hlo-opt)' %s | FileCheck %s --check-prefix=REVERSE-RFFT

module {
  func.func private @"Const{typeof(fn)}(Main.fn)_autodiff"(%arg0: tensor<4xf64>) -> (tensor<f64>, tensor<4xf64>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.fft %arg0, type =  RFFT, length = [4] : (tensor<4xf64>) -> tensor<3xcomplex<f64>>
    %1 = chlo.conj %0 : tensor<3xcomplex<f64>> -> tensor<3xcomplex<f64>>
    %2 = stablehlo.multiply %0, %1 : tensor<3xcomplex<f64>>
    %3 = stablehlo.real %2 : (tensor<3xcomplex<f64>>) -> tensor<3xf64>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
    return %4, %arg0 : tensor<f64>, tensor<4xf64>
  }
  func.func @main(%arg0: tensor<4xf64> {tf.aliasing_output = 1 : i32}) -> (tensor<4xf64>, tensor<4xf64>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
    %0:2 = enzyme.autodiff @"Const{typeof(fn)}(Main.fn)_autodiff"(%arg0, %cst, %cst_0) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_active>]} : (tensor<4xf64>, tensor<f64>, tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>)
    return %0#1, %0#0 : tensor<4xf64>, tensor<4xf64>
  }
}

// REVERSE-RFFT: func.func @main(%arg0: tensor<4xf64> {tf.aliasing_output = 1 : i32}) -> (tensor<4xf64>, tensor<4xf64>) {
// REVERSE-RFFT-NEXT:      %cst = stablehlo.constant dense<1.000000e+00> : tensor<3xf64>
// REVERSE-RFFT-NEXT:      %cst_0 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>>
// REVERSE-RFFT-NEXT:      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<3xf64>
// REVERSE-RFFT-NEXT:      %0 = stablehlo.fft %arg0, type =  RFFT, length = [4] : (tensor<4xf64>) -> tensor<3xcomplex<f64>>
// REVERSE-RFFT-NEXT:      %1 = chlo.conj %0 : tensor<3xcomplex<f64>> -> tensor<3xcomplex<f64>>
// REVERSE-RFFT-NEXT:      %2 = stablehlo.complex %cst, %cst_1 : tensor<3xcomplex<f64>>
// REVERSE-RFFT-NEXT:      %3 = stablehlo.multiply %2, %1 : tensor<3xcomplex<f64>>
// REVERSE-RFFT-NEXT:      %4 = chlo.conj %3 : tensor<3xcomplex<f64>> -> tensor<3xcomplex<f64>>
// REVERSE-RFFT-NEXT:      %5 = stablehlo.multiply %2, %0 : tensor<3xcomplex<f64>>
// REVERSE-RFFT-NEXT:      %6 = stablehlo.add %4, %5 : tensor<3xcomplex<f64>>
// REVERSE-RFFT-NEXT:      %7 = chlo.conj %6 : tensor<3xcomplex<f64>> -> tensor<3xcomplex<f64>>
// REVERSE-RFFT-NEXT:      %8 = stablehlo.pad %7, %cst_0, low = [0], high = [1], interior = [0] : (tensor<3xcomplex<f64>>, tensor<complex<f64>>) -> tensor<4xcomplex<f64>>
// REVERSE-RFFT-NEXT:      %9 = stablehlo.fft %8, type =  FFT, length = [4] : (tensor<4xcomplex<f64>>) -> tensor<4xcomplex<f64>>
// REVERSE-RFFT-NEXT:      %10 = stablehlo.real %9 : (tensor<4xcomplex<f64>>) -> tensor<4xf64>
// REVERSE-RFFT-NEXT:      return %10, %arg0 : tensor<4xf64>, tensor<4xf64>
// REVERSE-RFFT-NEXT:    }

// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=irfft outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD-IRFFT

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

// RUN: enzymexlamlir-opt --pass-pipeline='builtin.module(enzyme{postpasses="arith-raise{stablehlo=true},canonicalize,cse,canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math,canonicalize,cse,canonicalize"},remove-unnecessary-enzyme-ops,inline,enzyme-hlo-opt)' %s | FileCheck %s --check-prefix=REVERSE-IRFFT

module @reactant_gradient attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func private @"Const{typeof(fn3)}(Main.fn3)_autodiff"(%arg0: tensor<3xcomplex<f64>>) -> (tensor<f64>, tensor<3xcomplex<f64>>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.fft %arg0, type =  IRFFT, length = [4] : (tensor<3xcomplex<f64>>) -> tensor<4xf64>
    %1 = stablehlo.multiply %0, %0 : tensor<4xf64>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<4xf64>, tensor<f64>) -> tensor<f64>
    return %2, %arg0 : tensor<f64>, tensor<3xcomplex<f64>>
  }
  func.func @main(%arg0: tensor<3xcomplex<f64>> {tf.aliasing_output = 1 : i32}) -> (tensor<3xcomplex<f64>>, tensor<3xcomplex<f64>>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %cst_0 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<3xcomplex<f64>>
    %0:2 = enzyme.autodiff @"Const{typeof(fn3)}(Main.fn3)_autodiff"(%arg0, %cst, %cst_0) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_active>]} : (tensor<3xcomplex<f64>>, tensor<f64>, tensor<3xcomplex<f64>>) -> (tensor<3xcomplex<f64>>, tensor<3xcomplex<f64>>)
    return %0#1, %0#0 : tensor<3xcomplex<f64>>, tensor<3xcomplex<f64>>
  }
}

// REVERSE-IRFFT: func.func @main(%arg0: tensor<3xcomplex<f64>> {tf.aliasing_output = 1 : i32}) -> (tensor<3xcomplex<f64>>, tensor<3xcomplex<f64>>) {
// REVERSE-IRFFT-NEXT:      %cst = stablehlo.constant dense<[2.500000e-01, 5.000000e-01, 2.500000e-01]> : tensor<3xf64>
// REVERSE-IRFFT-NEXT:      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<3xf64>
// REVERSE-IRFFT-NEXT:      %0 = stablehlo.fft %arg0, type =  IRFFT, length = [4] : (tensor<3xcomplex<f64>>) -> tensor<4xf64>
// REVERSE-IRFFT-NEXT:      %1 = stablehlo.add %0, %0 : tensor<4xf64>
// REVERSE-IRFFT-NEXT:      %2 = stablehlo.fft %1, type =  RFFT, length = [4] : (tensor<4xf64>) -> tensor<3xcomplex<f64>>
// REVERSE-IRFFT-NEXT:      %3 = stablehlo.complex %cst, %cst_0 : tensor<3xcomplex<f64>>
// REVERSE-IRFFT-NEXT:      %4 = stablehlo.multiply %2, %3 : tensor<3xcomplex<f64>>
// REVERSE-IRFFT-NEXT:      return %4, %arg0 : tensor<3xcomplex<f64>>, tensor<3xcomplex<f64>>
// REVERSE-IRFFT-NEXT:    }
