// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main_fail(%arg0: tensor<2x2xf64>, %arg1: tensor<2x2xf64>, %arg2: tensor<2x2xf64>, %arg3: tensor<2x2xf64>) -> (tensor<2x2xcomplex<f64>>) {
    %0 = stablehlo.complex %arg0, %arg1 : tensor<2x2xcomplex<f64>>
    %1 = stablehlo.complex %arg2, %arg3 : tensor<2x2xcomplex<f64>>
    %2 = stablehlo.add %0, %1 : tensor<2x2xcomplex<f64>>
    return %2 : tensor<2x2xcomplex<f64>>
  }
}

// CHECK: func.func @main_fail(%arg0: tensor<2x2xf64>, %arg1: tensor<2x2xf64>, %arg2: tensor<2x2xf64>, %arg3: tensor<2x2xf64>) -> tensor<2x2xcomplex<f64>> {
// CHECK-NEXT:   %0 = stablehlo.complex %arg0, %arg1 {enzymexla.complex_is_purely_imaginary = [#enzymexla<guaranteed NOTGUARANTEED>], enzymexla.complex_is_purely_real = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2x2xcomplex<f64>>
// CHECK-NEXT:   %1 = stablehlo.complex %arg2, %arg3 {enzymexla.complex_is_purely_imaginary = [#enzymexla<guaranteed NOTGUARANTEED>], enzymexla.complex_is_purely_real = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2x2xcomplex<f64>>
// CHECK-NEXT:   %2 = stablehlo.add %0, %1 : tensor<2x2xcomplex<f64>>
// CHECK-NEXT:   return %2 : tensor<2x2xcomplex<f64>>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<2x2xf64>, %arg1: tensor<2x2xf64>, %arg2: tensor<2x2xf64>) -> (tensor<2x2xcomplex<f64>>) {
    %cst = stablehlo.constant dense<0.00e+0> : tensor<2x2xf64>
    %0 = stablehlo.complex %arg0, %arg1 : tensor<2x2xcomplex<f64>>
    %1 = stablehlo.complex %arg2, %cst : tensor<2x2xcomplex<f64>>
    %2 = stablehlo.add %0, %1 : tensor<2x2xcomplex<f64>>
    return %2 : tensor<2x2xcomplex<f64>>
  }
}

// CHECK: func.func @main(%arg0: tensor<2x2xf64>, %arg1: tensor<2x2xf64>, %arg2: tensor<2x2xf64>) -> tensor<2x2xcomplex<f64>> {
// CHECK-NEXT:   %0 = stablehlo.add %arg0, %arg2 : tensor<2x2xf64>
// CHECK-NEXT:   %1 = stablehlo.complex %0, %arg1 : tensor<2x2xcomplex<f64>>
// CHECK-NEXT:   return %1 : tensor<2x2xcomplex<f64>>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<2x2xf64>, %arg1: tensor<2x2xf64>, %arg2: tensor<2x2xf64>) -> (tensor<2x2xcomplex<f64>>) {
    %cst = stablehlo.constant dense<0.00e+0> : tensor<2x2xf64>
    %0 = stablehlo.complex %cst, %arg1 : tensor<2x2xcomplex<f64>>
    %1 = stablehlo.complex %arg2, %arg0 : tensor<2x2xcomplex<f64>>
    %2 = stablehlo.subtract %0, %1 : tensor<2x2xcomplex<f64>>
    return %2 : tensor<2x2xcomplex<f64>>
  }
}

// CHECK: func.func @main(%arg0: tensor<2x2xf64>, %arg1: tensor<2x2xf64>, %arg2: tensor<2x2xf64>) -> tensor<2x2xcomplex<f64>> {
// CHECK-NEXT:   %0 = stablehlo.negate %arg2 : tensor<2x2xf64>
// CHECK-NEXT:   %1 = stablehlo.subtract %arg1, %arg0 : tensor<2x2xf64>
// CHECK-NEXT:   %2 = stablehlo.complex %0, %1 : tensor<2x2xcomplex<f64>>
// CHECK-NEXT:   return %2 : tensor<2x2xcomplex<f64>>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<2x2xf64> {enzymexla.memory_effects = []}, %arg1: tensor<2x2xf64> {enzymexla.memory_effects = []}) -> (tensor<2x2xf64>, tensor<2x2xf64>) attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<-0.000000e+00> : tensor<2x2xf64>
    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<2x2xf64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<2x2xf64>
    %0 = stablehlo.complex %arg0, %arg1 : tensor<2x2xcomplex<f64>>
    %1 = stablehlo.fft %0, type =  FFT, length = [2, 2] : (tensor<2x2xcomplex<f64>>) -> tensor<2x2xcomplex<f64>>
    %2 = stablehlo.real %1 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<2x2xcomplex<f64>>) -> tensor<2x2xf64>
    %3 = stablehlo.imag %1 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<2x2xcomplex<f64>>) -> tensor<2x2xf64>
    %4 = stablehlo.multiply %3, %cst_0 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2x2xf64>
    %5 = stablehlo.multiply %2, %cst_0 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2x2xf64>
    %6 = stablehlo.complex %cst_1, %4 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2x2xcomplex<f64>>
    %7 = stablehlo.complex %5, %cst {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2x2xcomplex<f64>>
    %8 = stablehlo.add %6, %7 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2x2xcomplex<f64>>
    %9 = chlo.conj %8 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2x2xcomplex<f64>> -> tensor<2x2xcomplex<f64>>
    %10 = stablehlo.fft %9, type =  FFT, length = [2, 2] : (tensor<2x2xcomplex<f64>>) -> tensor<2x2xcomplex<f64>>
    %11 = stablehlo.real %10 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<2x2xcomplex<f64>>) -> tensor<2x2xf64>
    %12 = stablehlo.imag %10 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<2x2xcomplex<f64>>) -> tensor<2x2xf64>
    return %11, %12 : tensor<2x2xf64>, tensor<2x2xf64>
  }
}

// CHECK: func.func @main(%arg0: tensor<2x2xf64> {enzymexla.memory_effects = []}, %arg1: tensor<2x2xf64> {enzymexla.memory_effects = []}) -> (tensor<2x2xf64>, tensor<2x2xf64>) attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:   %cst = stablehlo.constant dense<2.000000e+00> : tensor<2x2xf64>
// CHECK-NEXT:   %0 = stablehlo.complex %arg0, %arg1 : tensor<2x2xcomplex<f64>>
// CHECK-NEXT:   %1 = stablehlo.fft %0, type =  FFT, length = [2, 2] {enzymexla.complex_is_purely_imaginary = [#enzymexla<guaranteed NOTGUARANTEED>], enzymexla.complex_is_purely_real = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<2x2xcomplex<f64>>) -> tensor<2x2xcomplex<f64>>
// CHECK-NEXT:   %2 = stablehlo.real %1 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<2x2xcomplex<f64>>) -> tensor<2x2xf64>
// CHECK-NEXT:   %3 = stablehlo.imag %1 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<2x2xcomplex<f64>>) -> tensor<2x2xf64>
// CHECK-NEXT:   %4 = stablehlo.multiply %3, %cst {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2x2xf64>
// CHECK-NEXT:   %5 = stablehlo.multiply %2, %cst {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2x2xf64>
// CHECK-NEXT:   %6 = stablehlo.complex %5, %4 {enzymexla.complex_is_purely_real = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2x2xcomplex<f64>>
// CHECK-NEXT:   %7 = chlo.conj %6 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2x2xcomplex<f64>> -> tensor<2x2xcomplex<f64>>
// CHECK-NEXT:   %8 = stablehlo.fft %7, type =  FFT, length = [2, 2] {enzymexla.complex_is_purely_imaginary = [#enzymexla<guaranteed NOTGUARANTEED>], enzymexla.complex_is_purely_real = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<2x2xcomplex<f64>>) -> tensor<2x2xcomplex<f64>>
// CHECK-NEXT:   %9 = stablehlo.real %8 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<2x2xcomplex<f64>>) -> tensor<2x2xf64>
// CHECK-NEXT:   %10 = stablehlo.imag %8 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<2x2xcomplex<f64>>) -> tensor<2x2xf64>
// CHECK-NEXT:   return %9, %10 : tensor<2x2xf64>, tensor<2x2xf64>
// CHECK-NEXT: }
