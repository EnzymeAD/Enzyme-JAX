// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// Test that FFT of zero is optimized to zero
func.func @fft_of_zero() -> tensor<12x12xcomplex<f64>> {
  %cst_0 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<12x12xcomplex<f64>>
  %0 = stablehlo.fft %cst_0, type = FFT, length = [12, 12] : (tensor<12x12xcomplex<f64>>) -> tensor<12x12xcomplex<f64>>
  return %0 : tensor<12x12xcomplex<f64>>
}

// CHECK-LABEL: func.func @fft_of_zero
// CHECK-NEXT:    %[[CST:.*]] = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<12x12xcomplex<f64>>
// CHECK-NEXT:    return %[[CST]] : tensor<12x12xcomplex<f64>>

// Test that IFFT of zero is optimized to zero
func.func @ifft_of_zero() -> tensor<4xcomplex<f32>> {
  %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<4xcomplex<f32>>
  %0 = stablehlo.fft %cst, type = IFFT, length = [4] : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
  return %0 : tensor<4xcomplex<f32>>
}

// CHECK-LABEL: func.func @ifft_of_zero
// CHECK-NEXT:    %[[CST:.*]] = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<4xcomplex<f32>>
// CHECK-NEXT:    return %[[CST]] : tensor<4xcomplex<f32>>

// Test that RFFT of zero is optimized to zero (output type is different)
func.func @rfft_of_zero() -> tensor<3xcomplex<f32>> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
  %0 = stablehlo.fft %cst, type = RFFT, length = [4] : (tensor<4xf32>) -> tensor<3xcomplex<f32>>
  return %0 : tensor<3xcomplex<f32>>
}

// CHECK-LABEL: func.func @rfft_of_zero
// CHECK-NEXT:    %[[CST:.*]] = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<3xcomplex<f32>>
// CHECK-NEXT:    return %[[CST]] : tensor<3xcomplex<f32>>

// Test that IRFFT of zero is optimized to zero (output type is different)
func.func @irfft_of_zero() -> tensor<4xf32> {
  %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<3xcomplex<f32>>
  %0 = stablehlo.fft %cst, type = IRFFT, length = [4] : (tensor<3xcomplex<f32>>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @irfft_of_zero
// CHECK-NEXT:    %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
// CHECK-NEXT:    return %[[CST]] : tensor<4xf32>

// Test that FFT of non-zero is not optimized away
func.func @fft_of_nonzero(%arg0: tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %0 = stablehlo.fft %arg0, type = FFT, length = [4] : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
  return %0 : tensor<4xcomplex<f32>>
}

// CHECK-LABEL: func.func @fft_of_nonzero
// CHECK:         %[[FFT:.*]] = stablehlo.fft
// CHECK:         return %[[FFT]]
