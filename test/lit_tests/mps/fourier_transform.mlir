// RUN: enzymexlamlir-opt --split-input-file %s | enzymexlamlir-opt --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @fast_fourier_transform(
func.func @fast_fourier_transform(%c: tensor<8xcomplex<f32>>) {
  // CHECK: %{{.+}} = mps.fast_fourier_transform %{{.+}} <axes = [0], inverse = false, scaling_mode = none, round_to_odd_hermitean = false> : (tensor<8xcomplex<f32>>) -> tensor<8xcomplex<f32>>
  %0 = mps.fast_fourier_transform %c <axes = [0], inverse = false, scaling_mode = none, round_to_odd_hermitean = false> : (tensor<8xcomplex<f32>>) -> tensor<8xcomplex<f32>>
  return
}

// -----

// CHECK-LABEL: func.func @real_to_hermitean_fft(
func.func @real_to_hermitean_fft(%r: tensor<8xf32>) {
  // CHECK: %{{.+}} = mps.real_to_hermitean_fft %{{.+}} <axes = [0], inverse = false, scaling_mode = size, round_to_odd_hermitean = false> : (tensor<8xf32>) -> tensor<5xcomplex<f32>>
  %0 = mps.real_to_hermitean_fft %r <axes = [0], inverse = false, scaling_mode = size, round_to_odd_hermitean = false> : (tensor<8xf32>) -> tensor<5xcomplex<f32>>
  return
}

// -----

// CHECK-LABEL: func.func @hermitean_to_real_fft(
func.func @hermitean_to_real_fft(%h: tensor<5xcomplex<f32>>) {
  // CHECK: %{{.+}} = mps.hermitean_to_real_fft %{{.+}} <axes = [0], inverse = true, scaling_mode = unitary, round_to_odd_hermitean = false> : (tensor<5xcomplex<f32>>) -> tensor<8xf32>
  %0 = mps.hermitean_to_real_fft %h <axes = [0], inverse = true, scaling_mode = unitary, round_to_odd_hermitean = false> : (tensor<5xcomplex<f32>>) -> tensor<8xf32>
  return
}
