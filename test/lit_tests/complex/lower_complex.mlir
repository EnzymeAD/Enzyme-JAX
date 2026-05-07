// RUN: enzymexlamlir-opt --lower-complex-operations="concat-dimension=first" %s | FileCheck %s

// CHECK-LABEL: @add_complex
func.func @add_complex(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>) -> tensor<complex<f32>> {
  // CHECK: %[[REAL0:.*]] = stablehlo.real %arg0
  // CHECK: %[[IMAG0:.*]] = stablehlo.imag %arg0
  // CHECK: %[[REAL1:.*]] = stablehlo.real %arg1
  // CHECK: %[[IMAG1:.*]] = stablehlo.imag %arg1
  // CHECK: %[[ADD:.*]] = stablehlo.add
  // CHECK: %[[RES:.*]] = stablehlo.complex
  // CHECK: return %[[RES]]
  %0 = stablehlo.add %arg0, %arg1 : tensor<complex<f32>>
  return %0 : tensor<complex<f32>>
}

// CHECK-LABEL: @mul_complex
func.func @mul_complex(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>) -> tensor<complex<f32>> {
  // CHECK: stablehlo.subtract
  // CHECK: stablehlo.add
  // CHECK: stablehlo.complex
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<complex<f32>>
  return %0 : tensor<complex<f32>>
}

// CHECK-LABEL: @real_imag
func.func @real_imag(%arg0: tensor<complex<f32>>) -> (tensor<f32>, tensor<f32>) {
  // CHECK: stablehlo.real %arg0
  // CHECK: stablehlo.imag %arg0
  %0 = stablehlo.real %arg0 : (tensor<complex<f32>>) -> tensor<f32>
  %1 = stablehlo.imag %arg0 : (tensor<complex<f32>>) -> tensor<f32>
  return %0, %1 : tensor<f32>, tensor<f32>
}

// CHECK-LABEL: @create_complex
func.func @create_complex(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<complex<f32>> {
  // CHECK: stablehlo.complex %arg0, %arg1
  %0 = stablehlo.complex %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<complex<f32>>
  return %0 : tensor<complex<f32>>
}
