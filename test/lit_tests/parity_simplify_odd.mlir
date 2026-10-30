// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td=patterns=parity_simplify --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

func.func @sin_neg(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.negate %arg0 : tensor<4xf32>
  %1 = stablehlo.sine %0 : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// CHECK-LABEL: func.func @sin_neg
// CHECK-NEXT:    %[[SIN:.+]] = stablehlo.sine %arg0 : tensor<4xf32>
// CHECK-NEXT:    %[[NEG:.+]] = stablehlo.negate %[[SIN]] : tensor<4xf32>
// CHECK-NEXT:    return %[[NEG]] : tensor<4xf32>

func.func @tanh_neg_complex(%arg0: tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %0 = stablehlo.negate %arg0 : tensor<4xcomplex<f32>>
  %1 = stablehlo.tanh %0 : tensor<4xcomplex<f32>>
  return %1 : tensor<4xcomplex<f32>>
}

// CHECK-LABEL: func.func @tanh_neg_complex
// CHECK-NEXT:    %[[TANH:.+]] = stablehlo.tanh %arg0 : tensor<4xcomplex<f32>>
// CHECK-NEXT:    %[[NEG:.+]] = stablehlo.negate %[[TANH]] : tensor<4xcomplex<f32>>
// CHECK-NEXT:    return %[[NEG]] : tensor<4xcomplex<f32>>

func.func @real_neg(%arg0: tensor<4xcomplex<f32>>) -> tensor<4xf32> {
  %0 = stablehlo.negate %arg0 : tensor<4xcomplex<f32>>
  %1 = stablehlo.real %0 : (tensor<4xcomplex<f32>>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// CHECK-LABEL: func.func @real_neg
// CHECK-NEXT:    %[[REAL:.+]] = stablehlo.real %arg0 : (tensor<4xcomplex<f32>>) -> tensor<4xf32>
// CHECK-NEXT:    %[[NEG:.+]] = stablehlo.negate %[[REAL]] : tensor<4xf32>
// CHECK-NEXT:    return %[[NEG]] : tensor<4xf32>

func.func @imag_neg(%arg0: tensor<4xcomplex<f32>>) -> tensor<4xf32> {
  %0 = stablehlo.negate %arg0 : tensor<4xcomplex<f32>>
  %1 = stablehlo.imag %0 : (tensor<4xcomplex<f32>>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// CHECK-LABEL: func.func @imag_neg
// CHECK-NEXT:    %[[IMAG:.+]] = stablehlo.imag %arg0 : (tensor<4xcomplex<f32>>) -> tensor<4xf32>
// CHECK-NEXT:    %[[NEG:.+]] = stablehlo.negate %[[IMAG]] : tensor<4xf32>
// CHECK-NEXT:    return %[[NEG]] : tensor<4xf32>

func.func @conj_neg(%arg0: tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %0 = stablehlo.negate %arg0 : tensor<4xcomplex<f32>>
  %1 = chlo.conj %0 : tensor<4xcomplex<f32>> -> tensor<4xcomplex<f32>>
  return %1 : tensor<4xcomplex<f32>>
}

// CHECK-LABEL: func.func @conj_neg
// CHECK-NEXT:    %[[CONJ:.+]] = chlo.conj %arg0 : tensor<4xcomplex<f32>> -> tensor<4xcomplex<f32>>
// CHECK-NEXT:    %[[NEG:.+]] = stablehlo.negate %[[CONJ]] : tensor<4xcomplex<f32>>
// CHECK-NEXT:    return %[[NEG]] : tensor<4xcomplex<f32>>

func.func @erf_neg(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.negate %arg0 : tensor<4xf32>
  %1 = chlo.erf %0 : tensor<4xf32> -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// CHECK-LABEL: func.func @erf_neg
// CHECK-NEXT:    %[[ERF:.+]] = chlo.erf %arg0 : tensor<4xf32> -> tensor<4xf32>
// CHECK-NEXT:    %[[NEG:.+]] = stablehlo.negate %[[ERF]] : tensor<4xf32>
// CHECK-NEXT:    return %[[NEG]] : tensor<4xf32>

func.func @erf_inv_neg(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.negate %arg0 : tensor<4xf32>
  %1 = chlo.erf_inv %0 : tensor<4xf32> -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// CHECK-LABEL: func.func @erf_inv_neg
// CHECK-NEXT:    %[[ERFINV:.+]] = chlo.erf_inv %arg0 : tensor<4xf32> -> tensor<4xf32>
// CHECK-NEXT:    %[[NEG:.+]] = stablehlo.negate %[[ERFINV]] : tensor<4xf32>
// CHECK-NEXT:    return %[[NEG]] : tensor<4xf32>

// The odd rewrite would keep the original negate alive and add another one.
func.func @sin_neg_multi_use(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = stablehlo.negate %arg0 : tensor<4xf32>
  %1 = stablehlo.sine %0 : tensor<4xf32>
  return %0, %1 : tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: func.func @sin_neg_multi_use
// CHECK-NEXT:    %[[NEG:.+]] = stablehlo.negate %arg0 : tensor<4xf32>
// CHECK-NEXT:    %[[SIN:.+]] = stablehlo.sine %[[NEG]] : tensor<4xf32>
// CHECK-NEXT:    return %[[NEG]], %[[SIN]] : tensor<4xf32>, tensor<4xf32>

// sign(-INT_MIN) = -1 but -sign(INT_MIN) = 1.
func.func @sign_neg_int(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.negate %arg0 : tensor<4xi32>
  %1 = stablehlo.sign %0 : tensor<4xi32>
  return %1 : tensor<4xi32>
}

// CHECK-LABEL: func.func @sign_neg_int
// CHECK-NEXT:    %[[NEG:.+]] = stablehlo.negate %arg0 : tensor<4xi32>
// CHECK-NEXT:    %[[SIGN:.+]] = stablehlo.sign %[[NEG]] : tensor<4xi32>
// CHECK-NEXT:    return %[[SIGN]] : tensor<4xi32>

// Odd functions do not simplify through abs.
func.func @sin_abs(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.abs %arg0 : tensor<4xf32>
  %1 = stablehlo.sine %0 : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// CHECK-LABEL: func.func @sin_abs
// CHECK-NEXT:    %[[ABS:.+]] = stablehlo.abs %arg0 : tensor<4xf32>
// CHECK-NEXT:    %[[SIN:.+]] = stablehlo.sine %[[ABS]] : tensor<4xf32>
// CHECK-NEXT:    return %[[SIN]] : tensor<4xf32>
