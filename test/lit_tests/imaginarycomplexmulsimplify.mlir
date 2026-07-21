// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=imaginary_complex_mul_simplify" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect | FileCheck %s

func.func @main(%arg0: tensor<4368xcomplex<f64>>, %arg1: tensor<4368xcomplex<f64>>) -> tensor<4368xcomplex<f64>> {
  %0 = stablehlo.reshape %arg0 {enzymexla.complex_is_purely_imaginary = [#enzymexla<guaranteed GUARANTEED>]} : (tensor<4368xcomplex<f64>>) -> tensor<4368xcomplex<f64>>
  %1 = stablehlo.multiply %0, %arg1 : tensor<4368xcomplex<f64>>
  return %1 : tensor<4368xcomplex<f64>>
}

// CHECK: %[[IMAG_LHS:.*]] = stablehlo.imag %0
// CHECK: %[[REAL_RHS:.*]] = stablehlo.real %arg1
// CHECK: %[[IMAG_RHS:.*]] = stablehlo.imag %arg1
// CHECK: %[[MUL_IMAGS:.*]] = stablehlo.multiply %[[IMAG_LHS]], %[[IMAG_RHS]]
// CHECK: %[[NEG_REAL:.*]] = stablehlo.negate %[[MUL_IMAGS]]
// CHECK: %[[MUL_MIXED:.*]] = stablehlo.multiply %[[IMAG_LHS]], %[[REAL_RHS]]
// CHECK: %[[COMPLEX:.*]] = stablehlo.complex %[[NEG_REAL]], %[[MUL_MIXED]]
// CHECK: return %[[COMPLEX]]