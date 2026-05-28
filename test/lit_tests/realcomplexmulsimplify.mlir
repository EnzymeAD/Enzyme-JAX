// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=real_complex_mul_simplify" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect | FileCheck %s

func.func @main(%arg0: tensor<4368xf64>, %arg1: tensor<4368xcomplex<f64>>) -> tensor<4368xcomplex<f64>> {
  // Add the attribute right here so the pattern knows it is allowed to fire!
  %0 = stablehlo.convert %arg0 {enzymexla.complex_is_purely_real = [#enzymexla<guaranteed GUARANTEED>]} : (tensor<4368xf64>) -> tensor<4368xcomplex<f64>>
  %1 = stablehlo.multiply %0, %arg1 : tensor<4368xcomplex<f64>>
  return %1 : tensor<4368xcomplex<f64>>
}

// CHECK: %[[REAL_LHS:.*]] = stablehlo.real %0
// CHECK: %[[REAL_RHS:.*]] = stablehlo.real %arg1
// CHECK: %[[IMAG_RHS:.*]] = stablehlo.imag %arg1
// CHECK: %[[MUL_REAL:.*]] = stablehlo.multiply %[[REAL_LHS]], %[[REAL_RHS]]
// CHECK: %[[MUL_IMAG:.*]] = stablehlo.multiply %[[REAL_LHS]], %[[IMAG_RHS]]
// CHECK: %[[COMPLEX:.*]] = stablehlo.complex %[[MUL_REAL]], %[[MUL_IMAG]]
// CHECK: return %[[COMPLEX]]