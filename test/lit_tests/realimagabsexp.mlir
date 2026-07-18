// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @real_of_exp(%arg0: tensor<complex<f64>>) -> tensor<f64> {
  %0 = stablehlo.exponential %arg0 : tensor<complex<f64>>
  %1 = stablehlo.real %0 : (tensor<complex<f64>>) -> tensor<f64>
  return %1 : tensor<f64>
}

// CHECK-LABEL: func.func @real_of_exp(
// CHECK-DAG:   %[[RE:.+]] = stablehlo.real %arg0
// CHECK-DAG:   %[[IM:.+]] = stablehlo.imag %arg0
// CHECK-DAG:   %[[EXP:.+]] = stablehlo.exponential %[[RE]]
// CHECK-DAG:   %[[COS:.+]] = stablehlo.cosine %[[IM]]
// CHECK:       %[[MUL:.+]] = stablehlo.multiply %[[EXP]], %[[COS]]
// CHECK:       return %[[MUL]]

func.func @imag_of_exp(%arg0: tensor<complex<f64>>) -> tensor<f64> {
  %0 = stablehlo.exponential %arg0 : tensor<complex<f64>>
  %1 = stablehlo.imag %0 : (tensor<complex<f64>>) -> tensor<f64>
  return %1 : tensor<f64>
}

// CHECK-LABEL: func.func @imag_of_exp(
// CHECK-DAG:   %[[RE:.+]] = stablehlo.real %arg0
// CHECK-DAG:   %[[IM:.+]] = stablehlo.imag %arg0
// CHECK-DAG:   %[[EXP:.+]] = stablehlo.exponential %[[RE]]
// CHECK-DAG:   %[[SIN:.+]] = stablehlo.sine %[[IM]]
// CHECK:       %[[MUL:.+]] = stablehlo.multiply %[[EXP]], %[[SIN]]
// CHECK:       return %[[MUL]]

func.func @abs_of_exp(%arg0: tensor<complex<f64>>) -> tensor<f64> {
  %0 = stablehlo.exponential %arg0 : tensor<complex<f64>>
  %1 = stablehlo.abs %0 : (tensor<complex<f64>>) -> tensor<f64>
  return %1 : tensor<f64>
}

// CHECK-LABEL: func.func @abs_of_exp(
// CHECK:       %[[RE:.+]] = stablehlo.real %arg0
// CHECK:       %[[EXP:.+]] = stablehlo.exponential %[[RE]]
// CHECK:       return %[[EXP]]
