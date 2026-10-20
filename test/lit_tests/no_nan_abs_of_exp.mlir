// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{no_nan=true})" %s | FileCheck %s --check-prefix=NONAN
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{no_nan=false})" %s | FileCheck %s --check-prefix=NAN

// abs(exp(x)) -> exp(real(x)) is wrong when imag(x) is inf or NaN. For
// x = -2 + inf*i the original is NaN while exp(real(x)) is exp(-2), so this
// needs no_nan.
func.func @abs_of_exp(%arg0: tensor<complex<f64>>) -> tensor<f64> {
  %0 = stablehlo.exponential %arg0 : tensor<complex<f64>>
  %1 = stablehlo.abs %0 : (tensor<complex<f64>>) -> tensor<f64>
  return %1 : tensor<f64>
}

// NONAN-LABEL: func.func @abs_of_exp(
// NONAN:       %[[RE:.+]] = stablehlo.real %arg0
// NONAN:       %[[EXP:.+]] = stablehlo.exponential %[[RE]]
// NONAN:       return %[[EXP]]

// NAN-LABEL: func.func @abs_of_exp(
// NAN:       %[[EXP:.+]] = stablehlo.exponential %arg0
// NAN:       %[[ABS:.+]] = stablehlo.abs %[[EXP]]
// NAN:       return %[[ABS]]
