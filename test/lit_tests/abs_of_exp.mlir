// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{no_nan=false})" %s | FileCheck %s --check-prefix=NAN
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{no_nan=true})" %s | FileCheck %s --check-prefix=NONAN

// abs(exp(x)) -> exp(real(x)) is wrong when imag(x) is inf or NaN. For
// x = -2 + inf*i the original is NaN while exp(real(x)) is exp(-2), so by
// default the result is guarded on imag(x) being finite, and with no_nan the
// guard is dropped.
func.func @abs_of_exp(%arg0: tensor<complex<f64>>) -> tensor<f64> {
  %0 = stablehlo.exponential %arg0 : tensor<complex<f64>>
  %1 = stablehlo.abs %0 : (tensor<complex<f64>>) -> tensor<f64>
  return %1 : tensor<f64>
}

// NAN-LABEL: func.func @abs_of_exp(
// NAN-DAG:   %[[NAN:.+]] = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
// NAN-DAG:   %[[RE:.+]] = stablehlo.real %arg0
// NAN-DAG:   %[[IM:.+]] = stablehlo.imag %arg0
// NAN-DAG:   %[[FINITE:.+]] = stablehlo.is_finite %[[IM]]
// NAN-DAG:   %[[EXP:.+]] = stablehlo.exponential %[[RE]]
// NAN:       %[[SEL:.+]] = stablehlo.select %[[FINITE]], %[[EXP]], %[[NAN]]
// NAN:       return %[[SEL]]

// NONAN-LABEL: func.func @abs_of_exp(
// NONAN:       %[[RE:.+]] = stablehlo.real %arg0
// NONAN:       %[[EXP:.+]] = stablehlo.exponential %[[RE]]
// NONAN:       return %[[EXP]]
