// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @imag_of_exp(%arg0: tensor<complex<f64>>) -> tensor<f64> {
  %0 = stablehlo.exponential %arg0 : tensor<complex<f64>>
  %1 = stablehlo.imag %0 : (tensor<complex<f64>>) -> tensor<f64>
  return %1 : tensor<f64>
}

// imag(x) == 0 is selected to 0, as XLA does, so x = 710 + 0i still gives 0
// rather than exp(710) * sin(0) = inf * 0 = NaN. The non-zero case uses the
// same overflow fallback as the real part.
// CHECK-LABEL: func.func @imag_of_exp(
// CHECK-DAG:   %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG:   %[[HALF:.+]] = stablehlo.constant dense<5.000000e-01> : tensor<f64>
// CHECK-DAG:   %[[INF:.+]] = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
// CHECK-DAG:   %[[RE:.+]] = stablehlo.real %arg0
// CHECK-DAG:   %[[IM:.+]] = stablehlo.imag %arg0
// CHECK-DAG:   %[[SIN:.+]] = stablehlo.sine %[[IM]]
// CHECK-DAG:   %[[EXP:.+]] = stablehlo.exponential %[[RE]]
// CHECK-DAG:   %[[REH:.+]] = stablehlo.multiply %[[RE]], %[[HALF]]
// CHECK-DAG:   %[[EXPH:.+]] = stablehlo.exponential %[[REH]]
// CHECK-DAG:   %[[ISINF:.+]] = stablehlo.compare  EQ, %[[EXP]], %[[INF]]
// CHECK-DAG:   %[[NORMAL:.+]] = stablehlo.multiply %[[EXP]], %[[SIN]]
// CHECK-DAG:   %[[PART:.+]] = stablehlo.multiply %[[EXPH]], %[[SIN]]
// CHECK-DAG:   %[[OVERFLOW:.+]] = stablehlo.multiply %[[PART]], %[[EXPH]]
// CHECK-DAG:   %[[NONZERO:.+]] = stablehlo.select %[[ISINF]], %[[OVERFLOW]], %[[NORMAL]]
// CHECK-DAG:   %[[ISZERO:.+]] = stablehlo.compare  EQ, %[[IM]], %[[ZERO]]
// CHECK:       %[[SEL:.+]] = stablehlo.select %[[ISZERO]], %[[ZERO]], %[[NONZERO]]
// CHECK:       return %[[SEL]]
