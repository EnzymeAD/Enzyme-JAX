// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @real_of_exp(%arg0: tensor<complex<f64>>) -> tensor<f64> {
  %0 = stablehlo.exponential %arg0 : tensor<complex<f64>>
  %1 = stablehlo.real %0 : (tensor<complex<f64>>) -> tensor<f64>
  return %1 : tensor<f64>
}

// When exp(real(x)) overflows, the result falls back to
// exp(real(x) / 2) * cos(imag(x)) * exp(real(x) / 2) as XLA does, so
// x = 709.9 + 1i in f64 (or 89 + 1i in f32) stays finite instead of inf.
// CHECK-LABEL: func.func @real_of_exp(
// CHECK-DAG:   %[[HALF:.+]] = stablehlo.constant dense<5.000000e-01> : tensor<f64>
// CHECK-DAG:   %[[INF:.+]] = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
// CHECK-DAG:   %[[RE:.+]] = stablehlo.real %arg0
// CHECK-DAG:   %[[IM:.+]] = stablehlo.imag %arg0
// CHECK-DAG:   %[[COS:.+]] = stablehlo.cosine %[[IM]]
// CHECK-DAG:   %[[EXP:.+]] = stablehlo.exponential %[[RE]]
// CHECK-DAG:   %[[REH:.+]] = stablehlo.multiply %[[RE]], %[[HALF]]
// CHECK-DAG:   %[[EXPH:.+]] = stablehlo.exponential %[[REH]]
// CHECK-DAG:   %[[ISINF:.+]] = stablehlo.compare  EQ, %[[EXP]], %[[INF]]
// CHECK-DAG:   %[[NORMAL:.+]] = stablehlo.multiply %[[EXP]], %[[COS]]
// CHECK-DAG:   %[[PART:.+]] = stablehlo.multiply %[[EXPH]], %[[COS]]
// CHECK-DAG:   %[[OVERFLOW:.+]] = stablehlo.multiply %[[PART]], %[[EXPH]]
// CHECK:       %[[SEL:.+]] = stablehlo.select %[[ISINF]], %[[OVERFLOW]], %[[NORMAL]]
// CHECK:       return %[[SEL]]

// The rewrite is skipped when the exp has another user, so it stays as-is.
func.func @real_of_exp_multi_use(%arg0: tensor<complex<f64>>)
    -> (tensor<f64>, tensor<complex<f64>>) {
  %0 = stablehlo.exponential %arg0 : tensor<complex<f64>>
  %1 = stablehlo.real %0 : (tensor<complex<f64>>) -> tensor<f64>
  return %1, %0 : tensor<f64>, tensor<complex<f64>>
}

// CHECK-LABEL: func.func @real_of_exp_multi_use(
// CHECK:       %[[EXP:.+]] = stablehlo.exponential %arg0
// CHECK:       %[[REAL:.+]] = stablehlo.real %[[EXP]]
// CHECK:       return %[[REAL]], %[[EXP]]
