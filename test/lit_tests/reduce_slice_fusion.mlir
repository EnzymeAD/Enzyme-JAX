// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main(%arg0: tensor<2x2xf64>) -> (tensor<f64>, tensor<2x2xf64>) {
  %0 = stablehlo.slice %arg0 [0:1, 0:1] {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<2x2xf64>) -> tensor<1x1xf64>
  %1 = stablehlo.reshape %0 : (tensor<1x1xf64>) -> tensor<f64>
  %2 = stablehlo.slice %arg0 [1:2, 1:2] {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<2x2xf64>) -> tensor<1x1xf64>
  %3 = stablehlo.reshape %2 : (tensor<1x1xf64>) -> tensor<f64>
  %4 = stablehlo.add %1, %3 : tensor<f64>
  return %4, %arg0 : tensor<f64>, tensor<2x2xf64>
}

// CHECK: func.func @main
