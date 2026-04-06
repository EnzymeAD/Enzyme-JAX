// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck %s

func.func @reduce_test(%arg0: tensor<2x2xf64>) -> tensor<2xf64> {
  %cst = stablehlo.constant dense<0.0> : tensor<f64>
  %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
  return %0 : tensor<2xf64>
}
