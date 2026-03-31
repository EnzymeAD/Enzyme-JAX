// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f32 target-type=bf16 concat-dimension=first expansion-size=2" %s | FileCheck %s

func.func @convert_f32_to_f64_expanded(%arg0: tensor<4x4xf32>) -> tensor<4x4xf64> {
  %0 = stablehlo.convert %arg0 : (tensor<4x4xf32>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}
// CHECK-LABEL: func.func @convert_f32_to_f64_expanded
// CHECK: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK: %[[OUT:.*]] = stablehlo.reduce(%{{.*}} init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x4x4xf64>, tensor<f64>) -> tensor<4x4xf64>
// CHECK: return %[[OUT]]
