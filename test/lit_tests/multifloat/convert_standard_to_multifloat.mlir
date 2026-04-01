// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f16 concat-dimension=first expansion-size=2" %s | FileCheck %s

func.func @convert_f32_to_f64_multi(%arg0: tensor<4x4xf32>) -> tensor<4x4xf64> {
  %0 = stablehlo.convert %arg0 : (tensor<4x4xf32>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}
// CHECK-LABEL: func.func @convert_f32_to_f64_multi
// CHECK-SAME: (%[[ARG:.*]]: tensor<4x4xf32>) -> tensor<4x4xf64> {
// CHECK: %[[C1:.*]] = stablehlo.convert %[[ARG]] : (tensor<4x4xf32>) -> tensor<4x4xf16>
// CHECK: %[[C1_EXT:.*]] = stablehlo.convert %[[C1]] : (tensor<4x4xf16>) -> tensor<4x4xf32>
// CHECK: %[[REM:.*]] = stablehlo.subtract %[[ARG]], %[[C1_EXT]] : tensor<4x4xf32>
// CHECK: %[[C2:.*]] = stablehlo.convert %[[REM]] : (tensor<4x4xf32>) -> tensor<4x4xf16>
// CHECK: %[[R1:.*]] = stablehlo.reshape %[[C1]] : (tensor<4x4xf16>) -> tensor<1x4x4xf16>
// CHECK: %[[R2:.*]] = stablehlo.reshape %[[C2]] : (tensor<4x4xf16>) -> tensor<1x4x4xf16>
// CHECK: %[[CONCAT:.*]] = stablehlo.concatenate %[[R1]], %[[R2]], dim = 0 : (tensor<1x4x4xf16>, tensor<1x4x4xf16>) -> tensor<2x4x4xf16>
// CHECK: %[[EXT:.*]] = stablehlo.convert %[[CONCAT]] : (tensor<2x4x4xf16>) -> tensor<2x4x4xf64>
// CHECK: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK: %[[RES:.*]] = stablehlo.reduce(%[[EXT]] init: %[[ZERO]]) applies stablehlo.add across dimensions = [0] : (tensor<2x4x4xf64>, tensor<f64>) -> tensor<4x4xf64>
// CHECK: return %[[RES]]
