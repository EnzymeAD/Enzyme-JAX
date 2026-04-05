// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f32 target-type=bf16 concat-dimension=first expansion-size=2" %s | FileCheck %s

func.func @convert_f32_to_f64_expanded(%arg0: tensor<4x4xf32>) -> tensor<4x4xf64> {
  %0 = stablehlo.convert %arg0 : (tensor<4x4xf32>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}
// CHECK-LABEL: func.func @convert_f32_to_f64_expanded
// CHECK: %[[C0:.*]] = stablehlo.convert %arg0 : (tensor<4x4xf32>) -> tensor<4x4xbf16>
// CHECK: %[[C1:.*]] = stablehlo.convert %[[C0]] : (tensor<4x4xbf16>) -> tensor<4x4xf32>
// CHECK: %[[SUB:.*]] = stablehlo.subtract %arg0, %[[C1]] : tensor<4x4xf32>
// CHECK: %[[C2:.*]] = stablehlo.convert %[[SUB]] : (tensor<4x4xf32>) -> tensor<4x4xbf16>
// CHECK: %[[R1:.*]] = stablehlo.reshape %[[C0]] : (tensor<4x4xbf16>) -> tensor<1x4x4xbf16>
// CHECK: %[[R2:.*]] = stablehlo.reshape %[[C2]] : (tensor<4x4xbf16>) -> tensor<1x4x4xbf16>
// CHECK: %[[CAT:.*]] = stablehlo.concatenate %[[R1]], %[[R2]], dim = 0 : (tensor<1x4x4xbf16>, tensor<1x4x4xbf16>) -> tensor<2x4x4xbf16>
// CHECK: %[[C3:.*]] = stablehlo.convert %[[CAT]] : (tensor<2x4x4xbf16>) -> tensor<2x4x4xf64>
// CHECK: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK: %[[RES:.*]] = stablehlo.reduce(%[[C3]] init: %[[ZERO]]) applies stablehlo.add across dimensions = [0] : (tensor<2x4x4xf64>, tensor<f64>) -> tensor<4x4xf64>
// CHECK: return %[[RES]] : tensor<4x4xf64>
