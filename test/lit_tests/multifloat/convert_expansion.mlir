// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first expansion-size=2" %s | FileCheck %s

func.func @convert_f64_to_f32(%arg0: tensor<4x4xf64>) -> tensor<4x4xf32> {
  %0 = stablehlo.convert %arg0 : (tensor<4x4xf64>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
// CHECK-LABEL: func.func @convert_f64_to_f32
// CHECK: %[[C0:.*]] = stablehlo.convert %arg0 : (tensor<4x4xf64>) -> tensor<4x4xf32>
// CHECK: %[[C1:.*]] = stablehlo.convert %[[C0]] : (tensor<4x4xf32>) -> tensor<4x4xf64>
// CHECK: %[[SUB:.*]] = stablehlo.subtract %arg0, %[[C1]] : tensor<4x4xf64>
// CHECK: %[[C2:.*]] = stablehlo.convert %[[SUB]] : (tensor<4x4xf64>) -> tensor<4x4xf32>
// CHECK: %[[R1:.*]] = stablehlo.reshape %[[C0]] : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
// CHECK: %[[R2:.*]] = stablehlo.reshape %[[C2]] : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
// CHECK: %[[CAT:.*]] = stablehlo.concatenate %[[R1]], %[[R2]], dim = 0 : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<2x4x4xf32>
// CHECK: %[[SLICE:.*]] = stablehlo.slice %[[CAT]] [0:1, 0:4, 0:4] : (tensor<2x4x4xf32>) -> tensor<1x4x4xf32>
// CHECK: %[[RES:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1x4x4xf32>) -> tensor<4x4xf32>
// CHECK: return %[[RES]] : tensor<4x4xf32>

func.func @convert_f32_to_f64(%arg0: tensor<4x4xf32>) -> tensor<4x4xf64> {
  %0 = stablehlo.convert %arg0 : (tensor<4x4xf32>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}
// CHECK-LABEL: func.func @convert_f32_to_f64
// CHECK:     %[[V_0:.*]] = stablehlo.reshape %arg0 : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
// CHECK:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<1x4x4xf32>
// CHECK:     %[[V_1:.*]] = stablehlo.concatenate %[[V_0]], %[[CST]], dim = 0 : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<2x4x4xf32>
// CHECK:     %[[V_2:.*]] = stablehlo.convert %[[V_1]] : (tensor<2x4x4xf32>) -> tensor<2x4x4xf64>
// CHECK:     %[[CST_0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK:     %[[V_3:.*]] = stablehlo.reduce(%[[V_2]] init: %[[CST_0]]) applies stablehlo.add across dimensions = [0] : (tensor<2x4x4xf64>, tensor<f64>) -> tensor<4x4xf64>
// CHECK:     return %[[V_3]] : tensor<4x4xf64>

func.func @convert_i32_to_f64(%arg0: tensor<4x4xi32>) -> tensor<4x4xf64> {
  %0 = stablehlo.convert %arg0 : (tensor<4x4xi32>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}
// CHECK-LABEL: func.func @convert_i32_to_f64
// CHECK:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<4x4xi32>) -> tensor<4x4xf32>
// CHECK:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<4x4xf32>) -> tensor<4x4xi32>
// CHECK:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<4x4xi32>
// CHECK:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<4x4xi32>) -> tensor<4x4xf32>
// CHECK:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
// CHECK:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
// CHECK:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 0 : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<2x4x4xf32>
// CHECK:     %[[V_7:.*]] = stablehlo.convert %[[V_6]] : (tensor<2x4x4xf32>) -> tensor<2x4x4xf64>
// CHECK:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK:     %[[V_8:.*]] = stablehlo.reduce(%[[V_7]] init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x4x4xf64>, tensor<f64>) -> tensor<4x4xf64>
// CHECK:     return %[[V_8]] : tensor<4x4xf64>

func.func @convert_f64_to_i32(%arg0: tensor<4x4xf64>) -> tensor<4x4xi32> {
  %0 = stablehlo.convert %arg0 : (tensor<4x4xf64>) -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}
// CHECK-LABEL: func.func @convert_f64_to_i32
// CHECK:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<4x4xf64>) -> tensor<4x4xf32>
// CHECK:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<4x4xf32>) -> tensor<4x4xf64>
// CHECK:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<4x4xf64>
// CHECK:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<4x4xf64>) -> tensor<4x4xf32>
// CHECK:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
// CHECK:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
// CHECK:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 0 : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<2x4x4xf32>
// CHECK:     %[[V_7:.*]] = stablehlo.slice %[[V_6]] [0:1, 0:4, 0:4] : (tensor<2x4x4xf32>) -> tensor<1x4x4xf32>
// CHECK:     %[[V_8:.*]] = stablehlo.reshape %[[V_7]] : (tensor<1x4x4xf32>) -> tensor<4x4xf32>
// CHECK:     %[[V_9:.*]] = stablehlo.convert %[[V_8]] : (tensor<4x4xf32>) -> tensor<4x4xi32>
// CHECK:     %[[V_10:.*]] = stablehlo.slice %[[V_6]] [1:2, 0:4, 0:4] : (tensor<2x4x4xf32>) -> tensor<1x4x4xf32>
// CHECK:     %[[V_11:.*]] = stablehlo.reshape %[[V_10]] : (tensor<1x4x4xf32>) -> tensor<4x4xf32>
// CHECK:     %[[V_12:.*]] = stablehlo.convert %[[V_11]] : (tensor<4x4xf32>) -> tensor<4x4xi32>
// CHECK:     %[[V_13:.*]] = stablehlo.add %[[V_9]], %[[V_12]] : tensor<4x4xi32>
// CHECK:     return %[[V_13]] : tensor<4x4xi32>
