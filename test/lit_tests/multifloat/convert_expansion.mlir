// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first expansion-size=2" %s | FileCheck %s

func.func @convert_f64_to_f32(%arg0: tensor<4x4xf64>) -> tensor<4x4xf32> {
  %0 = stablehlo.convert %arg0 : (tensor<4x4xf64>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
// CHECK-LABEL: func.func @convert_f64_to_f32
// CHECK: %[[EXTRACTED:.*]] = stablehlo.slice %{{.*}} [0:1, 0:4, 0:4]
// CHECK: %[[RESHAPED:.*]] = stablehlo.reshape %[[EXTRACTED]]
// CHECK: return %[[RESHAPED]]

func.func @convert_f32_to_f64(%arg0: tensor<4x4xf32>) -> tensor<4x4xf64> {
  %0 = stablehlo.convert %arg0 : (tensor<4x4xf32>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}
// CHECK-LABEL: func.func @convert_f32_to_f64
// CHECK: %[[RESHAPED:.*]] = stablehlo.reshape %{{.*}} : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
// CHECK: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<1x4x4xf32>
// CHECK: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK: %[[OUT:.*]] = stablehlo.reduce(%{{.*}} init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x4x4xf64>, tensor<f64>) -> tensor<4x4xf64>
// CHECK: return %[[OUT]]

func.func @convert_i32_to_f64(%arg0: tensor<4x4xi32>) -> tensor<4x4xf64> {
  %0 = stablehlo.convert %arg0 : (tensor<4x4xi32>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}
// CHECK-LABEL: func.func @convert_i32_to_f64
// CHECK: %[[HIGH:.*]] = stablehlo.convert %{{.*}} : (tensor<4x4xi32>) -> tensor<4x4xf32>
// CHECK: %[[HIGH_BACK:.*]] = stablehlo.convert %[[HIGH]] : (tensor<4x4xf32>) -> tensor<4x4xi32>
// CHECK: %[[REM:.*]] = stablehlo.subtract %{{.*}}, %[[HIGH_BACK]]
// CHECK: %[[LOW:.*]] = stablehlo.convert %[[REM]] : (tensor<4x4xi32>) -> tensor<4x4xf32>
// CHECK: %[[HIGH_RESHAPED:.*]] = stablehlo.reshape %[[HIGH]] : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
// CHECK: %[[LOW_RESHAPED:.*]] = stablehlo.reshape %[[LOW]] : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
// CHECK: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK: %[[OUT:.*]] = stablehlo.reduce(%{{.*}} init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x4x4xf64>, tensor<f64>) -> tensor<4x4xf64>
// CHECK: return %[[OUT]]

func.func @convert_f64_to_i32(%arg0: tensor<4x4xf64>) -> tensor<4x4xi32> {
  %0 = stablehlo.convert %arg0 : (tensor<4x4xf64>) -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}
// CHECK-LABEL: func.func @convert_f64_to_i32
// CHECK: %[[SLICE0:.*]] = stablehlo.slice %{{.*}} [0:1, 0:4, 0:4]
// CHECK: %[[RESHAPED0:.*]] = stablehlo.reshape %[[SLICE0]]
// CHECK: %[[CONV0:.*]] = stablehlo.convert %[[RESHAPED0]] : (tensor<4x4xf32>) -> tensor<4x4xi32>
// CHECK: %[[SLICE1:.*]] = stablehlo.slice %{{.*}} [1:2, 0:4, 0:4]
// CHECK: %[[RESHAPED1:.*]] = stablehlo.reshape %[[SLICE1]]
// CHECK: %[[CONV1:.*]] = stablehlo.convert %[[RESHAPED1]] : (tensor<4x4xf32>) -> tensor<4x4xi32>
// CHECK: %[[SUM:.*]] = stablehlo.add %[[CONV0]], %[[CONV1]]
// CHECK: return %[[SUM]]
