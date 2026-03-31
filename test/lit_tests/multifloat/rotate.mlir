// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first expansion-size=2" %s | FileCheck %s

func.func @rotate(%arg0: tensor<4x4xf64>) -> tensor<4x4xf64> {
  %0 = "enzymexla.rotate"(%arg0) <{amount = 1 : i32, dimension = 0 : i32}> : (tensor<4x4xf64>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}
// CHECK-LABEL: func.func @rotate
// CHECK: %[[A:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 0 : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<2x4x4xf32>
// CHECK: %[[NEW_OP:.*]] = "enzymexla.rotate"(%[[A]]) <{amount = 1 : i32, dimension = 1 : i32}> : (tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
// CHECK: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK: %[[OUT:.*]] = stablehlo.reduce(%{{.*}} init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x4x4xf64>, tensor<f64>) -> tensor<4x4xf64>
// CHECK: return %[[OUT]]
