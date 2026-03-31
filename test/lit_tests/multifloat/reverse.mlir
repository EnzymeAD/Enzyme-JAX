// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck %s

func.func @reverse(%arg0: tensor<2xf64>) -> tensor<2xf64> {
  %0 = stablehlo.reverse %arg0, dims = [0] : tensor<2xf64>
  return %0 : tensor<2xf64>
}
// CHECK: func @reverse(%arg0: tensor<2xf64>) -> tensor<2xf64> {
// CHECK: %{{.*}} = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// CHECK: %{{.*}} = stablehlo.convert %{{.*}} : (tensor<2xf32>) -> tensor<2xf64>
// CHECK: %{{.*}} = stablehlo.subtract %arg0, %{{.*}} : tensor<2xf64>
// CHECK: %{{.*}} = stablehlo.convert %{{.*}} : (tensor<2xf64>) -> tensor<2xf32>
// CHECK: %{{.*}} = stablehlo.reshape %{{.*}} : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK: %{{.*}} = stablehlo.reshape %{{.*}} : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK: %{{.*}} = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK: %{{.*}} = stablehlo.reverse %{{.*}}, dims = [1] : tensor<2x2xf32>
// CHECK: %{{.*}} = stablehlo.convert %{{.*}} : (tensor<2x2xf32>) -> tensor<2x2xf64>
// CHECK: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK: %{{.*}} = stablehlo.reduce(%{{.*}} init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
// CHECK: return %{{.*}} : tensor<2xf64>
