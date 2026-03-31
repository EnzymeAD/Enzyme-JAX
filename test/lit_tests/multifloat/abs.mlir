// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck %s

func.func @abs(%arg0: tensor<2xf64>) -> tensor<2xf64> {
  %0 = stablehlo.abs %arg0 : tensor<2xf64>
  return %0 : tensor<2xf64>
}
// CHECK: func @abs(%arg0: tensor<2xf64>) -> tensor<2xf64> {
// CHECK: %{{.*}} = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// CHECK: %{{.*}} = stablehlo.convert %{{.*}} : (tensor<2xf32>) -> tensor<2xf64>
// CHECK: %{{.*}} = stablehlo.subtract %arg0, %{{.*}} : tensor<2xf64>
// CHECK: %{{.*}} = stablehlo.convert %{{.*}} : (tensor<2xf64>) -> tensor<2xf32>
// CHECK: %{{.*}} = stablehlo.reshape %{{.*}} : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK: %{{.*}} = stablehlo.reshape %{{.*}} : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK: %{{.*}} = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK: %{{.*}} = stablehlo.slice %{{.*}} [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// CHECK: %{{.*}} = stablehlo.slice %{{.*}} [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// CHECK: %{{.*}} = stablehlo.compare  GE, %{{.*}}, %{{.*}} : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x2xi1>
// CHECK: %{{.*}} = stablehlo.negate %{{.*}} : tensor<1x2xf32>
// CHECK: %{{.*}} = stablehlo.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<1x2xi1>, tensor<1x2xf32>
// CHECK: %{{.*}} = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK: %{{.*}} = stablehlo.convert %{{.*}} : (tensor<1x2xf32>) -> tensor<1x2xf64>
// CHECK: %{{.*}} = stablehlo.convert %{{.*}} : (tensor<1x2xf32>) -> tensor<1x2xf64>
// CHECK: %{{.*}} = stablehlo.add %{{.*}}, %{{.*}} : tensor<1x2xf64>
// CHECK: %{{.*}} = stablehlo.reshape %{{.*}} : (tensor<1x2xf64>) -> tensor<2xf64>
// CHECK: return %{{.*}} : tensor<2xf64>
