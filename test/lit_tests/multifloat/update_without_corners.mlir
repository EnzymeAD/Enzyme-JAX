// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck %s

func.func @update_without_corners(%arg0: tensor<6x1522x3056xf64>, %arg1: tensor<6x1522x3056xf64>) -> tensor<6x1522x3056xf64> {
  %0 = "enzymexla.update_without_corners"(%arg0, %arg1) <{dimensionX = 0 : i64, x1 = 1 : i64, x2 = 5 : i64, dimensionY = 1 : i64, y1 = 1 : i64, y2 = 1521 : i64}> : (tensor<6x1522x3056xf64>, tensor<6x1522x3056xf64>) -> tensor<6x1522x3056xf64>
  return %0 : tensor<6x1522x3056xf64>
}

// CHECK-LABEL: func.func @update_without_corners
// CHECK: %{{.*}} = stablehlo.convert %arg0 : (tensor<6x1522x3056xf64>) -> tensor<6x1522x3056xf32>
// CHECK: %{{.*}} = stablehlo.convert %{{.*}} : (tensor<6x1522x3056xf32>) -> tensor<6x1522x3056xf64>
// CHECK: %{{.*}} = stablehlo.subtract %arg0, %{{.*}} : tensor<6x1522x3056xf64>
// CHECK: %{{.*}} = stablehlo.convert %{{.*}} : (tensor<6x1522x3056xf64>) -> tensor<6x1522x3056xf32>
// CHECK: %{{.*}} = stablehlo.reshape %{{.*}} : (tensor<6x1522x3056xf32>) -> tensor<1x6x1522x3056xf32>
// CHECK: %{{.*}} = stablehlo.reshape %{{.*}} : (tensor<6x1522x3056xf32>) -> tensor<1x6x1522x3056xf32>
// CHECK: %{{.*}} = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 0 : (tensor<1x6x1522x3056xf32>, tensor<1x6x1522x3056xf32>) -> tensor<2x6x1522x3056xf32>
// CHECK: %{{.*}} = stablehlo.convert %arg1 : (tensor<6x1522x3056xf64>) -> tensor<6x1522x3056xf32>
// CHECK: %{{.*}} = stablehlo.convert %{{.*}} : (tensor<6x1522x3056xf32>) -> tensor<6x1522x3056xf64>
// CHECK: %{{.*}} = stablehlo.subtract %arg1, %{{.*}} : tensor<6x1522x3056xf64>
// CHECK: %{{.*}} = stablehlo.convert %{{.*}} : (tensor<6x1522x3056xf64>) -> tensor<6x1522x3056xf32>
// CHECK: %{{.*}} = stablehlo.reshape %{{.*}} : (tensor<6x1522x3056xf32>) -> tensor<1x6x1522x3056xf32>
// CHECK: %{{.*}} = stablehlo.reshape %{{.*}} : (tensor<6x1522x3056xf32>) -> tensor<1x6x1522x3056xf32>
// CHECK: %{{.*}} = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 0 : (tensor<1x6x1522x3056xf32>, tensor<1x6x1522x3056xf32>) -> tensor<2x6x1522x3056xf32>
// CHECK: "enzymexla.update_without_corners"(%{{.*}}, %{{.*}}) <{dimensionX = 1 : i64, {{.*}}dimensionY = 2 : i64{{.*}}}> : (tensor<2x6x1522x3056xf32>, tensor<2x6x1522x3056xf32>) -> tensor<2x6x1522x3056xf32>
// CHECK: %{{.*}} = stablehlo.slice %{{.*}} [0:1, 0:6, 0:1522, 0:3056] : (tensor<2x6x1522x3056xf32>) -> tensor<1x6x1522x3056xf32>
// CHECK: %{{.*}} = stablehlo.slice %{{.*}} [1:2, 0:6, 0:1522, 0:3056] : (tensor<2x6x1522x3056xf32>) -> tensor<1x6x1522x3056xf32>
// CHECK: %{{.*}} = stablehlo.convert %{{.*}} : (tensor<1x6x1522x3056xf32>) -> tensor<1x6x1522x3056xf64>
// CHECK: %{{.*}} = stablehlo.convert %{{.*}} : (tensor<1x6x1522x3056xf32>) -> tensor<1x6x1522x3056xf64>
// CHECK: %{{.*}} = stablehlo.add %{{.*}}, %{{.*}} : tensor<1x6x1522x3056xf64>
// CHECK: %{{.*}} = stablehlo.reshape %{{.*}} : (tensor<1x6x1522x3056xf64>) -> tensor<6x1522x3056xf64>
// CHECK: return %{{.*}} : tensor<6x1522x3056xf64>
