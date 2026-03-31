// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck %s

func.func @update_without_corners(%arg0: tensor<6x1522x3056xf64>, %arg1: tensor<6x1522x3056xf64>) -> tensor<6x1522x3056xf64> {
  %0 = "enzymexla.update_without_corners"(%arg0, %arg1) <{dimensionX = 0 : i64, x1 = 1 : i64, x2 = 5 : i64, dimensionY = 1 : i64, y1 = 1 : i64, y2 = 1521 : i64}> : (tensor<6x1522x3056xf64>, tensor<6x1522x3056xf64>) -> tensor<6x1522x3056xf64>
  return %0 : tensor<6x1522x3056xf64>
}

// CHECK-LABEL: func.func @update_without_corners
// CHECK: builtin.unrealized_conversion_cast %arg0 : tensor<6x1522x3056xf64> to tensor<2x6x1522x3056xf32>
// CHECK: builtin.unrealized_conversion_cast %arg1 : tensor<6x1522x3056xf64> to tensor<2x6x1522x3056xf32>
// CHECK: "enzymexla.update_without_corners"(%{{.*}}, %{{.*}}) <{dimensionX = 1 : i64, {{.*}}dimensionY = 2 : i64{{.*}}}> : (tensor<2x6x1522x3056xf32>, tensor<2x6x1522x3056xf32>) -> tensor<2x6x1522x3056xf32>
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<2x6x1522x3056xf32> to tensor<6x1522x3056xf64>
// CHECK: return
