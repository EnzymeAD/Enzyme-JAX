// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32" %s | FileCheck %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=CHECK-LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=CHECK-TUPLE %s

func.func @main(%arg0: tensor<5xf64>) -> tensor<3xf64> {
  // CHECK-LABEL: @main
  // CHECK: %[[A:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 0 : (tensor<1x5xf32>, tensor<1x5xf32>) -> tensor<2x5xf32>
  // CHECK: %[[SLICE:.*]] = stablehlo.slice %[[A]] [0:2, 1:4] : (tensor<2x5xf32>) -> tensor<2x3xf32>
// CHECK: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK: %[[OUT:.*]] = stablehlo.reduce(%{{.*}} init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x3xf64>, tensor<f64>) -> tensor<3xf64>
// CHECK: return %[[OUT]] : tensor<3xf64>

  // CHECK-LAST-LABEL: @main
  // CHECK-LAST: %[[A:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 1 : (tensor<5x1xf32>, tensor<5x1xf32>) -> tensor<5x2xf32>
  // CHECK-LAST: %[[SLICE:.*]] = stablehlo.slice %[[A]] [1:4, 0:2] : (tensor<5x2xf32>) -> tensor<3x2xf32>
// CHECK-LAST: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-LAST: %[[OUT:.*]] = stablehlo.reduce(%{{.*}} init: %[[CST]]) applies stablehlo.add across dimensions = [1] : (tensor<3x2xf64>, tensor<f64>) -> tensor<3xf64>
// CHECK-LAST: return %[[OUT]] : tensor<3xf64>

  // CHECK-TUPLE-LABEL: @main
  // CHECK-TUPLE: %[[TUPLE:.*]] = stablehlo.tuple %{{.*}}, %{{.*}} : tuple<tensor<5xf32>, tensor<5xf32>>
  // CHECK-TUPLE: %[[HIGH:.*]] = stablehlo.get_tuple_element %[[TUPLE]][0] : (tuple<tensor<5xf32>, tensor<5xf32>>) -> tensor<5xf32>
  // CHECK-TUPLE: %[[LOW:.*]] = stablehlo.get_tuple_element %[[TUPLE]][1] : (tuple<tensor<5xf32>, tensor<5xf32>>) -> tensor<5xf32>
  // CHECK-TUPLE: %[[SLICE_HIGH:.*]] = stablehlo.slice %[[HIGH]] [1:4] : (tensor<5xf32>) -> tensor<3xf32>
  // CHECK-TUPLE: %[[SLICE_LOW:.*]] = stablehlo.slice %[[LOW]] [1:4] : (tensor<5xf32>) -> tensor<3xf32>
  // CHECK-TUPLE: %[[CONV1:.*]] = stablehlo.convert %[[SLICE_HIGH]] : (tensor<3xf32>) -> tensor<3xf64>
  // CHECK-TUPLE: %[[CONV2:.*]] = stablehlo.convert %[[SLICE_LOW]] : (tensor<3xf32>) -> tensor<3xf64>
  // CHECK-TUPLE: %[[OUT:.*]] = stablehlo.add %[[CONV1]], %[[CONV2]] : tensor<3xf64>
  // CHECK-TUPLE: return %[[OUT]] : tensor<3xf64>
  %0 = stablehlo.slice %arg0 [1:4] : (tensor<5xf64>) -> tensor<3xf64>
  return %0 : tensor<3xf64>
}
