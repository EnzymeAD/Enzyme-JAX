// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32" %s | FileCheck %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=CHECK-LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=CHECK-TUPLE %s

func.func @main(%arg0: tensor<4xf64>) -> tensor<2x2xf64> {
  // CHECK-LABEL: @main
  // CHECK: %[[A:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 0 : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<2x4xf32>
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %[[A]] : (tensor<2x4xf32>) -> tensor<2x2x2xf32>
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  // CHECK: %[[OUT:.*]] = stablehlo.reduce(%{{.*}} init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x2x2xf64>, tensor<f64>) -> tensor<2x2xf64>
  // CHECK: return %[[OUT]] : tensor<2x2xf64>

  // CHECK-LAST-LABEL: @main
  // CHECK-LAST: %[[A:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 1 : (tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<4x2xf32>
  // CHECK-LAST: %[[RESHAPE:.*]] = stablehlo.reshape %[[A]] : (tensor<4x2xf32>) -> tensor<2x2x2xf32>
  // CHECK-LAST: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  // CHECK-LAST: %[[OUT:.*]] = stablehlo.reduce(%{{.*}} init: %[[CST]]) applies stablehlo.add across dimensions = [2] : (tensor<2x2x2xf64>, tensor<f64>) -> tensor<2x2xf64>
  // CHECK-LAST: return %[[OUT]] : tensor<2x2xf64>

  // CHECK-TUPLE-LABEL: @main
  // CHECK-TUPLE: %[[TUPLE:.*]] = stablehlo.tuple %{{.*}}, %{{.*}} : tuple<tensor<4xf32>, tensor<4xf32>>
  // CHECK-TUPLE: %[[HIGH:.*]] = stablehlo.get_tuple_element %[[TUPLE]][0] : (tuple<tensor<4xf32>, tensor<4xf32>>) -> tensor<4xf32>
  // CHECK-TUPLE: %[[LOW:.*]] = stablehlo.get_tuple_element %[[TUPLE]][1] : (tuple<tensor<4xf32>, tensor<4xf32>>) -> tensor<4xf32>
  // CHECK-TUPLE: %[[RESHAPE_HIGH:.*]] = stablehlo.reshape %[[HIGH]] : (tensor<4xf32>) -> tensor<2x2xf32>
  // CHECK-TUPLE: %[[RESHAPE_LOW:.*]] = stablehlo.reshape %[[LOW]] : (tensor<4xf32>) -> tensor<2x2xf32>
  // CHECK-TUPLE: %[[CONV1:.*]] = stablehlo.convert %[[RESHAPE_HIGH]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
  // CHECK-TUPLE: %[[CONV2:.*]] = stablehlo.convert %[[RESHAPE_LOW]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
  // CHECK-TUPLE: %[[OUT:.*]] = stablehlo.add %[[CONV1]], %[[CONV2]] : tensor<2x2xf64>
  // CHECK-TUPLE: return %[[OUT]] : tensor<2x2xf64>
  %0 = stablehlo.reshape %arg0 : (tensor<4xf64>) -> tensor<2x2xf64>
  return %0 : tensor<2x2xf64>
}
