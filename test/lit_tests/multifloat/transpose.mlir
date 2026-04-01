// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32" %s | FileCheck %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=CHECK-LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=CHECK-TUPLE %s

func.func @main(%arg0: tensor<2x3xf64>) -> tensor<3x2xf64> {
  // CHECK-LABEL: @main
  // CHECK: %[[A:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 0 : (tensor<1x2x3xf32>, tensor<1x2x3xf32>) -> tensor<2x2x3xf32>
  // CHECK: %[[TRANS:.*]] = stablehlo.transpose %[[A]], dims = [0, 2, 1] : (tensor<2x2x3xf32>) -> tensor<2x3x2xf32>
// CHECK: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK: %[[OUT:.*]] = stablehlo.reduce(%{{.*}} init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x3x2xf64>, tensor<f64>) -> tensor<3x2xf64>
// CHECK: return %[[OUT]] : tensor<3x2xf64>

  // CHECK-LAST-LABEL: @main
  // CHECK-LAST: %[[A:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 2 : (tensor<2x3x1xf32>, tensor<2x3x1xf32>) -> tensor<2x3x2xf32>
  // CHECK-LAST: %[[TRANS:.*]] = stablehlo.transpose %[[A]], dims = [1, 0, 2] : (tensor<2x3x2xf32>) -> tensor<3x2x2xf32>
// CHECK-LAST: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-LAST: %[[OUT:.*]] = stablehlo.reduce(%{{.*}} init: %[[CST]]) applies stablehlo.add across dimensions = [2] : (tensor<3x2x2xf64>, tensor<f64>) -> tensor<3x2xf64>
// CHECK-LAST: return %[[OUT]] : tensor<3x2xf64>

  // CHECK-TUPLE-LABEL: @main
  // CHECK-TUPLE: %[[TUPLE:.*]] = stablehlo.tuple %{{.*}}, %{{.*}} : tuple<tensor<2x3xf32>, tensor<2x3xf32>>
  // CHECK-TUPLE: %[[HIGH:.*]] = stablehlo.get_tuple_element %[[TUPLE]][0] : (tuple<tensor<2x3xf32>, tensor<2x3xf32>>) -> tensor<2x3xf32>
  // CHECK-TUPLE: %[[LOW:.*]] = stablehlo.get_tuple_element %[[TUPLE]][1] : (tuple<tensor<2x3xf32>, tensor<2x3xf32>>) -> tensor<2x3xf32>
  // CHECK-TUPLE: %[[TRANS_HIGH:.*]] = stablehlo.transpose %[[HIGH]], dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
  // CHECK-TUPLE: %[[TRANS_LOW:.*]] = stablehlo.transpose %[[LOW]], dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
  // CHECK-TUPLE: %[[CONV1:.*]] = stablehlo.convert %[[TRANS_HIGH]] : (tensor<3x2xf32>) -> tensor<3x2xf64>
  // CHECK-TUPLE: %[[CONV2:.*]] = stablehlo.convert %[[TRANS_LOW]] : (tensor<3x2xf32>) -> tensor<3x2xf64>
  // CHECK-TUPLE: %[[OUT:.*]] = stablehlo.add %[[CONV1]], %[[CONV2]] : tensor<3x2xf64>
  // CHECK-TUPLE: return %[[OUT]] : tensor<3x2xf64>
  %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x3xf64>) -> tensor<3x2xf64>
  return %0 : tensor<3x2xf64>
}
