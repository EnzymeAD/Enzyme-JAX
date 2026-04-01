// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32" %s | FileCheck %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=CHECK-LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=CHECK-TUPLE %s

func.func @main(%pred: tensor<2xi1>, %on_true: tensor<2xf64>, %on_false: tensor<2xf64>) -> tensor<2xf64> {
  // CHECK-LABEL: @main
  // CHECK: %[[TRUE_CONV:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
  // CHECK: %[[FALSE_CONV:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
  // CHECK: %[[PRED_BCAST:.*]] = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<2xi1>) -> tensor<2x2xi1>
  // CHECK: %[[SELECT:.*]] = stablehlo.select %[[PRED_BCAST]], %[[TRUE_CONV]], %[[FALSE_CONV]] : tensor<2x2xi1>, tensor<2x2xf32>
// CHECK: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK: %[[OUT:.*]] = stablehlo.reduce(%{{.*}} init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
// CHECK: return %[[OUT]] : tensor<2xf64>

  // CHECK-LAST-LABEL: @main
  // CHECK-LAST: %[[TRUE_CONV:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
  // CHECK-LAST: %[[FALSE_CONV:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
  // CHECK-LAST: %[[PRED_BCAST:.*]] = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<2xi1>) -> tensor<2x2xi1>
  // CHECK-LAST: %[[SELECT:.*]] = stablehlo.select %[[PRED_BCAST]], %[[TRUE_CONV]], %[[FALSE_CONV]] : tensor<2x2xi1>, tensor<2x2xf32>
// CHECK-LAST: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-LAST: %[[OUT:.*]] = stablehlo.reduce(%{{.*}} init: %[[CST]]) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
// CHECK-LAST: return %[[OUT]] : tensor<2xf64>

  // CHECK-TUPLE-LABEL: @main
  // CHECK-TUPLE: %[[TRUE_TUPLE:.*]] = stablehlo.tuple %{{.*}}, %{{.*}} : tuple<tensor<2xf32>, tensor<2xf32>>
  // CHECK-TUPLE: %[[FALSE_TUPLE:.*]] = stablehlo.tuple %{{.*}}, %{{.*}} : tuple<tensor<2xf32>, tensor<2xf32>>
  // CHECK-TUPLE: %[[TRUE_HIGH:.*]] = stablehlo.get_tuple_element %[[TRUE_TUPLE]][0] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
  // CHECK-TUPLE: %[[TRUE_LOW:.*]] = stablehlo.get_tuple_element %[[TRUE_TUPLE]][1] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
  // CHECK-TUPLE: %[[FALSE_HIGH:.*]] = stablehlo.get_tuple_element %[[FALSE_TUPLE]][0] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
  // CHECK-TUPLE: %[[FALSE_LOW:.*]] = stablehlo.get_tuple_element %[[FALSE_TUPLE]][1] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
  // CHECK-TUPLE: %[[SELECT_HIGH:.*]] = stablehlo.select %arg0, %[[TRUE_HIGH]], %[[FALSE_HIGH]] : tensor<2xi1>, tensor<2xf32>
  // CHECK-TUPLE: %[[SELECT_LOW:.*]] = stablehlo.select %arg0, %[[TRUE_LOW]], %[[FALSE_LOW]] : tensor<2xi1>, tensor<2xf32>
  // CHECK-TUPLE: %[[PACKED:.*]] = stablehlo.tuple %[[SELECT_HIGH]], %[[SELECT_LOW]] : tuple<tensor<2xf32>, tensor<2xf32>>
  // CHECK-TUPLE: %[[CONV1:.*]] = stablehlo.convert %{{.*}} : (tensor<2xf32>) -> tensor<2xf64>
  // CHECK-TUPLE: %[[CONV2:.*]] = stablehlo.convert %{{.*}} : (tensor<2xf32>) -> tensor<2xf64>
  // CHECK-TUPLE: %[[OUT:.*]] = stablehlo.add %[[CONV1]], %[[CONV2]] : tensor<2xf64>
  // CHECK-TUPLE: return %[[OUT]] : tensor<2xf64>

  %0 = stablehlo.select %pred, %on_true, %on_false : tensor<2xi1>, tensor<2xf64>
  return %0 : tensor<2xf64>
}
