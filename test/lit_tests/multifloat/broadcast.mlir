// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32" %s | FileCheck %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=CHECK-LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=CHECK-TUPLE %s

func.func @main(%arg0: tensor<4xf64>) -> tensor<4x5xf64> {
  // CHECK-LABEL: @main
  // CHECK: %[[CONCAT:.*]] = stablehlo.concatenate
  // CHECK: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[CONCAT]], dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4x5xf32>

  // CHECK-LAST-LABEL: @main
  // CHECK-LAST: %[[CONCAT:.*]] = stablehlo.concatenate
  // CHECK-LAST: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[CONCAT]], dims = [0, 2] : (tensor<4x2xf32>) -> tensor<4x5x2xf32>

  // CHECK-TUPLE-LABEL: @main
  // CHECK-TUPLE: %[[TUPLE:.*]] = stablehlo.tuple
  // CHECK-TUPLE: %[[HIGH:.*]] = stablehlo.get_tuple_element %[[TUPLE]][0]
  // CHECK-TUPLE: %[[LOW:.*]] = stablehlo.get_tuple_element %[[TUPLE]][1]
  // CHECK-TUPLE: %[[BCAST_HIGH:.*]] = stablehlo.broadcast_in_dim %[[HIGH]], dims = [0]
  // CHECK-TUPLE: %[[BCAST_LOW:.*]] = stablehlo.broadcast_in_dim %[[LOW]], dims = [0]
  // CHECK-TUPLE: %[[PACKED:.*]] = stablehlo.tuple %[[BCAST_HIGH]], %[[BCAST_LOW]]
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<4xf64>) -> tensor<4x5xf64>
  return %0 : tensor<4x5xf64>
}
