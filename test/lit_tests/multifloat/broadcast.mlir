// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32" %s | FileCheck %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=CHECK-LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=CHECK-TUPLE %s

func.func @main(%arg0: tensor<4xf64>) -> tensor<4x5xf64> {
  // CHECK-LABEL: @main
  // CHECK: %[[C0:.*]] = stablehlo.convert %arg0 : (tensor<4xf64>) -> tensor<4xf32>
  // CHECK: %[[C1:.*]] = stablehlo.convert %[[C0]] : (tensor<4xf32>) -> tensor<4xf64>
  // CHECK: %[[SUB:.*]] = stablehlo.subtract %arg0, %[[C1]] : tensor<4xf64>
  // CHECK: %[[C2:.*]] = stablehlo.convert %[[SUB]] : (tensor<4xf64>) -> tensor<4xf32>
  // CHECK: %[[R1:.*]] = stablehlo.reshape %[[C0]] : (tensor<4xf32>) -> tensor<1x4xf32>
  // CHECK: %[[R2:.*]] = stablehlo.reshape %[[C2]] : (tensor<4xf32>) -> tensor<1x4xf32>
  // CHECK: %[[CAT:.*]] = stablehlo.concatenate %[[R1]], %[[R2]], dim = 0 : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<2x4xf32>
  // CHECK: %[[BC:.*]] = stablehlo.broadcast_in_dim %[[CAT]], dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4x5xf32>
  // CHECK: %[[C3:.*]] = stablehlo.convert %[[BC]] : (tensor<2x4x5xf32>) -> tensor<2x4x5xf64>
  // CHECK: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  // CHECK: %[[RES:.*]] = stablehlo.reduce(%[[C3]] init: %[[ZERO]]) applies stablehlo.add across dimensions = [0] : (tensor<2x4x5xf64>, tensor<f64>) -> tensor<4x5xf64>
  // CHECK: return %[[RES]] : tensor<4x5xf64>

  // CHECK-LAST-LABEL: @main
  // CHECK-LAST: %[[C0:.*]] = stablehlo.convert %arg0 : (tensor<4xf64>) -> tensor<4xf32>
  // CHECK-LAST: %[[C1:.*]] = stablehlo.convert %[[C0]] : (tensor<4xf32>) -> tensor<4xf64>
  // CHECK-LAST: %[[SUB:.*]] = stablehlo.subtract %arg0, %[[C1]] : tensor<4xf64>
  // CHECK-LAST: %[[C2:.*]] = stablehlo.convert %[[SUB]] : (tensor<4xf64>) -> tensor<4xf32>
  // CHECK-LAST: %[[R1:.*]] = stablehlo.reshape %[[C0]] : (tensor<4xf32>) -> tensor<4x1xf32>
  // CHECK-LAST: %[[R2:.*]] = stablehlo.reshape %[[C2]] : (tensor<4xf32>) -> tensor<4x1xf32>
  // CHECK-LAST: %[[CAT:.*]] = stablehlo.concatenate %[[R1]], %[[R2]], dim = 1 : (tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<4x2xf32>
  // CHECK-LAST: %[[BC:.*]] = stablehlo.broadcast_in_dim %[[CAT]], dims = [0, 2] : (tensor<4x2xf32>) -> tensor<4x5x2xf32>
  // CHECK-LAST: %[[C3:.*]] = stablehlo.convert %[[BC]] : (tensor<4x5x2xf32>) -> tensor<4x5x2xf64>
  // CHECK-LAST: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  // CHECK-LAST: %[[RES:.*]] = stablehlo.reduce(%[[C3]] init: %[[ZERO]]) applies stablehlo.add across dimensions = [2] : (tensor<4x5x2xf64>, tensor<f64>) -> tensor<4x5xf64>
  // CHECK-LAST: return %[[RES]] : tensor<4x5xf64>

  // CHECK-TUPLE-LABEL: @main
  // CHECK-TUPLE: %[[C0:.*]] = stablehlo.convert %arg0 : (tensor<4xf64>) -> tensor<4xf32>
  // CHECK-TUPLE: %[[C1:.*]] = stablehlo.convert %[[C0]] : (tensor<4xf32>) -> tensor<4xf64>
  // CHECK-TUPLE: %[[SUB:.*]] = stablehlo.subtract %arg0, %[[C1]] : tensor<4xf64>
  // CHECK-TUPLE: %[[C2:.*]] = stablehlo.convert %[[SUB]] : (tensor<4xf64>) -> tensor<4xf32>
  // CHECK-TUPLE: %[[TUP:.*]] = stablehlo.tuple %[[C0]], %[[C2]] : tuple<tensor<4xf32>, tensor<4xf32>>
  // CHECK-TUPLE: %[[HI:.*]] = stablehlo.get_tuple_element %[[TUP]][0] : (tuple<tensor<4xf32>, tensor<4xf32>>) -> tensor<4xf32>
  // CHECK-TUPLE: %[[LO:.*]] = stablehlo.get_tuple_element %[[TUP]][1] : (tuple<tensor<4xf32>, tensor<4xf32>>) -> tensor<4xf32>
  // CHECK-TUPLE: %[[BC1:.*]] = stablehlo.broadcast_in_dim %[[HI]], dims = [0] : (tensor<4xf32>) -> tensor<4x5xf32>
  // CHECK-TUPLE: %[[BC2:.*]] = stablehlo.broadcast_in_dim %[[LO]], dims = [0] : (tensor<4xf32>) -> tensor<4x5xf32>
  // CHECK-TUPLE: %[[RES_TUP:.*]] = stablehlo.tuple %[[BC1]], %[[BC2]] : tuple<tensor<4x5xf32>, tensor<4x5xf32>>
  // CHECK-TUPLE: %[[C3:.*]] = stablehlo.convert %[[BC1]] : (tensor<4x5xf32>) -> tensor<4x5xf64>
  // CHECK-TUPLE: %[[C4:.*]] = stablehlo.convert %[[BC2]] : (tensor<4x5xf32>) -> tensor<4x5xf64>
  // CHECK-TUPLE: %[[ADD:.*]] = stablehlo.add %[[C3]], %[[C4]] : tensor<4x5xf64>
  // CHECK-TUPLE: return %[[ADD]] : tensor<4x5xf64>
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<4xf64>) -> tensor<4x5xf64>
  return %0 : tensor<4x5xf64>
}
