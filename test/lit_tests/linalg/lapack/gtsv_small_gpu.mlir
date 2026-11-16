// RUN: enzymexlamlir-opt --lower-enzymexla-lapack="backend=cuda" %s | FileCheck %s

// gtsv2 requires m >= 3, smaller systems are solved with a loop

func.func @m2(%dl: tensor<2xf32>, %d: tensor<2xf32>, %du: tensor<2xf32>, %b: tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<i32>) {
  %0:2 = enzymexla.lapack.gtsv %dl, %d, %du, %b : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<i32>)
  return %0#0, %0#1 : tensor<2x3xf32>, tensor<i32>
}

// CHECK: func.func @m2(%[[ARG0:[^:]+]]: tensor<2xf32>, %[[ARG1:[^:]+]]: tensor<2xf32>, %[[ARG2:[^:]+]]: tensor<2xf32>, %[[ARG3:[^:]+]]: tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<i32>) {
// CHECK-NEXT: %[[CONSTANT_0:.*]] = stablehlo.constant dense<true> : tensor<i1>
// CHECK-NEXT: %[[CONSTANT_1:.*]] = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT: %[[CONSTANT_2:.*]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT: %[[CONSTANT_3:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %[[BROADCAST_IN_DIM_0:.*]] = stablehlo.broadcast_in_dim %[[ARG0]], dims = [0] : (tensor<2xf32>) -> tensor<2x3xf32>
// CHECK-NEXT: %[[BROADCAST_IN_DIM_1:.*]] = stablehlo.broadcast_in_dim %[[ARG1]], dims = [0] : (tensor<2xf32>) -> tensor<2x3xf32>
// CHECK-NEXT: %[[BROADCAST_IN_DIM_2:.*]] = stablehlo.broadcast_in_dim %[[ARG2]], dims = [0] : (tensor<2xf32>) -> tensor<2x3xf32>
// CHECK-NEXT: %[[DYNAMIC_SLICE_0:.*]] = stablehlo.dynamic_slice %[[BROADCAST_IN_DIM_1]], %[[CONSTANT_3]], %[[CONSTANT_3]], sizes = [1, 3] : (tensor<2x3xf32>, tensor<i64>, tensor<i64>) -> tensor<1x3xf32>
// CHECK-NEXT: %[[DYNAMIC_SLICE_1:.*]] = stablehlo.dynamic_slice %[[BROADCAST_IN_DIM_2]], %[[CONSTANT_3]], %[[CONSTANT_3]], sizes = [1, 3] : (tensor<2x3xf32>, tensor<i64>, tensor<i64>) -> tensor<1x3xf32>
// CHECK-NEXT: %[[DIVIDE_0:.*]] = stablehlo.divide %[[DYNAMIC_SLICE_1]], %[[DYNAMIC_SLICE_0]] : tensor<1x3xf32>
// CHECK-NEXT: %[[DYNAMIC_UPDATE_SLICE_0:.*]] = stablehlo.dynamic_update_slice %[[ARG3]], %[[DIVIDE_0]], %[[CONSTANT_3]], %[[CONSTANT_3]] : (tensor<2x3xf32>, tensor<1x3xf32>, tensor<i64>, tensor<i64>) -> tensor<2x3xf32>
// CHECK-NEXT: %[[DYNAMIC_SLICE_2:.*]] = stablehlo.dynamic_slice %[[ARG3]], %[[CONSTANT_3]], %[[CONSTANT_3]], sizes = [1, 3] : (tensor<2x3xf32>, tensor<i64>, tensor<i64>) -> tensor<1x3xf32>
// CHECK-NEXT: %[[DIVIDE_1:.*]] = stablehlo.divide %[[DYNAMIC_SLICE_2]], %[[DYNAMIC_SLICE_0]] : tensor<1x3xf32>
// CHECK-NEXT: %[[DYNAMIC_UPDATE_SLICE_1:.*]] = stablehlo.dynamic_update_slice %[[ARG3]], %[[DIVIDE_1]], %[[CONSTANT_3]], %[[CONSTANT_3]] : (tensor<2x3xf32>, tensor<1x3xf32>, tensor<i64>, tensor<i64>) -> tensor<2x3xf32>
// CHECK-NEXT: %[[WHILE_0:.*]]:3 = stablehlo.while(%[[VAL_0:.*]] = %[[CONSTANT_2]], %[[VAL_1:.*]] = %[[DYNAMIC_UPDATE_SLICE_0]], %[[VAL_2:.*]] = %[[DYNAMIC_UPDATE_SLICE_1]]) : tensor<i64>, tensor<2x3xf32>, tensor<2x3xf32>
// CHECK-NEXT: cond {
// CHECK-NEXT: %[[COMPARE_0:.*]] = stablehlo.compare LT, %[[VAL_0]], %[[CONSTANT_1]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT: stablehlo.return %[[COMPARE_0]] : tensor<i1>
// CHECK-NEXT: } do {
// CHECK-NEXT: %[[ADD_0:.*]] = stablehlo.add %[[VAL_0]], %[[CONSTANT_2]] : tensor<i64>
// CHECK-NEXT: %[[SUBTRACT_0:.*]] = stablehlo.subtract %[[VAL_0]], %[[CONSTANT_2]] : tensor<i64>
// CHECK-NEXT: %[[DYNAMIC_SLICE_3:.*]] = stablehlo.dynamic_slice %[[BROADCAST_IN_DIM_0]], %[[VAL_0]], %[[CONSTANT_3]], sizes = [1, 3] : (tensor<2x3xf32>, tensor<i64>, tensor<i64>) -> tensor<1x3xf32>
// CHECK-NEXT: %[[DYNAMIC_SLICE_4:.*]] = stablehlo.dynamic_slice %[[BROADCAST_IN_DIM_1]], %[[VAL_0]], %[[CONSTANT_3]], sizes = [1, 3] : (tensor<2x3xf32>, tensor<i64>, tensor<i64>) -> tensor<1x3xf32>
// CHECK-NEXT: %[[DYNAMIC_SLICE_5:.*]] = stablehlo.dynamic_slice %[[VAL_1]], %[[SUBTRACT_0]], %[[CONSTANT_3]], sizes = [1, 3] : (tensor<2x3xf32>, tensor<i64>, tensor<i64>) -> tensor<1x3xf32>
// CHECK-NEXT: %[[MULTIPLY_0:.*]] = stablehlo.multiply %[[DYNAMIC_SLICE_3]], %[[DYNAMIC_SLICE_5]] : tensor<1x3xf32>
// CHECK-NEXT: %[[SUBTRACT_1:.*]] = stablehlo.subtract %[[DYNAMIC_SLICE_4]], %[[MULTIPLY_0]] : tensor<1x3xf32>
// CHECK-NEXT: %[[DYNAMIC_SLICE_6:.*]] = stablehlo.dynamic_slice %[[BROADCAST_IN_DIM_2]], %[[VAL_0]], %[[CONSTANT_3]], sizes = [1, 3] : (tensor<2x3xf32>, tensor<i64>, tensor<i64>) -> tensor<1x3xf32>
// CHECK-NEXT: %[[DIVIDE_2:.*]] = stablehlo.divide %[[DYNAMIC_SLICE_6]], %[[SUBTRACT_1]] : tensor<1x3xf32>
// CHECK-NEXT: %[[DYNAMIC_SLICE_7:.*]] = stablehlo.dynamic_slice %[[ARG3]], %[[VAL_0]], %[[CONSTANT_3]], sizes = [1, 3] : (tensor<2x3xf32>, tensor<i64>, tensor<i64>) -> tensor<1x3xf32>
// CHECK-NEXT: %[[DYNAMIC_SLICE_8:.*]] = stablehlo.dynamic_slice %[[VAL_2]], %[[SUBTRACT_0]], %[[CONSTANT_3]], sizes = [1, 3] : (tensor<2x3xf32>, tensor<i64>, tensor<i64>) -> tensor<1x3xf32>
// CHECK-NEXT: %[[MULTIPLY_1:.*]] = stablehlo.multiply %[[DYNAMIC_SLICE_3]], %[[DYNAMIC_SLICE_8]] : tensor<1x3xf32>
// CHECK-NEXT: %[[SUBTRACT_2:.*]] = stablehlo.subtract %[[DYNAMIC_SLICE_7]], %[[MULTIPLY_1]] : tensor<1x3xf32>
// CHECK-NEXT: %[[DIVIDE_3:.*]] = stablehlo.divide %[[SUBTRACT_2]], %[[SUBTRACT_1]] : tensor<1x3xf32>
// CHECK-NEXT: %[[DYNAMIC_UPDATE_SLICE_2:.*]] = stablehlo.dynamic_update_slice %[[VAL_1]], %[[DIVIDE_2]], %[[VAL_0]], %[[CONSTANT_3]] : (tensor<2x3xf32>, tensor<1x3xf32>, tensor<i64>, tensor<i64>) -> tensor<2x3xf32>
// CHECK-NEXT: %[[DYNAMIC_UPDATE_SLICE_3:.*]] = stablehlo.dynamic_update_slice %[[VAL_2]], %[[DIVIDE_3]], %[[VAL_0]], %[[CONSTANT_3]] : (tensor<2x3xf32>, tensor<1x3xf32>, tensor<i64>, tensor<i64>) -> tensor<2x3xf32>
// CHECK-NEXT: stablehlo.return %[[ADD_0]], %[[DYNAMIC_UPDATE_SLICE_2]], %[[DYNAMIC_UPDATE_SLICE_3]] : tensor<i64>, tensor<2x3xf32>, tensor<2x3xf32>
// CHECK-NEXT: }
// CHECK-NEXT: %[[WHILE_1:.*]]:2 = stablehlo.while(%[[VAL_3:.*]] = %[[CONSTANT_2]], %[[VAL_4:.*]] = %[[WHILE_0]]#2) : tensor<i64>, tensor<2x3xf32>
// CHECK-NEXT: cond {
// CHECK-NEXT: %[[COMPARE_1:.*]] = stablehlo.compare LT, %[[VAL_3]], %[[CONSTANT_1]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT: stablehlo.return %[[COMPARE_1]] : tensor<i1>
// CHECK-NEXT: } do {
// CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[VAL_3]], %[[CONSTANT_2]] : tensor<i64>
// CHECK-NEXT: %[[SUBTRACT_3:.*]] = stablehlo.subtract %[[CONSTANT_2]], %[[VAL_3]] : tensor<i64>
// CHECK-NEXT: %[[ADD_2:.*]] = stablehlo.add %[[SUBTRACT_3]], %[[CONSTANT_2]] : tensor<i64>
// CHECK-NEXT: %[[DYNAMIC_SLICE_9:.*]] = stablehlo.dynamic_slice %[[WHILE_0]]#2, %[[SUBTRACT_3]], %[[CONSTANT_3]], sizes = [1, 3] : (tensor<2x3xf32>, tensor<i64>, tensor<i64>) -> tensor<1x3xf32>
// CHECK-NEXT: %[[DYNAMIC_SLICE_10:.*]] = stablehlo.dynamic_slice %[[WHILE_0]]#1, %[[SUBTRACT_3]], %[[CONSTANT_3]], sizes = [1, 3] : (tensor<2x3xf32>, tensor<i64>, tensor<i64>) -> tensor<1x3xf32>
// CHECK-NEXT: %[[DYNAMIC_SLICE_11:.*]] = stablehlo.dynamic_slice %[[VAL_4]], %[[ADD_2]], %[[CONSTANT_3]], sizes = [1, 3] : (tensor<2x3xf32>, tensor<i64>, tensor<i64>) -> tensor<1x3xf32>
// CHECK-NEXT: %[[MULTIPLY_2:.*]] = stablehlo.multiply %[[DYNAMIC_SLICE_10]], %[[DYNAMIC_SLICE_11]] : tensor<1x3xf32>
// CHECK-NEXT: %[[SUBTRACT_4:.*]] = stablehlo.subtract %[[DYNAMIC_SLICE_9]], %[[MULTIPLY_2]] : tensor<1x3xf32>
// CHECK-NEXT: %[[DYNAMIC_UPDATE_SLICE_4:.*]] = stablehlo.dynamic_update_slice %[[VAL_4]], %[[SUBTRACT_4]], %[[SUBTRACT_3]], %[[CONSTANT_3]] : (tensor<2x3xf32>, tensor<1x3xf32>, tensor<i64>, tensor<i64>) -> tensor<2x3xf32>
// CHECK-NEXT: stablehlo.return %[[ADD_1]], %[[DYNAMIC_UPDATE_SLICE_4]] : tensor<i64>, tensor<2x3xf32>
// CHECK-NEXT: }
// CHECK-NEXT: %[[REAL_0:.*]] = stablehlo.real %[[WHILE_1]]#1 : tensor<2x3xf32>
// CHECK-NEXT: %[[IS_FINITE_0:.*]] = stablehlo.is_finite %[[REAL_0]] : (tensor<2x3xf32>) -> tensor<2x3xi1>
// CHECK-NEXT: %[[IMAG_0:.*]] = stablehlo.imag %[[WHILE_1]]#1 : tensor<2x3xf32>
// CHECK-NEXT: %[[IS_FINITE_1:.*]] = stablehlo.is_finite %[[IMAG_0]] : (tensor<2x3xf32>) -> tensor<2x3xi1>
// CHECK-NEXT: %[[AND_0:.*]] = stablehlo.and %[[IS_FINITE_0]], %[[IS_FINITE_1]] : tensor<2x3xi1>
// CHECK-NEXT: %[[REDUCE_0:.*]] = stablehlo.reduce(%[[AND_0]] init: %[[CONSTANT_0]]) applies stablehlo.and across dimensions = [0, 1] : (tensor<2x3xi1>, tensor<i1>) -> tensor<i1>
// CHECK-NEXT: %[[NOT_0:.*]] = stablehlo.not %[[REDUCE_0]] : tensor<i1>
// CHECK-NEXT: %[[CONVERT_0:.*]] = stablehlo.convert %[[NOT_0]] : (tensor<i1>) -> tensor<i32>
// CHECK-NEXT: return %[[WHILE_1]]#1, %[[CONVERT_0]] : tensor<2x3xf32>, tensor<i32>
// CHECK-NEXT: }
