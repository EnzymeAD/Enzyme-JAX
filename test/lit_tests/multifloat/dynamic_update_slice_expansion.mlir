// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck --check-prefix=FIRST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=TUPLE %s
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @dus_f64(%operand: tensor<4x4xf64>, %update: tensor<2x2xf64>, %i: tensor<i32>, %j: tensor<i32>) -> tensor<4x4xf64> {
  %0 = stablehlo.dynamic_update_slice %operand, %update, %i, %j : (tensor<4x4xf64>, tensor<2x2xf64>, tensor<i32>, tensor<i32>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}

// FIRST-LABEL: func.func @dus_f64
// FIRST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<4x4xf64>) -> tensor<4x4xf32>
// FIRST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<4x4xf32>) -> tensor<4x4xf64>
// FIRST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<4x4xf64>
// FIRST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<4x4xf64>) -> tensor<4x4xf32>
// FIRST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
// FIRST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
// FIRST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 0 : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<2x4x4xf32>
// FIRST:     %[[V_7:.*]] = stablehlo.convert %arg1 : (tensor<2x2xf64>) -> tensor<2x2xf32>
// FIRST:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
// FIRST:     %[[V_9:.*]] = stablehlo.subtract %arg1, %[[V_8]] : tensor<2x2xf64>
// FIRST:     %[[V_10:.*]] = stablehlo.convert %[[V_9]] : (tensor<2x2xf64>) -> tensor<2x2xf32>
// FIRST:     %[[V_11:.*]] = stablehlo.reshape %[[V_7]] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
// FIRST:     %[[V_12:.*]] = stablehlo.reshape %[[V_10]] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
// FIRST:     %[[V_13:.*]] = stablehlo.concatenate %[[V_11]], %[[V_12]], dim = 0 : (tensor<1x2x2xf32>, tensor<1x2x2xf32>) -> tensor<2x2x2xf32>
// FIRST:     %[[C:.*]] = stablehlo.constant dense<0> : tensor<i32>
// FIRST:     %[[V_14:.*]] = stablehlo.dynamic_update_slice %[[V_6]], %[[V_13]], %[[C]], %arg2, %arg3 : (tensor<2x4x4xf32>, tensor<2x2x2xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x4x4xf32>
// FIRST:     %[[V_15:.*]] = stablehlo.convert %[[V_14]] : (tensor<2x4x4xf32>) -> tensor<2x4x4xf64>
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// FIRST:     %[[V_16:.*]] = stablehlo.reduce(%[[V_15]] init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x4x4xf64>, tensor<f64>) -> tensor<4x4xf64>
// FIRST:     return %[[V_16]] : tensor<4x4xf64>

// LAST-LABEL: func.func @dus_f64
// LAST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<4x4xf64>) -> tensor<4x4xf32>
// LAST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<4x4xf32>) -> tensor<4x4xf64>
// LAST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<4x4xf64>
// LAST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<4x4xf64>) -> tensor<4x4xf32>
// LAST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<4x4xf32>) -> tensor<4x4x1xf32>
// LAST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<4x4xf32>) -> tensor<4x4x1xf32>
// LAST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 2 : (tensor<4x4x1xf32>, tensor<4x4x1xf32>) -> tensor<4x4x2xf32>
// LAST:     %[[V_7:.*]] = stablehlo.convert %arg1 : (tensor<2x2xf64>) -> tensor<2x2xf32>
// LAST:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
// LAST:     %[[V_9:.*]] = stablehlo.subtract %arg1, %[[V_8]] : tensor<2x2xf64>
// LAST:     %[[V_10:.*]] = stablehlo.convert %[[V_9]] : (tensor<2x2xf64>) -> tensor<2x2xf32>
// LAST:     %[[V_11:.*]] = stablehlo.reshape %[[V_7]] : (tensor<2x2xf32>) -> tensor<2x2x1xf32>
// LAST:     %[[V_12:.*]] = stablehlo.reshape %[[V_10]] : (tensor<2x2xf32>) -> tensor<2x2x1xf32>
// LAST:     %[[V_13:.*]] = stablehlo.concatenate %[[V_11]], %[[V_12]], dim = 2 : (tensor<2x2x1xf32>, tensor<2x2x1xf32>) -> tensor<2x2x2xf32>
// LAST:     %[[C:.*]] = stablehlo.constant dense<0> : tensor<i32>
// LAST:     %[[V_14:.*]] = stablehlo.dynamic_update_slice %[[V_6]], %[[V_13]], %arg2, %arg3, %[[C]] : (tensor<4x4x2xf32>, tensor<2x2x2xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x4x2xf32>
// LAST:     %[[V_15:.*]] = stablehlo.convert %[[V_14]] : (tensor<4x4x2xf32>) -> tensor<4x4x2xf64>
// LAST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// LAST:     %[[V_16:.*]] = stablehlo.reduce(%[[V_15]] init: %[[CST]]) applies stablehlo.add across dimensions = [2] : (tensor<4x4x2xf64>, tensor<f64>) -> tensor<4x4xf64>
// LAST:     return %[[V_16]] : tensor<4x4xf64>

// TUPLE-LABEL: func.func @dus_f64
// TUPLE:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<4x4xf64>) -> tensor<4x4xf32>
// TUPLE:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<4x4xf32>) -> tensor<4x4xf64>
// TUPLE:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<4x4xf64>
// TUPLE:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<4x4xf64>) -> tensor<4x4xf32>
// TUPLE:     %[[V_4:.*]] = stablehlo.tuple %[[V_0]], %[[V_3]] : tuple<tensor<4x4xf32>, tensor<4x4xf32>>
// TUPLE:     %[[V_5:.*]] = stablehlo.convert %arg1 : (tensor<2x2xf64>) -> tensor<2x2xf32>
// TUPLE:     %[[V_6:.*]] = stablehlo.convert %[[V_5]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
// TUPLE:     %[[V_7:.*]] = stablehlo.subtract %arg1, %[[V_6]] : tensor<2x2xf64>
// TUPLE:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<2x2xf64>) -> tensor<2x2xf32>
// TUPLE:     %[[V_9:.*]] = stablehlo.tuple %[[V_5]], %[[V_8]] : tuple<tensor<2x2xf32>, tensor<2x2xf32>>
// TUPLE:     %[[V_10:.*]] = stablehlo.get_tuple_element %[[V_4]][0] : (tuple<tensor<4x4xf32>, tensor<4x4xf32>>) -> tensor<4x4xf32>
// TUPLE:     %[[V_11:.*]] = stablehlo.get_tuple_element %[[V_4]][1] : (tuple<tensor<4x4xf32>, tensor<4x4xf32>>) -> tensor<4x4xf32>
// TUPLE:     %[[V_12:.*]] = stablehlo.get_tuple_element %[[V_9]][0] : (tuple<tensor<2x2xf32>, tensor<2x2xf32>>) -> tensor<2x2xf32>
// TUPLE:     %[[V_13:.*]] = stablehlo.get_tuple_element %[[V_9]][1] : (tuple<tensor<2x2xf32>, tensor<2x2xf32>>) -> tensor<2x2xf32>
// TUPLE:     %[[V_14:.*]] = stablehlo.dynamic_update_slice %[[V_10]], %[[V_12]], %arg2, %arg3 : (tensor<4x4xf32>, tensor<2x2xf32>, tensor<i32>, tensor<i32>) -> tensor<4x4xf32>
// TUPLE:     %[[V_15:.*]] = stablehlo.dynamic_update_slice %[[V_11]], %[[V_13]], %arg2, %arg3 : (tensor<4x4xf32>, tensor<2x2xf32>, tensor<i32>, tensor<i32>) -> tensor<4x4xf32>
// TUPLE:     %[[V_16:.*]] = stablehlo.tuple %[[V_14]], %[[V_15]] : tuple<tensor<4x4xf32>, tensor<4x4xf32>>
// TUPLE:     %[[V_17:.*]] = stablehlo.convert %[[V_14]] : (tensor<4x4xf32>) -> tensor<4x4xf64>
// TUPLE:     %[[V_18:.*]] = stablehlo.convert %[[V_15]] : (tensor<4x4xf32>) -> tensor<4x4xf64>
// TUPLE:     %[[V_19:.*]] = stablehlo.add %[[V_17]], %[[V_18]] : tensor<4x4xf64>
// TUPLE:     return %[[V_19]] : tensor<4x4xf64>

func.func @main() attributes {enzyme.no_multifloat} {
  %operand = stablehlo.constant dense<1.1> : tensor<4x4xf64>
  %update = stablehlo.constant dense<2.2> : tensor<2x2xf64>
  %i = stablehlo.constant dense<1> : tensor<i32>
  %j = stablehlo.constant dense<1> : tensor<i32>
  
  %expected = stablehlo.constant dense<[[1.1, 1.1, 1.1, 1.1],
                                       [1.1, 2.2, 2.2, 1.1],
                                       [1.1, 2.2, 2.2, 1.1],
                                       [1.1, 1.1, 1.1, 1.1]]> : tensor<4x4xf64>
                                       
  %res = func.call @dus_f64(%operand, %update, %i, %j) : (tensor<4x4xf64>, tensor<2x2xf64>, tensor<i32>, tensor<i32>) -> tensor<4x4xf64>
  
  "check.expect_almost_eq"(%res, %expected) : (tensor<4x4xf64>, tensor<4x4xf64>) -> ()
  return
}

// FIRST-LABEL: func.func @main
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<4x4xf64>
// FIRST:     %[[CST_0:.*]] = stablehlo.constant dense<2.200000e+00> : tensor<2x2xf64>
// FIRST:     %[[C:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<i32>
// FIRST:     %[[C_1:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<i32>
// FIRST:     %[[CST_2:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<4x4xf64>
// FIRST:     %[[V_0:.*]] = call @dus_f64(%[[CST]], %[[CST_0]], %[[C]], %[[C_1]]) : (tensor<4x4xf64>, tensor<2x2xf64>, tensor<i32>, tensor<i32>) -> tensor<4x4xf64>
// FIRST:     check.expect_almost_eq %[[V_0]], %[[CST_2]] : tensor<4x4xf64>
// FIRST:     return
