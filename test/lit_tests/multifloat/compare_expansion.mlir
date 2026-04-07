// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first expansion-size=2" %s | FileCheck --check-prefix=FIRST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last expansion-size=2" %s | FileCheck --check-prefix=LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple expansion-size=2" %s | FileCheck --check-prefix=TUPLE %s
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @compare_f64_eq(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>) -> tensor<2xi1> {
  %0 = stablehlo.compare EQ, %arg0, %arg1 : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xi1>
  return %0 : tensor<2xi1>
}

func.func @compare_f64_ne(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>) -> tensor<2xi1> {
  %0 = stablehlo.compare NE, %arg0, %arg1 : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xi1>
  return %0 : tensor<2xi1>
}

func.func @compare_f64_ge(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>) -> tensor<2xi1> {
  %0 = stablehlo.compare GE, %arg0, %arg1 : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xi1>
  return %0 : tensor<2xi1>
}

// TUPLE-LABEL: func.func @compare_f64_eq(
// TUPLE:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// TUPLE:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<2xf32>) -> tensor<2xf64>
// TUPLE:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<2xf64>
// TUPLE:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<2xf64>) -> tensor<2xf32>
// TUPLE:     %[[V_4:.*]] = stablehlo.tuple %[[V_0]], %[[V_3]] : tuple<tensor<2xf32>, tensor<2xf32>>
// TUPLE:     %[[V_5:.*]] = stablehlo.convert %arg1 : (tensor<2xf64>) -> tensor<2xf32>
// TUPLE:     %[[V_6:.*]] = stablehlo.convert %[[V_5]] : (tensor<2xf32>) -> tensor<2xf64>
// TUPLE:     %[[V_7:.*]] = stablehlo.subtract %arg1, %[[V_6]] : tensor<2xf64>
// TUPLE:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<2xf64>) -> tensor<2xf32>
// TUPLE:     %[[V_9:.*]] = stablehlo.tuple %[[V_5]], %[[V_8]] : tuple<tensor<2xf32>, tensor<2xf32>>
// TUPLE:     %[[V_10:.*]] = stablehlo.get_tuple_element %[[V_4]][0] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE:     %[[V_11:.*]] = stablehlo.get_tuple_element %[[V_4]][1] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE:     %[[V_12:.*]] = stablehlo.get_tuple_element %[[V_9]][0] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE:     %[[V_13:.*]] = stablehlo.get_tuple_element %[[V_9]][1] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE:     %[[V_14:.*]] = stablehlo.compare EQ, %[[V_10]], %[[V_12]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// TUPLE:     %[[V_15:.*]] = stablehlo.compare EQ, %[[V_11]], %[[V_13]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// TUPLE:     %[[V_16:.*]] = stablehlo.reshape %[[V_14]] : (tensor<2xi1>) -> tensor<2xi1>
// TUPLE:     %[[V_17:.*]] = stablehlo.reshape %[[V_15]] : (tensor<2xi1>) -> tensor<2xi1>
// TUPLE:     %[[V_18:.*]] = stablehlo.and %[[V_16]], %[[V_17]] : tensor<2xi1>
// TUPLE:     return %[[V_18]] : tensor<2xi1>
// TUPLE-LABEL: func.func @compare_f64_ne
// TUPLE:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// TUPLE:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<2xf32>) -> tensor<2xf64>
// TUPLE:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<2xf64>
// TUPLE:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<2xf64>) -> tensor<2xf32>
// TUPLE:     %[[V_4:.*]] = stablehlo.tuple %[[V_0]], %[[V_3]] : tuple<tensor<2xf32>, tensor<2xf32>>
// TUPLE:     %[[V_5:.*]] = stablehlo.convert %arg1 : (tensor<2xf64>) -> tensor<2xf32>
// TUPLE:     %[[V_6:.*]] = stablehlo.convert %[[V_5]] : (tensor<2xf32>) -> tensor<2xf64>
// TUPLE:     %[[V_7:.*]] = stablehlo.subtract %arg1, %[[V_6]] : tensor<2xf64>
// TUPLE:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<2xf64>) -> tensor<2xf32>
// TUPLE:     %[[V_9:.*]] = stablehlo.tuple %[[V_5]], %[[V_8]] : tuple<tensor<2xf32>, tensor<2xf32>>
// TUPLE:     %[[V_10:.*]] = stablehlo.get_tuple_element %[[V_4]][0] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE:     %[[V_11:.*]] = stablehlo.get_tuple_element %[[V_4]][1] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE:     %[[V_12:.*]] = stablehlo.get_tuple_element %[[V_9]][0] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE:     %[[V_13:.*]] = stablehlo.get_tuple_element %[[V_9]][1] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE:     %[[V_14:.*]] = stablehlo.compare NE, %[[V_10]], %[[V_12]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// TUPLE:     %[[V_15:.*]] = stablehlo.compare NE, %[[V_11]], %[[V_13]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// TUPLE:     %[[V_16:.*]] = stablehlo.reshape %[[V_14]] : (tensor<2xi1>) -> tensor<2xi1>
// TUPLE:     %[[V_17:.*]] = stablehlo.reshape %[[V_15]] : (tensor<2xi1>) -> tensor<2xi1>
// TUPLE:     %[[V_18:.*]] = stablehlo.or %[[V_16]], %[[V_17]] : tensor<2xi1>
// TUPLE:     return %[[V_18]] : tensor<2xi1>
// TUPLE-LABEL: func.func @compare_f64_ge
// TUPLE:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// TUPLE:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<2xf32>) -> tensor<2xf64>
// TUPLE:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<2xf64>
// TUPLE:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<2xf64>) -> tensor<2xf32>
// TUPLE:     %[[V_4:.*]] = stablehlo.tuple %[[V_0]], %[[V_3]] : tuple<tensor<2xf32>, tensor<2xf32>>
// TUPLE:     %[[V_5:.*]] = stablehlo.convert %arg1 : (tensor<2xf64>) -> tensor<2xf32>
// TUPLE:     %[[V_6:.*]] = stablehlo.convert %[[V_5]] : (tensor<2xf32>) -> tensor<2xf64>
// TUPLE:     %[[V_7:.*]] = stablehlo.subtract %arg1, %[[V_6]] : tensor<2xf64>
// TUPLE:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<2xf64>) -> tensor<2xf32>
// TUPLE:     %[[V_9:.*]] = stablehlo.tuple %[[V_5]], %[[V_8]] : tuple<tensor<2xf32>, tensor<2xf32>>
// TUPLE:     %[[V_10:.*]] = stablehlo.get_tuple_element %[[V_4]][0] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE:     %[[V_11:.*]] = stablehlo.get_tuple_element %[[V_4]][1] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE:     %[[V_12:.*]] = stablehlo.get_tuple_element %[[V_9]][0] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE:     %[[V_13:.*]] = stablehlo.get_tuple_element %[[V_9]][1] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE:     %[[V_14:.*]] = stablehlo.compare GT, %[[V_10]], %[[V_12]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// TUPLE:     %[[V_15:.*]] = stablehlo.compare EQ, %[[V_10]], %[[V_12]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// TUPLE:     %[[V_16:.*]] = stablehlo.compare GE, %[[V_11]], %[[V_13]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// TUPLE:     %[[V_17:.*]] = stablehlo.select %[[V_15]], %[[V_16]], %[[V_14]] : tensor<2xi1>, tensor<2xi1>
// TUPLE:     return %[[V_17]] : tensor<2xi1>

// FIRST-LABEL: func.func @compare_f64_eq(
// FIRST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// FIRST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<2xf32>) -> tensor<2xf64>
// FIRST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<2xf64>
// FIRST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<2xf64>) -> tensor<2xf32>
// FIRST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST:     %[[V_7:.*]] = stablehlo.convert %arg1 : (tensor<2xf64>) -> tensor<2xf32>
// FIRST:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<2xf32>) -> tensor<2xf64>
// FIRST:     %[[V_9:.*]] = stablehlo.subtract %arg1, %[[V_8]] : tensor<2xf64>
// FIRST:     %[[V_10:.*]] = stablehlo.convert %[[V_9]] : (tensor<2xf64>) -> tensor<2xf32>
// FIRST:     %[[V_11:.*]] = stablehlo.reshape %[[V_7]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_12:.*]] = stablehlo.reshape %[[V_10]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_13:.*]] = stablehlo.concatenate %[[V_11]], %[[V_12]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST:     %[[V_14:.*]] = stablehlo.compare EQ, %[[V_6]], %[[V_13]] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
// FIRST:     %[[C:.*]] = stablehlo.constant dense<true> : tensor<i1>
// FIRST:     %[[V_15:.*]] = stablehlo.reduce(%[[V_14]] init: %[[C]]) applies stablehlo.and across dimensions = [0] : (tensor<2x2xi1>, tensor<i1>) -> tensor<2xi1>
// FIRST:     return %[[V_15]] : tensor<2xi1>
// FIRST-LABEL: func.func @compare_f64_ne
// FIRST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// FIRST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<2xf32>) -> tensor<2xf64>
// FIRST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<2xf64>
// FIRST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<2xf64>) -> tensor<2xf32>
// FIRST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST:     %[[V_7:.*]] = stablehlo.convert %arg1 : (tensor<2xf64>) -> tensor<2xf32>
// FIRST:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<2xf32>) -> tensor<2xf64>
// FIRST:     %[[V_9:.*]] = stablehlo.subtract %arg1, %[[V_8]] : tensor<2xf64>
// FIRST:     %[[V_10:.*]] = stablehlo.convert %[[V_9]] : (tensor<2xf64>) -> tensor<2xf32>
// FIRST:     %[[V_11:.*]] = stablehlo.reshape %[[V_7]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_12:.*]] = stablehlo.reshape %[[V_10]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_13:.*]] = stablehlo.concatenate %[[V_11]], %[[V_12]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST:     %[[V_14:.*]] = stablehlo.compare NE, %[[V_6]], %[[V_13]] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
// FIRST:     %[[C:.*]] = stablehlo.constant dense<false> : tensor<i1>
// FIRST:     %[[V_15:.*]] = stablehlo.reduce(%[[V_14]] init: %[[C]]) applies stablehlo.or across dimensions = [0] : (tensor<2x2xi1>, tensor<i1>) -> tensor<2xi1>
// FIRST:     return %[[V_15]] : tensor<2xi1>
// FIRST-LABEL: func.func @compare_f64_ge
// FIRST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// FIRST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<2xf32>) -> tensor<2xf64>
// FIRST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<2xf64>
// FIRST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<2xf64>) -> tensor<2xf32>
// FIRST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST:     %[[V_7:.*]] = stablehlo.convert %arg1 : (tensor<2xf64>) -> tensor<2xf32>
// FIRST:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<2xf32>) -> tensor<2xf64>
// FIRST:     %[[V_9:.*]] = stablehlo.subtract %arg1, %[[V_8]] : tensor<2xf64>
// FIRST:     %[[V_10:.*]] = stablehlo.convert %[[V_9]] : (tensor<2xf64>) -> tensor<2xf32>
// FIRST:     %[[V_11:.*]] = stablehlo.reshape %[[V_7]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_12:.*]] = stablehlo.reshape %[[V_10]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_13:.*]] = stablehlo.concatenate %[[V_11]], %[[V_12]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST:     %[[V_14:.*]] = stablehlo.slice %[[V_6]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_15:.*]] = stablehlo.slice %[[V_6]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_16:.*]] = stablehlo.slice %[[V_13]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_17:.*]] = stablehlo.slice %[[V_13]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_18:.*]] = stablehlo.compare GT, %[[V_14]], %[[V_16]] : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x2xi1>
// FIRST:     %[[V_19:.*]] = stablehlo.compare EQ, %[[V_14]], %[[V_16]] : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x2xi1>
// FIRST:     %[[V_20:.*]] = stablehlo.compare GE, %[[V_15]], %[[V_17]] : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x2xi1>
// FIRST:     %[[V_21:.*]] = stablehlo.select %[[V_19]], %[[V_20]], %[[V_18]] : tensor<1x2xi1>, tensor<1x2xi1>
// FIRST:     %[[V_22:.*]] = stablehlo.reshape %[[V_21]] : (tensor<1x2xi1>) -> tensor<2xi1>
// FIRST:     return %[[V_22]] : tensor<2xi1>

// LAST-LABEL: func.func @compare_f64_eq(
// LAST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// LAST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<2xf32>) -> tensor<2xf64>
// LAST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<2xf64>
// LAST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<2xf64>) -> tensor<2xf32>
// LAST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// LAST:     %[[V_7:.*]] = stablehlo.convert %arg1 : (tensor<2xf64>) -> tensor<2xf32>
// LAST:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<2xf32>) -> tensor<2xf64>
// LAST:     %[[V_9:.*]] = stablehlo.subtract %arg1, %[[V_8]] : tensor<2xf64>
// LAST:     %[[V_10:.*]] = stablehlo.convert %[[V_9]] : (tensor<2xf64>) -> tensor<2xf32>
// LAST:     %[[V_11:.*]] = stablehlo.reshape %[[V_7]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_12:.*]] = stablehlo.reshape %[[V_10]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_13:.*]] = stablehlo.concatenate %[[V_11]], %[[V_12]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// LAST:     %[[V_14:.*]] = stablehlo.compare EQ, %[[V_6]], %[[V_13]] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
// LAST:     %[[C:.*]] = stablehlo.constant dense<true> : tensor<i1>
// LAST:     %[[V_15:.*]] = stablehlo.reduce(%[[V_14]] init: %[[C]]) applies stablehlo.and across dimensions = [1] : (tensor<2x2xi1>, tensor<i1>) -> tensor<2xi1>
// LAST:     return %[[V_15]] : tensor<2xi1>
// LAST-LABEL: func.func @compare_f64_ne
// LAST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// LAST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<2xf32>) -> tensor<2xf64>
// LAST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<2xf64>
// LAST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<2xf64>) -> tensor<2xf32>
// LAST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// LAST:     %[[V_7:.*]] = stablehlo.convert %arg1 : (tensor<2xf64>) -> tensor<2xf32>
// LAST:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<2xf32>) -> tensor<2xf64>
// LAST:     %[[V_9:.*]] = stablehlo.subtract %arg1, %[[V_8]] : tensor<2xf64>
// LAST:     %[[V_10:.*]] = stablehlo.convert %[[V_9]] : (tensor<2xf64>) -> tensor<2xf32>
// LAST:     %[[V_11:.*]] = stablehlo.reshape %[[V_7]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_12:.*]] = stablehlo.reshape %[[V_10]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_13:.*]] = stablehlo.concatenate %[[V_11]], %[[V_12]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// LAST:     %[[V_14:.*]] = stablehlo.compare NE, %[[V_6]], %[[V_13]] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
// LAST:     %[[C:.*]] = stablehlo.constant dense<false> : tensor<i1>
// LAST:     %[[V_15:.*]] = stablehlo.reduce(%[[V_14]] init: %[[C]]) applies stablehlo.or across dimensions = [1] : (tensor<2x2xi1>, tensor<i1>) -> tensor<2xi1>
// LAST:     return %[[V_15]] : tensor<2xi1>
// LAST-LABEL: func.func @compare_f64_ge
// LAST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// LAST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<2xf32>) -> tensor<2xf64>
// LAST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<2xf64>
// LAST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<2xf64>) -> tensor<2xf32>
// LAST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// LAST:     %[[V_7:.*]] = stablehlo.convert %arg1 : (tensor<2xf64>) -> tensor<2xf32>
// LAST:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<2xf32>) -> tensor<2xf64>
// LAST:     %[[V_9:.*]] = stablehlo.subtract %arg1, %[[V_8]] : tensor<2xf64>
// LAST:     %[[V_10:.*]] = stablehlo.convert %[[V_9]] : (tensor<2xf64>) -> tensor<2xf32>
// LAST:     %[[V_11:.*]] = stablehlo.reshape %[[V_7]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_12:.*]] = stablehlo.reshape %[[V_10]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_13:.*]] = stablehlo.concatenate %[[V_11]], %[[V_12]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// LAST:     %[[V_14:.*]] = stablehlo.slice %[[V_6]] [0:2, 0:1] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_15:.*]] = stablehlo.slice %[[V_6]] [0:2, 1:2] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_16:.*]] = stablehlo.slice %[[V_13]] [0:2, 0:1] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_17:.*]] = stablehlo.slice %[[V_13]] [0:2, 1:2] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_18:.*]] = stablehlo.compare GT, %[[V_14]], %[[V_16]] : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x1xi1>
// LAST:     %[[V_19:.*]] = stablehlo.compare EQ, %[[V_14]], %[[V_16]] : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x1xi1>
// LAST:     %[[V_20:.*]] = stablehlo.compare GE, %[[V_15]], %[[V_17]] : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x1xi1>
// LAST:     %[[V_21:.*]] = stablehlo.select %[[V_19]], %[[V_20]], %[[V_18]] : tensor<2x1xi1>, tensor<2x1xi1>
// LAST:     %[[V_22:.*]] = stablehlo.reshape %[[V_21]] : (tensor<2x1xi1>) -> tensor<2xi1>
// LAST:     return %[[V_22]] : tensor<2xi1>

// FIRST-LABEL: func.func @main

// LAST-LABEL: func.func @main

// TUPLE-LABEL: func.func @main

func.func @main() attributes {enzyme.no_multifloat} {
  %cst1 = stablehlo.constant dense<[1.1, 1.1]> : tensor<2xf64>
  %cst2 = stablehlo.constant dense<[1.1, 1.100000000000001]> : tensor<2xf64>
  
  %eq_expected = stablehlo.constant dense<[true, false]> : tensor<2xi1>
  %ne_expected = stablehlo.constant dense<[false, true]> : tensor<2xi1>
  %ge_expected = stablehlo.constant dense<[true, false]> : tensor<2xi1>
  
  %res_eq = func.call @compare_f64_eq(%cst1, %cst2) : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xi1>
  %res_ne = func.call @compare_f64_ne(%cst1, %cst2) : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xi1>
  %res_ge = func.call @compare_f64_ge(%cst1, %cst2) : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xi1>
  
  %res_eq_f64 = stablehlo.convert %res_eq : (tensor<2xi1>) -> tensor<2xf64>
  %eq_expected_f64 = stablehlo.convert %eq_expected : (tensor<2xi1>) -> tensor<2xf64>
  "check.expect_close"(%res_eq_f64, %eq_expected_f64) {max_ulp_difference = 0 : ui64} : (tensor<2xf64>, tensor<2xf64>) -> ()
  "check.expect_close"(%res_eq_f64, %eq_expected_f64) {max_ulp_difference = 0 : ui64} : (tensor<2xf64>, tensor<2xf64>) -> ()
  
  %res_ne_f64 = stablehlo.convert %res_ne : (tensor<2xi1>) -> tensor<2xf64>
  %ne_expected_f64 = stablehlo.convert %ne_expected : (tensor<2xi1>) -> tensor<2xf64>
  "check.expect_close"(%res_ne_f64, %ne_expected_f64) {max_ulp_difference = 0 : ui64} : (tensor<2xf64>, tensor<2xf64>) -> ()
  "check.expect_close"(%res_ne_f64, %ne_expected_f64) {max_ulp_difference = 0 : ui64} : (tensor<2xf64>, tensor<2xf64>) -> ()
  
  %res_ge_f64 = stablehlo.convert %res_ge : (tensor<2xi1>) -> tensor<2xf64>
  %ge_expected_f64 = stablehlo.convert %ge_expected : (tensor<2xi1>) -> tensor<2xf64>
  "check.expect_close"(%res_ge_f64, %ge_expected_f64) {max_ulp_difference = 0 : ui64} : (tensor<2xf64>, tensor<2xf64>) -> ()
  "check.expect_close"(%res_ge_f64, %ge_expected_f64) {max_ulp_difference = 0 : ui64} : (tensor<2xf64>, tensor<2xf64>) -> ()
  
  return
}
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<1.100000e+00> : tensor<2xf64>
// FIRST:     %[[CST_0:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<2xf64>
// FIRST:     %[[C:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<2xi1>
// FIRST:     %[[C_1:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<2xi1>
// FIRST:     %[[C_2:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<2xi1>
// FIRST:     %[[V_0:.*]] = call @compare_f64_eq(%[[CST]], %[[CST_0]]) : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xi1>
// FIRST:     %[[V_1:.*]] = call @compare_f64_ne(%[[CST]], %[[CST_0]]) : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xi1>
// FIRST:     %[[V_2:.*]] = call @compare_f64_ge(%[[CST]], %[[CST_0]]) : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xi1>
// FIRST:     %[[V_3:.*]] = stablehlo.convert %[[V_0]] : (tensor<2xi1>) -> tensor<2xf64>
// FIRST:     %[[V_4:.*]] = stablehlo.convert %[[C]] : (tensor<2xi1>) -> tensor<2xf64>
// FIRST:     check.expect_close %[[V_3]], %[[V_4]], max_ulp_difference = 0 : tensor<2xf64>, tensor<2xf64>
// FIRST:     check.expect_close %[[V_3]], %[[V_4]], max_ulp_difference = 0 : tensor<2xf64>, tensor<2xf64>
// FIRST:     %[[V_5:.*]] = stablehlo.convert %[[V_1]] : (tensor<2xi1>) -> tensor<2xf64>
// FIRST:     %[[V_6:.*]] = stablehlo.convert %[[C_1]] : (tensor<2xi1>) -> tensor<2xf64>
// FIRST:     check.expect_close %[[V_5]], %[[V_6]], max_ulp_difference = 0 : tensor<2xf64>, tensor<2xf64>
// FIRST:     check.expect_close %[[V_5]], %[[V_6]], max_ulp_difference = 0 : tensor<2xf64>, tensor<2xf64>
// FIRST:     %[[V_7:.*]] = stablehlo.convert %[[V_2]] : (tensor<2xi1>) -> tensor<2xf64>
// FIRST:     %[[V_8:.*]] = stablehlo.convert %[[C_2]] : (tensor<2xi1>) -> tensor<2xf64>
// FIRST:     check.expect_close %[[V_7]], %[[V_8]], max_ulp_difference = 0 : tensor<2xf64>, tensor<2xf64>
// FIRST:     check.expect_close %[[V_7]], %[[V_8]], max_ulp_difference = 0 : tensor<2xf64>, tensor<2xf64>
// FIRST:     return
