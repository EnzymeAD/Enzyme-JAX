// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck --check-prefix=FIRST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=TUPLE %s
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @concat_2op(%arg0: tensor<3x4xf64>, %arg1: tensor<1x4xf64>) -> tensor<4x4xf64> {
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<3x4xf64>, tensor<1x4xf64>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}

func.func @concat_3op(%arg0: tensor<1x4xf64>, %arg1: tensor<2x4xf64>, %arg2: tensor<1x4xf64>) -> tensor<4x4xf64> {
  %0 = stablehlo.concatenate %arg0, %arg1, %arg2, dim = 0 : (tensor<1x4xf64>, tensor<2x4xf64>, tensor<1x4xf64>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}

func.func @concat_5op(%a: tensor<4x1x8xf64>, %b: tensor<4x1x8xf64>, %c: tensor<4x3x8xf64>, %d: tensor<4x1x8xf64>, %e: tensor<4x1x8xf64>) -> tensor<4x7x8xf64> {
  %0 = stablehlo.concatenate %a, %b, %c, %d, %e, dim = 1 : (tensor<4x1x8xf64>, tensor<4x1x8xf64>, tensor<4x3x8xf64>, tensor<4x1x8xf64>, tensor<4x1x8xf64>) -> tensor<4x7x8xf64>
  return %0 : tensor<4x7x8xf64>
}

// TUPLE-LABEL: func.func @concat_2op(
// TUPLE:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<3x4xf64>) -> tensor<3x4xf32>
// TUPLE:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<3x4xf32>) -> tensor<3x4xf64>
// TUPLE:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<3x4xf64>
// TUPLE:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<3x4xf64>) -> tensor<3x4xf32>
// TUPLE:     %[[V_4:.*]] = stablehlo.tuple %[[V_0]], %[[V_3]] : tuple<tensor<3x4xf32>, tensor<3x4xf32>>
// TUPLE:     %[[V_5:.*]] = stablehlo.convert %arg1 : (tensor<1x4xf64>) -> tensor<1x4xf32>
// TUPLE:     %[[V_6:.*]] = stablehlo.convert %[[V_5]] : (tensor<1x4xf32>) -> tensor<1x4xf64>
// TUPLE:     %[[V_7:.*]] = stablehlo.subtract %arg1, %[[V_6]] : tensor<1x4xf64>
// TUPLE:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<1x4xf64>) -> tensor<1x4xf32>
// TUPLE:     %[[V_9:.*]] = stablehlo.tuple %[[V_5]], %[[V_8]] : tuple<tensor<1x4xf32>, tensor<1x4xf32>>
// TUPLE:     %[[V_10:.*]] = stablehlo.get_tuple_element %[[V_4]][0] : (tuple<tensor<3x4xf32>, tensor<3x4xf32>>) -> tensor<3x4xf32>
// TUPLE:     %[[V_11:.*]] = stablehlo.get_tuple_element %[[V_4]][1] : (tuple<tensor<3x4xf32>, tensor<3x4xf32>>) -> tensor<3x4xf32>
// TUPLE:     %[[V_12:.*]] = stablehlo.get_tuple_element %[[V_9]][0] : (tuple<tensor<1x4xf32>, tensor<1x4xf32>>) -> tensor<1x4xf32>
// TUPLE:     %[[V_13:.*]] = stablehlo.get_tuple_element %[[V_9]][1] : (tuple<tensor<1x4xf32>, tensor<1x4xf32>>) -> tensor<1x4xf32>
// TUPLE:     %[[V_14:.*]] = stablehlo.concatenate %[[V_10]], %[[V_12]], dim = 0 : (tensor<3x4xf32>, tensor<1x4xf32>) -> tensor<4x4xf32>
// TUPLE:     %[[V_15:.*]] = stablehlo.concatenate %[[V_11]], %[[V_13]], dim = 0 : (tensor<3x4xf32>, tensor<1x4xf32>) -> tensor<4x4xf32>
// TUPLE:     %[[V_16:.*]] = stablehlo.tuple %[[V_14]], %[[V_15]] : tuple<tensor<4x4xf32>, tensor<4x4xf32>>
// TUPLE:     %[[V_17:.*]] = stablehlo.convert %[[V_14]] : (tensor<4x4xf32>) -> tensor<4x4xf64>
// TUPLE:     %[[V_18:.*]] = stablehlo.convert %[[V_15]] : (tensor<4x4xf32>) -> tensor<4x4xf64>
// TUPLE:     %[[V_19:.*]] = stablehlo.add %[[V_17]], %[[V_18]] : tensor<4x4xf64>
// TUPLE:     return %[[V_19]] : tensor<4x4xf64>
// TUPLE-LABEL: func.func @concat_3op
// TUPLE:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<1x4xf64>) -> tensor<1x4xf32>
// TUPLE:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<1x4xf32>) -> tensor<1x4xf64>
// TUPLE:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<1x4xf64>
// TUPLE:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<1x4xf64>) -> tensor<1x4xf32>
// TUPLE:     %[[V_4:.*]] = stablehlo.tuple %[[V_0]], %[[V_3]] : tuple<tensor<1x4xf32>, tensor<1x4xf32>>
// TUPLE:     %[[V_5:.*]] = stablehlo.convert %arg1 : (tensor<2x4xf64>) -> tensor<2x4xf32>
// TUPLE:     %[[V_6:.*]] = stablehlo.convert %[[V_5]] : (tensor<2x4xf32>) -> tensor<2x4xf64>
// TUPLE:     %[[V_7:.*]] = stablehlo.subtract %arg1, %[[V_6]] : tensor<2x4xf64>
// TUPLE:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<2x4xf64>) -> tensor<2x4xf32>
// TUPLE:     %[[V_9:.*]] = stablehlo.tuple %[[V_5]], %[[V_8]] : tuple<tensor<2x4xf32>, tensor<2x4xf32>>
// TUPLE:     %[[V_10:.*]] = stablehlo.convert %arg2 : (tensor<1x4xf64>) -> tensor<1x4xf32>
// TUPLE:     %[[V_11:.*]] = stablehlo.convert %[[V_10]] : (tensor<1x4xf32>) -> tensor<1x4xf64>
// TUPLE:     %[[V_12:.*]] = stablehlo.subtract %arg2, %[[V_11]] : tensor<1x4xf64>
// TUPLE:     %[[V_13:.*]] = stablehlo.convert %[[V_12]] : (tensor<1x4xf64>) -> tensor<1x4xf32>
// TUPLE:     %[[V_14:.*]] = stablehlo.tuple %[[V_10]], %[[V_13]] : tuple<tensor<1x4xf32>, tensor<1x4xf32>>
// TUPLE:     %[[V_15:.*]] = stablehlo.get_tuple_element %[[V_4]][0] : (tuple<tensor<1x4xf32>, tensor<1x4xf32>>) -> tensor<1x4xf32>
// TUPLE:     %[[V_16:.*]] = stablehlo.get_tuple_element %[[V_4]][1] : (tuple<tensor<1x4xf32>, tensor<1x4xf32>>) -> tensor<1x4xf32>
// TUPLE:     %[[V_17:.*]] = stablehlo.get_tuple_element %[[V_9]][0] : (tuple<tensor<2x4xf32>, tensor<2x4xf32>>) -> tensor<2x4xf32>
// TUPLE:     %[[V_18:.*]] = stablehlo.get_tuple_element %[[V_9]][1] : (tuple<tensor<2x4xf32>, tensor<2x4xf32>>) -> tensor<2x4xf32>
// TUPLE:     %[[V_19:.*]] = stablehlo.get_tuple_element %[[V_14]][0] : (tuple<tensor<1x4xf32>, tensor<1x4xf32>>) -> tensor<1x4xf32>
// TUPLE:     %[[V_20:.*]] = stablehlo.get_tuple_element %[[V_14]][1] : (tuple<tensor<1x4xf32>, tensor<1x4xf32>>) -> tensor<1x4xf32>
// TUPLE:     %[[V_21:.*]] = stablehlo.concatenate %[[V_15]], %[[V_17]], %[[V_19]], dim = 0 : (tensor<1x4xf32>, tensor<2x4xf32>, tensor<1x4xf32>) -> tensor<4x4xf32>
// TUPLE:     %[[V_22:.*]] = stablehlo.concatenate %[[V_16]], %[[V_18]], %[[V_20]], dim = 0 : (tensor<1x4xf32>, tensor<2x4xf32>, tensor<1x4xf32>) -> tensor<4x4xf32>
// TUPLE:     %[[V_23:.*]] = stablehlo.tuple %[[V_21]], %[[V_22]] : tuple<tensor<4x4xf32>, tensor<4x4xf32>>
// TUPLE:     %[[V_24:.*]] = stablehlo.convert %[[V_21]] : (tensor<4x4xf32>) -> tensor<4x4xf64>
// TUPLE:     %[[V_25:.*]] = stablehlo.convert %[[V_22]] : (tensor<4x4xf32>) -> tensor<4x4xf64>
// TUPLE:     %[[V_26:.*]] = stablehlo.add %[[V_24]], %[[V_25]] : tensor<4x4xf64>
// TUPLE:     return %[[V_26]] : tensor<4x4xf64>
// TUPLE-LABEL: func.func @concat_5op
// TUPLE:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<4x1x8xf64>) -> tensor<4x1x8xf32>
// TUPLE:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<4x1x8xf32>) -> tensor<4x1x8xf64>
// TUPLE:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<4x1x8xf64>
// TUPLE:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<4x1x8xf64>) -> tensor<4x1x8xf32>
// TUPLE:     %[[V_4:.*]] = stablehlo.tuple %[[V_0]], %[[V_3]] : tuple<tensor<4x1x8xf32>, tensor<4x1x8xf32>>
// TUPLE:     %[[V_5:.*]] = stablehlo.convert %arg1 : (tensor<4x1x8xf64>) -> tensor<4x1x8xf32>
// TUPLE:     %[[V_6:.*]] = stablehlo.convert %[[V_5]] : (tensor<4x1x8xf32>) -> tensor<4x1x8xf64>
// TUPLE:     %[[V_7:.*]] = stablehlo.subtract %arg1, %[[V_6]] : tensor<4x1x8xf64>
// TUPLE:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<4x1x8xf64>) -> tensor<4x1x8xf32>
// TUPLE:     %[[V_9:.*]] = stablehlo.tuple %[[V_5]], %[[V_8]] : tuple<tensor<4x1x8xf32>, tensor<4x1x8xf32>>
// TUPLE:     %[[V_10:.*]] = stablehlo.convert %arg2 : (tensor<4x3x8xf64>) -> tensor<4x3x8xf32>
// TUPLE:     %[[V_11:.*]] = stablehlo.convert %[[V_10]] : (tensor<4x3x8xf32>) -> tensor<4x3x8xf64>
// TUPLE:     %[[V_12:.*]] = stablehlo.subtract %arg2, %[[V_11]] : tensor<4x3x8xf64>
// TUPLE:     %[[V_13:.*]] = stablehlo.convert %[[V_12]] : (tensor<4x3x8xf64>) -> tensor<4x3x8xf32>
// TUPLE:     %[[V_14:.*]] = stablehlo.tuple %[[V_10]], %[[V_13]] : tuple<tensor<4x3x8xf32>, tensor<4x3x8xf32>>
// TUPLE:     %[[V_15:.*]] = stablehlo.convert %arg3 : (tensor<4x1x8xf64>) -> tensor<4x1x8xf32>
// TUPLE:     %[[V_16:.*]] = stablehlo.convert %[[V_15]] : (tensor<4x1x8xf32>) -> tensor<4x1x8xf64>
// TUPLE:     %[[V_17:.*]] = stablehlo.subtract %arg3, %[[V_16]] : tensor<4x1x8xf64>
// TUPLE:     %[[V_18:.*]] = stablehlo.convert %[[V_17]] : (tensor<4x1x8xf64>) -> tensor<4x1x8xf32>
// TUPLE:     %[[V_19:.*]] = stablehlo.tuple %[[V_15]], %[[V_18]] : tuple<tensor<4x1x8xf32>, tensor<4x1x8xf32>>
// TUPLE:     %[[V_20:.*]] = stablehlo.convert %arg4 : (tensor<4x1x8xf64>) -> tensor<4x1x8xf32>
// TUPLE:     %[[V_21:.*]] = stablehlo.convert %[[V_20]] : (tensor<4x1x8xf32>) -> tensor<4x1x8xf64>
// TUPLE:     %[[V_22:.*]] = stablehlo.subtract %arg4, %[[V_21]] : tensor<4x1x8xf64>
// TUPLE:     %[[V_23:.*]] = stablehlo.convert %[[V_22]] : (tensor<4x1x8xf64>) -> tensor<4x1x8xf32>
// TUPLE:     %[[V_24:.*]] = stablehlo.tuple %[[V_20]], %[[V_23]] : tuple<tensor<4x1x8xf32>, tensor<4x1x8xf32>>
// TUPLE:     %[[V_25:.*]] = stablehlo.get_tuple_element %[[V_4]][0] : (tuple<tensor<4x1x8xf32>, tensor<4x1x8xf32>>) -> tensor<4x1x8xf32>
// TUPLE:     %[[V_26:.*]] = stablehlo.get_tuple_element %[[V_4]][1] : (tuple<tensor<4x1x8xf32>, tensor<4x1x8xf32>>) -> tensor<4x1x8xf32>
// TUPLE:     %[[V_27:.*]] = stablehlo.get_tuple_element %[[V_9]][0] : (tuple<tensor<4x1x8xf32>, tensor<4x1x8xf32>>) -> tensor<4x1x8xf32>
// TUPLE:     %[[V_28:.*]] = stablehlo.get_tuple_element %[[V_9]][1] : (tuple<tensor<4x1x8xf32>, tensor<4x1x8xf32>>) -> tensor<4x1x8xf32>
// TUPLE:     %[[V_29:.*]] = stablehlo.get_tuple_element %[[V_14]][0] : (tuple<tensor<4x3x8xf32>, tensor<4x3x8xf32>>) -> tensor<4x3x8xf32>
// TUPLE:     %[[V_30:.*]] = stablehlo.get_tuple_element %[[V_14]][1] : (tuple<tensor<4x3x8xf32>, tensor<4x3x8xf32>>) -> tensor<4x3x8xf32>
// TUPLE:     %[[V_31:.*]] = stablehlo.get_tuple_element %[[V_19]][0] : (tuple<tensor<4x1x8xf32>, tensor<4x1x8xf32>>) -> tensor<4x1x8xf32>
// TUPLE:     %[[V_32:.*]] = stablehlo.get_tuple_element %[[V_19]][1] : (tuple<tensor<4x1x8xf32>, tensor<4x1x8xf32>>) -> tensor<4x1x8xf32>
// TUPLE:     %[[V_33:.*]] = stablehlo.get_tuple_element %[[V_24]][0] : (tuple<tensor<4x1x8xf32>, tensor<4x1x8xf32>>) -> tensor<4x1x8xf32>
// TUPLE:     %[[V_34:.*]] = stablehlo.get_tuple_element %[[V_24]][1] : (tuple<tensor<4x1x8xf32>, tensor<4x1x8xf32>>) -> tensor<4x1x8xf32>
// TUPLE:     %[[V_35:.*]] = stablehlo.concatenate %[[V_25]], %[[V_27]], %[[V_29]], %[[V_31]], %[[V_33]], dim = 1 : (tensor<4x1x8xf32>, tensor<4x1x8xf32>, tensor<4x3x8xf32>, tensor<4x1x8xf32>, tensor<4x1x8xf32>) -> tensor<4x7x8xf32>
// TUPLE:     %[[V_36:.*]] = stablehlo.concatenate %[[V_26]], %[[V_28]], %[[V_30]], %[[V_32]], %[[V_34]], dim = 1 : (tensor<4x1x8xf32>, tensor<4x1x8xf32>, tensor<4x3x8xf32>, tensor<4x1x8xf32>, tensor<4x1x8xf32>) -> tensor<4x7x8xf32>
// TUPLE:     %[[V_37:.*]] = stablehlo.tuple %[[V_35]], %[[V_36]] : tuple<tensor<4x7x8xf32>, tensor<4x7x8xf32>>
// TUPLE:     %[[V_38:.*]] = stablehlo.convert %[[V_35]] : (tensor<4x7x8xf32>) -> tensor<4x7x8xf64>
// TUPLE:     %[[V_39:.*]] = stablehlo.convert %[[V_36]] : (tensor<4x7x8xf32>) -> tensor<4x7x8xf64>
// TUPLE:     %[[V_40:.*]] = stablehlo.add %[[V_38]], %[[V_39]] : tensor<4x7x8xf64>
// TUPLE:     return %[[V_40]] : tensor<4x7x8xf64>

// FIRST-LABEL: func.func @concat_2op(
// FIRST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<3x4xf64>) -> tensor<3x4xf32>
// FIRST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<3x4xf32>) -> tensor<3x4xf64>
// FIRST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<3x4xf64>
// FIRST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<3x4xf64>) -> tensor<3x4xf32>
// FIRST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<3x4xf32>) -> tensor<1x3x4xf32>
// FIRST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<3x4xf32>) -> tensor<1x3x4xf32>
// FIRST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 0 : (tensor<1x3x4xf32>, tensor<1x3x4xf32>) -> tensor<2x3x4xf32>
// FIRST:     %[[V_7:.*]] = stablehlo.convert %arg1 : (tensor<1x4xf64>) -> tensor<1x4xf32>
// FIRST:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<1x4xf32>) -> tensor<1x4xf64>
// FIRST:     %[[V_9:.*]] = stablehlo.subtract %arg1, %[[V_8]] : tensor<1x4xf64>
// FIRST:     %[[V_10:.*]] = stablehlo.convert %[[V_9]] : (tensor<1x4xf64>) -> tensor<1x4xf32>
// FIRST:     %[[V_11:.*]] = stablehlo.reshape %[[V_7]] : (tensor<1x4xf32>) -> tensor<1x1x4xf32>
// FIRST:     %[[V_12:.*]] = stablehlo.reshape %[[V_10]] : (tensor<1x4xf32>) -> tensor<1x1x4xf32>
// FIRST:     %[[V_13:.*]] = stablehlo.concatenate %[[V_11]], %[[V_12]], dim = 0 : (tensor<1x1x4xf32>, tensor<1x1x4xf32>) -> tensor<2x1x4xf32>
// FIRST:     %[[V_14:.*]] = stablehlo.concatenate %[[V_6]], %[[V_13]], dim = 1 : (tensor<2x3x4xf32>, tensor<2x1x4xf32>) -> tensor<2x4x4xf32>
// FIRST:     %[[V_15:.*]] = stablehlo.convert %[[V_14]] : (tensor<2x4x4xf32>) -> tensor<2x4x4xf64>
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// FIRST:     %[[V_16:.*]] = stablehlo.reduce(%[[V_15]] init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x4x4xf64>, tensor<f64>) -> tensor<4x4xf64>
// FIRST:     return %[[V_16]] : tensor<4x4xf64>
// FIRST-LABEL: func.func @concat_3op
// FIRST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<1x4xf64>) -> tensor<1x4xf32>
// FIRST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<1x4xf32>) -> tensor<1x4xf64>
// FIRST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<1x4xf64>
// FIRST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<1x4xf64>) -> tensor<1x4xf32>
// FIRST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<1x4xf32>) -> tensor<1x1x4xf32>
// FIRST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<1x4xf32>) -> tensor<1x1x4xf32>
// FIRST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 0 : (tensor<1x1x4xf32>, tensor<1x1x4xf32>) -> tensor<2x1x4xf32>
// FIRST:     %[[V_7:.*]] = stablehlo.convert %arg1 : (tensor<2x4xf64>) -> tensor<2x4xf32>
// FIRST:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<2x4xf32>) -> tensor<2x4xf64>
// FIRST:     %[[V_9:.*]] = stablehlo.subtract %arg1, %[[V_8]] : tensor<2x4xf64>
// FIRST:     %[[V_10:.*]] = stablehlo.convert %[[V_9]] : (tensor<2x4xf64>) -> tensor<2x4xf32>
// FIRST:     %[[V_11:.*]] = stablehlo.reshape %[[V_7]] : (tensor<2x4xf32>) -> tensor<1x2x4xf32>
// FIRST:     %[[V_12:.*]] = stablehlo.reshape %[[V_10]] : (tensor<2x4xf32>) -> tensor<1x2x4xf32>
// FIRST:     %[[V_13:.*]] = stablehlo.concatenate %[[V_11]], %[[V_12]], dim = 0 : (tensor<1x2x4xf32>, tensor<1x2x4xf32>) -> tensor<2x2x4xf32>
// FIRST:     %[[V_14:.*]] = stablehlo.convert %arg2 : (tensor<1x4xf64>) -> tensor<1x4xf32>
// FIRST:     %[[V_15:.*]] = stablehlo.convert %[[V_14]] : (tensor<1x4xf32>) -> tensor<1x4xf64>
// FIRST:     %[[V_16:.*]] = stablehlo.subtract %arg2, %[[V_15]] : tensor<1x4xf64>
// FIRST:     %[[V_17:.*]] = stablehlo.convert %[[V_16]] : (tensor<1x4xf64>) -> tensor<1x4xf32>
// FIRST:     %[[V_18:.*]] = stablehlo.reshape %[[V_14]] : (tensor<1x4xf32>) -> tensor<1x1x4xf32>
// FIRST:     %[[V_19:.*]] = stablehlo.reshape %[[V_17]] : (tensor<1x4xf32>) -> tensor<1x1x4xf32>
// FIRST:     %[[V_20:.*]] = stablehlo.concatenate %[[V_18]], %[[V_19]], dim = 0 : (tensor<1x1x4xf32>, tensor<1x1x4xf32>) -> tensor<2x1x4xf32>
// FIRST:     %[[V_21:.*]] = stablehlo.concatenate %[[V_6]], %[[V_13]], %[[V_20]], dim = 1 : (tensor<2x1x4xf32>, tensor<2x2x4xf32>, tensor<2x1x4xf32>) -> tensor<2x4x4xf32>
// FIRST:     %[[V_22:.*]] = stablehlo.convert %[[V_21]] : (tensor<2x4x4xf32>) -> tensor<2x4x4xf64>
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// FIRST:     %[[V_23:.*]] = stablehlo.reduce(%[[V_22]] init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x4x4xf64>, tensor<f64>) -> tensor<4x4xf64>
// FIRST:     return %[[V_23]] : tensor<4x4xf64>
// FIRST-LABEL: func.func @concat_5op
// FIRST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<4x1x8xf64>) -> tensor<4x1x8xf32>
// FIRST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<4x1x8xf32>) -> tensor<4x1x8xf64>
// FIRST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<4x1x8xf64>
// FIRST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<4x1x8xf64>) -> tensor<4x1x8xf32>
// FIRST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<4x1x8xf32>) -> tensor<1x4x1x8xf32>
// FIRST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<4x1x8xf32>) -> tensor<1x4x1x8xf32>
// FIRST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 0 : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<2x4x1x8xf32>
// FIRST:     %[[V_7:.*]] = stablehlo.convert %arg1 : (tensor<4x1x8xf64>) -> tensor<4x1x8xf32>
// FIRST:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<4x1x8xf32>) -> tensor<4x1x8xf64>
// FIRST:     %[[V_9:.*]] = stablehlo.subtract %arg1, %[[V_8]] : tensor<4x1x8xf64>
// FIRST:     %[[V_10:.*]] = stablehlo.convert %[[V_9]] : (tensor<4x1x8xf64>) -> tensor<4x1x8xf32>
// FIRST:     %[[V_11:.*]] = stablehlo.reshape %[[V_7]] : (tensor<4x1x8xf32>) -> tensor<1x4x1x8xf32>
// FIRST:     %[[V_12:.*]] = stablehlo.reshape %[[V_10]] : (tensor<4x1x8xf32>) -> tensor<1x4x1x8xf32>
// FIRST:     %[[V_13:.*]] = stablehlo.concatenate %[[V_11]], %[[V_12]], dim = 0 : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<2x4x1x8xf32>
// FIRST:     %[[V_14:.*]] = stablehlo.convert %arg2 : (tensor<4x3x8xf64>) -> tensor<4x3x8xf32>
// FIRST:     %[[V_15:.*]] = stablehlo.convert %[[V_14]] : (tensor<4x3x8xf32>) -> tensor<4x3x8xf64>
// FIRST:     %[[V_16:.*]] = stablehlo.subtract %arg2, %[[V_15]] : tensor<4x3x8xf64>
// FIRST:     %[[V_17:.*]] = stablehlo.convert %[[V_16]] : (tensor<4x3x8xf64>) -> tensor<4x3x8xf32>
// FIRST:     %[[V_18:.*]] = stablehlo.reshape %[[V_14]] : (tensor<4x3x8xf32>) -> tensor<1x4x3x8xf32>
// FIRST:     %[[V_19:.*]] = stablehlo.reshape %[[V_17]] : (tensor<4x3x8xf32>) -> tensor<1x4x3x8xf32>
// FIRST:     %[[V_20:.*]] = stablehlo.concatenate %[[V_18]], %[[V_19]], dim = 0 : (tensor<1x4x3x8xf32>, tensor<1x4x3x8xf32>) -> tensor<2x4x3x8xf32>
// FIRST:     %[[V_21:.*]] = stablehlo.convert %arg3 : (tensor<4x1x8xf64>) -> tensor<4x1x8xf32>
// FIRST:     %[[V_22:.*]] = stablehlo.convert %[[V_21]] : (tensor<4x1x8xf32>) -> tensor<4x1x8xf64>
// FIRST:     %[[V_23:.*]] = stablehlo.subtract %arg3, %[[V_22]] : tensor<4x1x8xf64>
// FIRST:     %[[V_24:.*]] = stablehlo.convert %[[V_23]] : (tensor<4x1x8xf64>) -> tensor<4x1x8xf32>
// FIRST:     %[[V_25:.*]] = stablehlo.reshape %[[V_21]] : (tensor<4x1x8xf32>) -> tensor<1x4x1x8xf32>
// FIRST:     %[[V_26:.*]] = stablehlo.reshape %[[V_24]] : (tensor<4x1x8xf32>) -> tensor<1x4x1x8xf32>
// FIRST:     %[[V_27:.*]] = stablehlo.concatenate %[[V_25]], %[[V_26]], dim = 0 : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<2x4x1x8xf32>
// FIRST:     %[[V_28:.*]] = stablehlo.convert %arg4 : (tensor<4x1x8xf64>) -> tensor<4x1x8xf32>
// FIRST:     %[[V_29:.*]] = stablehlo.convert %[[V_28]] : (tensor<4x1x8xf32>) -> tensor<4x1x8xf64>
// FIRST:     %[[V_30:.*]] = stablehlo.subtract %arg4, %[[V_29]] : tensor<4x1x8xf64>
// FIRST:     %[[V_31:.*]] = stablehlo.convert %[[V_30]] : (tensor<4x1x8xf64>) -> tensor<4x1x8xf32>
// FIRST:     %[[V_32:.*]] = stablehlo.reshape %[[V_28]] : (tensor<4x1x8xf32>) -> tensor<1x4x1x8xf32>
// FIRST:     %[[V_33:.*]] = stablehlo.reshape %[[V_31]] : (tensor<4x1x8xf32>) -> tensor<1x4x1x8xf32>
// FIRST:     %[[V_34:.*]] = stablehlo.concatenate %[[V_32]], %[[V_33]], dim = 0 : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<2x4x1x8xf32>
// FIRST:     %[[V_35:.*]] = stablehlo.concatenate %[[V_6]], %[[V_13]], %[[V_20]], %[[V_27]], %[[V_34]], dim = 2 : (tensor<2x4x1x8xf32>, tensor<2x4x1x8xf32>, tensor<2x4x3x8xf32>, tensor<2x4x1x8xf32>, tensor<2x4x1x8xf32>) -> tensor<2x4x7x8xf32>
// FIRST:     %[[V_36:.*]] = stablehlo.convert %[[V_35]] : (tensor<2x4x7x8xf32>) -> tensor<2x4x7x8xf64>
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// FIRST:     %[[V_37:.*]] = stablehlo.reduce(%[[V_36]] init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x4x7x8xf64>, tensor<f64>) -> tensor<4x7x8xf64>
// FIRST:     return %[[V_37]] : tensor<4x7x8xf64>

// LAST-LABEL: func.func @concat_2op(
// LAST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<3x4xf64>) -> tensor<3x4xf32>
// LAST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<3x4xf32>) -> tensor<3x4xf64>
// LAST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<3x4xf64>
// LAST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<3x4xf64>) -> tensor<3x4xf32>
// LAST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<3x4xf32>) -> tensor<3x4x1xf32>
// LAST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<3x4xf32>) -> tensor<3x4x1xf32>
// LAST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 2 : (tensor<3x4x1xf32>, tensor<3x4x1xf32>) -> tensor<3x4x2xf32>
// LAST:     %[[V_7:.*]] = stablehlo.convert %arg1 : (tensor<1x4xf64>) -> tensor<1x4xf32>
// LAST:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<1x4xf32>) -> tensor<1x4xf64>
// LAST:     %[[V_9:.*]] = stablehlo.subtract %arg1, %[[V_8]] : tensor<1x4xf64>
// LAST:     %[[V_10:.*]] = stablehlo.convert %[[V_9]] : (tensor<1x4xf64>) -> tensor<1x4xf32>
// LAST:     %[[V_11:.*]] = stablehlo.reshape %[[V_7]] : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
// LAST:     %[[V_12:.*]] = stablehlo.reshape %[[V_10]] : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
// LAST:     %[[V_13:.*]] = stablehlo.concatenate %[[V_11]], %[[V_12]], dim = 2 : (tensor<1x4x1xf32>, tensor<1x4x1xf32>) -> tensor<1x4x2xf32>
// LAST:     %[[V_14:.*]] = stablehlo.concatenate %[[V_6]], %[[V_13]], dim = 0 : (tensor<3x4x2xf32>, tensor<1x4x2xf32>) -> tensor<4x4x2xf32>
// LAST:     %[[V_15:.*]] = stablehlo.convert %[[V_14]] : (tensor<4x4x2xf32>) -> tensor<4x4x2xf64>
// LAST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// LAST:     %[[V_16:.*]] = stablehlo.reduce(%[[V_15]] init: %[[CST]]) applies stablehlo.add across dimensions = [2] : (tensor<4x4x2xf64>, tensor<f64>) -> tensor<4x4xf64>
// LAST:     return %[[V_16]] : tensor<4x4xf64>
// LAST-LABEL: func.func @concat_3op
// LAST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<1x4xf64>) -> tensor<1x4xf32>
// LAST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<1x4xf32>) -> tensor<1x4xf64>
// LAST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<1x4xf64>
// LAST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<1x4xf64>) -> tensor<1x4xf32>
// LAST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
// LAST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
// LAST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 2 : (tensor<1x4x1xf32>, tensor<1x4x1xf32>) -> tensor<1x4x2xf32>
// LAST:     %[[V_7:.*]] = stablehlo.convert %arg1 : (tensor<2x4xf64>) -> tensor<2x4xf32>
// LAST:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<2x4xf32>) -> tensor<2x4xf64>
// LAST:     %[[V_9:.*]] = stablehlo.subtract %arg1, %[[V_8]] : tensor<2x4xf64>
// LAST:     %[[V_10:.*]] = stablehlo.convert %[[V_9]] : (tensor<2x4xf64>) -> tensor<2x4xf32>
// LAST:     %[[V_11:.*]] = stablehlo.reshape %[[V_7]] : (tensor<2x4xf32>) -> tensor<2x4x1xf32>
// LAST:     %[[V_12:.*]] = stablehlo.reshape %[[V_10]] : (tensor<2x4xf32>) -> tensor<2x4x1xf32>
// LAST:     %[[V_13:.*]] = stablehlo.concatenate %[[V_11]], %[[V_12]], dim = 2 : (tensor<2x4x1xf32>, tensor<2x4x1xf32>) -> tensor<2x4x2xf32>
// LAST:     %[[V_14:.*]] = stablehlo.convert %arg2 : (tensor<1x4xf64>) -> tensor<1x4xf32>
// LAST:     %[[V_15:.*]] = stablehlo.convert %[[V_14]] : (tensor<1x4xf32>) -> tensor<1x4xf64>
// LAST:     %[[V_16:.*]] = stablehlo.subtract %arg2, %[[V_15]] : tensor<1x4xf64>
// LAST:     %[[V_17:.*]] = stablehlo.convert %[[V_16]] : (tensor<1x4xf64>) -> tensor<1x4xf32>
// LAST:     %[[V_18:.*]] = stablehlo.reshape %[[V_14]] : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
// LAST:     %[[V_19:.*]] = stablehlo.reshape %[[V_17]] : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
// LAST:     %[[V_20:.*]] = stablehlo.concatenate %[[V_18]], %[[V_19]], dim = 2 : (tensor<1x4x1xf32>, tensor<1x4x1xf32>) -> tensor<1x4x2xf32>
// LAST:     %[[V_21:.*]] = stablehlo.concatenate %[[V_6]], %[[V_13]], %[[V_20]], dim = 0 : (tensor<1x4x2xf32>, tensor<2x4x2xf32>, tensor<1x4x2xf32>) -> tensor<4x4x2xf32>
// LAST:     %[[V_22:.*]] = stablehlo.convert %[[V_21]] : (tensor<4x4x2xf32>) -> tensor<4x4x2xf64>
// LAST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// LAST:     %[[V_23:.*]] = stablehlo.reduce(%[[V_22]] init: %[[CST]]) applies stablehlo.add across dimensions = [2] : (tensor<4x4x2xf64>, tensor<f64>) -> tensor<4x4xf64>
// LAST:     return %[[V_23]] : tensor<4x4xf64>
// LAST-LABEL: func.func @concat_5op
// LAST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<4x1x8xf64>) -> tensor<4x1x8xf32>
// LAST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<4x1x8xf32>) -> tensor<4x1x8xf64>
// LAST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<4x1x8xf64>
// LAST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<4x1x8xf64>) -> tensor<4x1x8xf32>
// LAST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<4x1x8xf32>) -> tensor<4x1x8x1xf32>
// LAST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<4x1x8xf32>) -> tensor<4x1x8x1xf32>
// LAST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 3 : (tensor<4x1x8x1xf32>, tensor<4x1x8x1xf32>) -> tensor<4x1x8x2xf32>
// LAST:     %[[V_7:.*]] = stablehlo.convert %arg1 : (tensor<4x1x8xf64>) -> tensor<4x1x8xf32>
// LAST:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<4x1x8xf32>) -> tensor<4x1x8xf64>
// LAST:     %[[V_9:.*]] = stablehlo.subtract %arg1, %[[V_8]] : tensor<4x1x8xf64>
// LAST:     %[[V_10:.*]] = stablehlo.convert %[[V_9]] : (tensor<4x1x8xf64>) -> tensor<4x1x8xf32>
// LAST:     %[[V_11:.*]] = stablehlo.reshape %[[V_7]] : (tensor<4x1x8xf32>) -> tensor<4x1x8x1xf32>
// LAST:     %[[V_12:.*]] = stablehlo.reshape %[[V_10]] : (tensor<4x1x8xf32>) -> tensor<4x1x8x1xf32>
// LAST:     %[[V_13:.*]] = stablehlo.concatenate %[[V_11]], %[[V_12]], dim = 3 : (tensor<4x1x8x1xf32>, tensor<4x1x8x1xf32>) -> tensor<4x1x8x2xf32>
// LAST:     %[[V_14:.*]] = stablehlo.convert %arg2 : (tensor<4x3x8xf64>) -> tensor<4x3x8xf32>
// LAST:     %[[V_15:.*]] = stablehlo.convert %[[V_14]] : (tensor<4x3x8xf32>) -> tensor<4x3x8xf64>
// LAST:     %[[V_16:.*]] = stablehlo.subtract %arg2, %[[V_15]] : tensor<4x3x8xf64>
// LAST:     %[[V_17:.*]] = stablehlo.convert %[[V_16]] : (tensor<4x3x8xf64>) -> tensor<4x3x8xf32>
// LAST:     %[[V_18:.*]] = stablehlo.reshape %[[V_14]] : (tensor<4x3x8xf32>) -> tensor<4x3x8x1xf32>
// LAST:     %[[V_19:.*]] = stablehlo.reshape %[[V_17]] : (tensor<4x3x8xf32>) -> tensor<4x3x8x1xf32>
// LAST:     %[[V_20:.*]] = stablehlo.concatenate %[[V_18]], %[[V_19]], dim = 3 : (tensor<4x3x8x1xf32>, tensor<4x3x8x1xf32>) -> tensor<4x3x8x2xf32>
// LAST:     %[[V_21:.*]] = stablehlo.convert %arg3 : (tensor<4x1x8xf64>) -> tensor<4x1x8xf32>
// LAST:     %[[V_22:.*]] = stablehlo.convert %[[V_21]] : (tensor<4x1x8xf32>) -> tensor<4x1x8xf64>
// LAST:     %[[V_23:.*]] = stablehlo.subtract %arg3, %[[V_22]] : tensor<4x1x8xf64>
// LAST:     %[[V_24:.*]] = stablehlo.convert %[[V_23]] : (tensor<4x1x8xf64>) -> tensor<4x1x8xf32>
// LAST:     %[[V_25:.*]] = stablehlo.reshape %[[V_21]] : (tensor<4x1x8xf32>) -> tensor<4x1x8x1xf32>
// LAST:     %[[V_26:.*]] = stablehlo.reshape %[[V_24]] : (tensor<4x1x8xf32>) -> tensor<4x1x8x1xf32>
// LAST:     %[[V_27:.*]] = stablehlo.concatenate %[[V_25]], %[[V_26]], dim = 3 : (tensor<4x1x8x1xf32>, tensor<4x1x8x1xf32>) -> tensor<4x1x8x2xf32>
// LAST:     %[[V_28:.*]] = stablehlo.convert %arg4 : (tensor<4x1x8xf64>) -> tensor<4x1x8xf32>
// LAST:     %[[V_29:.*]] = stablehlo.convert %[[V_28]] : (tensor<4x1x8xf32>) -> tensor<4x1x8xf64>
// LAST:     %[[V_30:.*]] = stablehlo.subtract %arg4, %[[V_29]] : tensor<4x1x8xf64>
// LAST:     %[[V_31:.*]] = stablehlo.convert %[[V_30]] : (tensor<4x1x8xf64>) -> tensor<4x1x8xf32>
// LAST:     %[[V_32:.*]] = stablehlo.reshape %[[V_28]] : (tensor<4x1x8xf32>) -> tensor<4x1x8x1xf32>
// LAST:     %[[V_33:.*]] = stablehlo.reshape %[[V_31]] : (tensor<4x1x8xf32>) -> tensor<4x1x8x1xf32>
// LAST:     %[[V_34:.*]] = stablehlo.concatenate %[[V_32]], %[[V_33]], dim = 3 : (tensor<4x1x8x1xf32>, tensor<4x1x8x1xf32>) -> tensor<4x1x8x2xf32>
// LAST:     %[[V_35:.*]] = stablehlo.concatenate %[[V_6]], %[[V_13]], %[[V_20]], %[[V_27]], %[[V_34]], dim = 1 : (tensor<4x1x8x2xf32>, tensor<4x1x8x2xf32>, tensor<4x3x8x2xf32>, tensor<4x1x8x2xf32>, tensor<4x1x8x2xf32>) -> tensor<4x7x8x2xf32>
// LAST:     %[[V_36:.*]] = stablehlo.convert %[[V_35]] : (tensor<4x7x8x2xf32>) -> tensor<4x7x8x2xf64>
// LAST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// LAST:     %[[V_37:.*]] = stablehlo.reduce(%[[V_36]] init: %[[CST]]) applies stablehlo.add across dimensions = [3] : (tensor<4x7x8x2xf64>, tensor<f64>) -> tensor<4x7x8xf64>
// LAST:     return %[[V_37]] : tensor<4x7x8xf64>

// FIRST-LABEL: func.func @main

// LAST-LABEL: func.func @main

// TUPLE-LABEL: func.func @main

func.func @main() attributes {enzyme.no_multifloat} {
  %arg0 = stablehlo.constant dense<[[1.0, 1.5, 2.0, 2.5],
                                   [3.0, 3.5, 4.0, 4.5],
                                   [5.0, 5.5, 6.0, 6.5]]> : tensor<3x4xf64>
                                   
  %arg1 = stablehlo.constant dense<[[7.0, 7.5, 8.0, 8.5]]> : tensor<1x4xf64>
  
  %expected = stablehlo.constant dense<[[1.0, 1.5, 2.0, 2.5],
                                        [3.0, 3.5, 4.0, 4.5],
                                        [5.0, 5.5, 6.0, 6.5],
                                        [7.0, 7.5, 8.0, 8.5]]> : tensor<4x4xf64>
                                        
  %res = func.call @concat_2op(%arg0, %arg1) : (tensor<3x4xf64>, tensor<1x4xf64>) -> tensor<4x4xf64>
  
  // Strict test against Julia MultiFloat
  "check.expect_close"(%res, %expected) {max_ulp_difference = 0 : ui64} : (tensor<4x4xf64>, tensor<4x4xf64>) -> ()
  
  // Approximate test against regular f64
  "check.expect_close"(%res, %expected) {max_ulp_difference = 0 : ui64} : (tensor<4x4xf64>, tensor<4x4xf64>) -> ()
  return
}
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<3x4xf64>
// FIRST:     %[[CST_0:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<1x4xf64>
// FIRST:     %[[CST_1:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<4x4xf64>
// FIRST:     %[[V_0:.*]] = call @concat_2op(%[[CST]], %[[CST_0]]) : (tensor<3x4xf64>, tensor<1x4xf64>) -> tensor<4x4xf64>
// FIRST:     check.expect_close %[[V_0]], %[[CST_1]], max_ulp_difference = 0 : tensor<4x4xf64>, tensor<4x4xf64>
// FIRST:     check.expect_close %[[V_0]], %[[CST_1]], max_ulp_difference = 0 : tensor<4x4xf64>, tensor<4x4xf64>
// FIRST:     return
