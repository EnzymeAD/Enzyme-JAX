// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck --check-prefix=FIRST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=TUPLE %s
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @divide(%arg0: tensor<8xf64>, %arg1: tensor<8xf64>) -> tensor<8xf64> {
  %0 = stablehlo.divide %arg0, %arg1 : tensor<8xf64>
  return %0 : tensor<8xf64>
}

// FIRST-LABEL: func.func @divide
// FIRST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<8xf64>) -> tensor<8xf32>
// FIRST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<8xf32>) -> tensor<8xf64>
// FIRST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<8xf64>
// FIRST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<8xf64>) -> tensor<8xf32>
// FIRST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<8xf32>) -> tensor<1x8xf32>
// FIRST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<8xf32>) -> tensor<1x8xf32>
// FIRST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 0 : (tensor<1x8xf32>, tensor<1x8xf32>) -> tensor<2x8xf32>
// FIRST:     %[[V_7:.*]] = stablehlo.convert %arg1 : (tensor<8xf64>) -> tensor<8xf32>
// FIRST:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<8xf32>) -> tensor<8xf64>
// FIRST:     %[[V_9:.*]] = stablehlo.subtract %arg1, %[[V_8]] : tensor<8xf64>
// FIRST:     %[[V_10:.*]] = stablehlo.convert %[[V_9]] : (tensor<8xf64>) -> tensor<8xf32>
// FIRST:     %[[V_11:.*]] = stablehlo.reshape %[[V_7]] : (tensor<8xf32>) -> tensor<1x8xf32>
// FIRST:     %[[V_12:.*]] = stablehlo.reshape %[[V_10]] : (tensor<8xf32>) -> tensor<1x8xf32>
// FIRST:     %[[V_13:.*]] = stablehlo.concatenate %[[V_11]], %[[V_12]], dim = 0 : (tensor<1x8xf32>, tensor<1x8xf32>) -> tensor<2x8xf32>
// FIRST:     %[[V_14:.*]] = stablehlo.slice %[[V_6]] [0:1, 0:8] : (tensor<2x8xf32>) -> tensor<1x8xf32>
// FIRST:     %[[V_15:.*]] = stablehlo.slice %[[V_6]] [1:2, 0:8] : (tensor<2x8xf32>) -> tensor<1x8xf32>
// FIRST:     %[[V_16:.*]] = stablehlo.slice %[[V_13]] [0:1, 0:8] : (tensor<2x8xf32>) -> tensor<1x8xf32>
// FIRST:     %[[V_17:.*]] = stablehlo.slice %[[V_13]] [1:2, 0:8] : (tensor<2x8xf32>) -> tensor<1x8xf32>
// FIRST:     %[[V_18:.*]] = stablehlo.divide %[[V_14]], %[[V_16]] : tensor<1x8xf32>
// FIRST:     %[[V_19:.*]] = stablehlo.multiply %[[V_18]], %[[V_16]] : tensor<1x8xf32>
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<4.097000e+03> : tensor<1x8xf32>
// FIRST:     %[[V_20:.*]] = stablehlo.multiply %[[V_18]], %[[CST]] : tensor<1x8xf32>
// FIRST:     %[[V_21:.*]] = stablehlo.subtract %[[V_20]], %[[V_18]] : tensor<1x8xf32>
// FIRST:     %[[V_22:.*]] = stablehlo.subtract %[[V_20]], %[[V_21]] : tensor<1x8xf32>
// FIRST:     %[[V_23:.*]] = stablehlo.subtract %[[V_18]], %[[V_22]] : tensor<1x8xf32>
// FIRST:     %[[CST_0:.*]] = stablehlo.constant dense<4.097000e+03> : tensor<1x8xf32>
// FIRST:     %[[V_24:.*]] = stablehlo.multiply %[[V_16]], %[[CST_0]] : tensor<1x8xf32>
// FIRST:     %[[V_25:.*]] = stablehlo.subtract %[[V_24]], %[[V_16]] : tensor<1x8xf32>
// FIRST:     %[[V_26:.*]] = stablehlo.subtract %[[V_24]], %[[V_25]] : tensor<1x8xf32>
// FIRST:     %[[V_27:.*]] = stablehlo.subtract %[[V_16]], %[[V_26]] : tensor<1x8xf32>
// FIRST:     %[[V_28:.*]] = stablehlo.multiply %[[V_22]], %[[V_26]] : tensor<1x8xf32>
// FIRST:     %[[V_29:.*]] = stablehlo.multiply %[[V_22]], %[[V_27]] : tensor<1x8xf32>
// FIRST:     %[[V_30:.*]] = stablehlo.multiply %[[V_23]], %[[V_26]] : tensor<1x8xf32>
// FIRST:     %[[V_31:.*]] = stablehlo.multiply %[[V_23]], %[[V_27]] : tensor<1x8xf32>
// FIRST:     %[[V_32:.*]] = stablehlo.subtract %[[V_28]], %[[V_19]] : tensor<1x8xf32>
// FIRST:     %[[V_33:.*]] = stablehlo.add %[[V_29]], %[[V_30]] : tensor<1x8xf32>
// FIRST:     %[[V_34:.*]] = stablehlo.add %[[V_32]], %[[V_33]] : tensor<1x8xf32>
// FIRST:     %[[V_35:.*]] = stablehlo.add %[[V_34]], %[[V_31]] : tensor<1x8xf32>
// FIRST:     %[[V_36:.*]] = stablehlo.negate %[[V_19]] : tensor<1x8xf32>
// FIRST:     %[[V_37:.*]] = stablehlo.negate %[[V_35]] : tensor<1x8xf32>
// FIRST:     %[[V_38:.*]] = stablehlo.add %[[V_14]], %[[V_36]] : tensor<1x8xf32>
// FIRST:     %[[V_39:.*]] = stablehlo.add %[[V_38]], %[[V_37]] : tensor<1x8xf32>
// FIRST:     %[[V_40:.*]] = stablehlo.add %[[V_39]], %[[V_15]] : tensor<1x8xf32>
// FIRST:     %[[V_41:.*]] = stablehlo.multiply %[[V_18]], %[[V_17]] : tensor<1x8xf32>
// FIRST:     %[[V_42:.*]] = stablehlo.negate %[[V_41]] : tensor<1x8xf32>
// FIRST:     %[[V_43:.*]] = stablehlo.add %[[V_40]], %[[V_42]] : tensor<1x8xf32>
// FIRST:     %[[V_44:.*]] = stablehlo.divide %[[V_43]], %[[V_16]] : tensor<1x8xf32>
// FIRST:     %[[V_45:.*]] = stablehlo.add %[[V_18]], %[[V_44]] : tensor<1x8xf32>
// FIRST:     %[[V_46:.*]] = stablehlo.subtract %[[V_45]], %[[V_18]] : tensor<1x8xf32>
// FIRST:     %[[V_47:.*]] = stablehlo.subtract %[[V_44]], %[[V_46]] : tensor<1x8xf32>
// FIRST:     %[[V_48:.*]] = stablehlo.concatenate %[[V_45]], %[[V_47]], dim = 0 : (tensor<1x8xf32>, tensor<1x8xf32>) -> tensor<2x8xf32>
// FIRST:     %[[V_49:.*]] = stablehlo.convert %[[V_48]] : (tensor<2x8xf32>) -> tensor<2x8xf64>
// FIRST:     %[[CST_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// FIRST:     %[[V_50:.*]] = stablehlo.reduce(%[[V_49]] init: %[[CST_1]]) applies stablehlo.add across dimensions = [0] : (tensor<2x8xf64>, tensor<f64>) -> tensor<8xf64>
// FIRST:     return %[[V_50]] : tensor<8xf64>

// LAST-LABEL: func.func @divide
// LAST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<8xf64>) -> tensor<8xf32>
// LAST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<8xf32>) -> tensor<8xf64>
// LAST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<8xf64>
// LAST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<8xf64>) -> tensor<8xf32>
// LAST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<8xf32>) -> tensor<8x1xf32>
// LAST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<8xf32>) -> tensor<8x1xf32>
// LAST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 1 : (tensor<8x1xf32>, tensor<8x1xf32>) -> tensor<8x2xf32>
// LAST:     %[[V_7:.*]] = stablehlo.convert %arg1 : (tensor<8xf64>) -> tensor<8xf32>
// LAST:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<8xf32>) -> tensor<8xf64>
// LAST:     %[[V_9:.*]] = stablehlo.subtract %arg1, %[[V_8]] : tensor<8xf64>
// LAST:     %[[V_10:.*]] = stablehlo.convert %[[V_9]] : (tensor<8xf64>) -> tensor<8xf32>
// LAST:     %[[V_11:.*]] = stablehlo.reshape %[[V_7]] : (tensor<8xf32>) -> tensor<8x1xf32>
// LAST:     %[[V_12:.*]] = stablehlo.reshape %[[V_10]] : (tensor<8xf32>) -> tensor<8x1xf32>
// LAST:     %[[V_13:.*]] = stablehlo.concatenate %[[V_11]], %[[V_12]], dim = 1 : (tensor<8x1xf32>, tensor<8x1xf32>) -> tensor<8x2xf32>
// LAST:     %[[V_14:.*]] = stablehlo.slice %[[V_6]] [0:8, 0:1] : (tensor<8x2xf32>) -> tensor<8x1xf32>
// LAST:     %[[V_15:.*]] = stablehlo.slice %[[V_6]] [0:8, 1:2] : (tensor<8x2xf32>) -> tensor<8x1xf32>
// LAST:     %[[V_16:.*]] = stablehlo.slice %[[V_13]] [0:8, 0:1] : (tensor<8x2xf32>) -> tensor<8x1xf32>
// LAST:     %[[V_17:.*]] = stablehlo.slice %[[V_13]] [0:8, 1:2] : (tensor<8x2xf32>) -> tensor<8x1xf32>
// LAST:     %[[V_18:.*]] = stablehlo.divide %[[V_14]], %[[V_16]] : tensor<8x1xf32>
// LAST:     %[[V_19:.*]] = stablehlo.multiply %[[V_18]], %[[V_16]] : tensor<8x1xf32>
// LAST:     %[[CST:.*]] = stablehlo.constant dense<4.097000e+03> : tensor<8x1xf32>
// LAST:     %[[V_20:.*]] = stablehlo.multiply %[[V_18]], %[[CST]] : tensor<8x1xf32>
// LAST:     %[[V_21:.*]] = stablehlo.subtract %[[V_20]], %[[V_18]] : tensor<8x1xf32>
// LAST:     %[[V_22:.*]] = stablehlo.subtract %[[V_20]], %[[V_21]] : tensor<8x1xf32>
// LAST:     %[[V_23:.*]] = stablehlo.subtract %[[V_18]], %[[V_22]] : tensor<8x1xf32>
// LAST:     %[[CST_0:.*]] = stablehlo.constant dense<4.097000e+03> : tensor<8x1xf32>
// LAST:     %[[V_24:.*]] = stablehlo.multiply %[[V_16]], %[[CST_0]] : tensor<8x1xf32>
// LAST:     %[[V_25:.*]] = stablehlo.subtract %[[V_24]], %[[V_16]] : tensor<8x1xf32>
// LAST:     %[[V_26:.*]] = stablehlo.subtract %[[V_24]], %[[V_25]] : tensor<8x1xf32>
// LAST:     %[[V_27:.*]] = stablehlo.subtract %[[V_16]], %[[V_26]] : tensor<8x1xf32>
// LAST:     %[[V_28:.*]] = stablehlo.multiply %[[V_22]], %[[V_26]] : tensor<8x1xf32>
// LAST:     %[[V_29:.*]] = stablehlo.multiply %[[V_22]], %[[V_27]] : tensor<8x1xf32>
// LAST:     %[[V_30:.*]] = stablehlo.multiply %[[V_23]], %[[V_26]] : tensor<8x1xf32>
// LAST:     %[[V_31:.*]] = stablehlo.multiply %[[V_23]], %[[V_27]] : tensor<8x1xf32>
// LAST:     %[[V_32:.*]] = stablehlo.subtract %[[V_28]], %[[V_19]] : tensor<8x1xf32>
// LAST:     %[[V_33:.*]] = stablehlo.add %[[V_29]], %[[V_30]] : tensor<8x1xf32>
// LAST:     %[[V_34:.*]] = stablehlo.add %[[V_32]], %[[V_33]] : tensor<8x1xf32>
// LAST:     %[[V_35:.*]] = stablehlo.add %[[V_34]], %[[V_31]] : tensor<8x1xf32>
// LAST:     %[[V_36:.*]] = stablehlo.negate %[[V_19]] : tensor<8x1xf32>
// LAST:     %[[V_37:.*]] = stablehlo.negate %[[V_35]] : tensor<8x1xf32>
// LAST:     %[[V_38:.*]] = stablehlo.add %[[V_14]], %[[V_36]] : tensor<8x1xf32>
// LAST:     %[[V_39:.*]] = stablehlo.add %[[V_38]], %[[V_37]] : tensor<8x1xf32>
// LAST:     %[[V_40:.*]] = stablehlo.add %[[V_39]], %[[V_15]] : tensor<8x1xf32>
// LAST:     %[[V_41:.*]] = stablehlo.multiply %[[V_18]], %[[V_17]] : tensor<8x1xf32>
// LAST:     %[[V_42:.*]] = stablehlo.negate %[[V_41]] : tensor<8x1xf32>
// LAST:     %[[V_43:.*]] = stablehlo.add %[[V_40]], %[[V_42]] : tensor<8x1xf32>
// LAST:     %[[V_44:.*]] = stablehlo.divide %[[V_43]], %[[V_16]] : tensor<8x1xf32>
// LAST:     %[[V_45:.*]] = stablehlo.add %[[V_18]], %[[V_44]] : tensor<8x1xf32>
// LAST:     %[[V_46:.*]] = stablehlo.subtract %[[V_45]], %[[V_18]] : tensor<8x1xf32>
// LAST:     %[[V_47:.*]] = stablehlo.subtract %[[V_44]], %[[V_46]] : tensor<8x1xf32>
// LAST:     %[[V_48:.*]] = stablehlo.concatenate %[[V_45]], %[[V_47]], dim = 1 : (tensor<8x1xf32>, tensor<8x1xf32>) -> tensor<8x2xf32>
// LAST:     %[[V_49:.*]] = stablehlo.convert %[[V_48]] : (tensor<8x2xf32>) -> tensor<8x2xf64>
// LAST:     %[[CST_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// LAST:     %[[V_50:.*]] = stablehlo.reduce(%[[V_49]] init: %[[CST_1]]) applies stablehlo.add across dimensions = [1] : (tensor<8x2xf64>, tensor<f64>) -> tensor<8xf64>
// LAST:     return %[[V_50]] : tensor<8xf64>

// TUPLE-LABEL: func.func @divide
// TUPLE:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<8xf64>) -> tensor<8xf32>
// TUPLE:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<8xf32>) -> tensor<8xf64>
// TUPLE:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<8xf64>
// TUPLE:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<8xf64>) -> tensor<8xf32>
// TUPLE:     %[[V_4:.*]] = stablehlo.tuple %[[V_0]], %[[V_3]] : tuple<tensor<8xf32>, tensor<8xf32>>
// TUPLE:     %[[V_5:.*]] = stablehlo.convert %arg1 : (tensor<8xf64>) -> tensor<8xf32>
// TUPLE:     %[[V_6:.*]] = stablehlo.convert %[[V_5]] : (tensor<8xf32>) -> tensor<8xf64>
// TUPLE:     %[[V_7:.*]] = stablehlo.subtract %arg1, %[[V_6]] : tensor<8xf64>
// TUPLE:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<8xf64>) -> tensor<8xf32>
// TUPLE:     %[[V_9:.*]] = stablehlo.tuple %[[V_5]], %[[V_8]] : tuple<tensor<8xf32>, tensor<8xf32>>
// TUPLE:     %[[V_10:.*]] = stablehlo.get_tuple_element %[[V_4]][0] : (tuple<tensor<8xf32>, tensor<8xf32>>) -> tensor<8xf32>
// TUPLE:     %[[V_11:.*]] = stablehlo.get_tuple_element %[[V_4]][1] : (tuple<tensor<8xf32>, tensor<8xf32>>) -> tensor<8xf32>
// TUPLE:     %[[V_12:.*]] = stablehlo.get_tuple_element %[[V_9]][0] : (tuple<tensor<8xf32>, tensor<8xf32>>) -> tensor<8xf32>
// TUPLE:     %[[V_13:.*]] = stablehlo.get_tuple_element %[[V_9]][1] : (tuple<tensor<8xf32>, tensor<8xf32>>) -> tensor<8xf32>
// TUPLE:     %[[V_14:.*]] = stablehlo.divide %[[V_10]], %[[V_12]] : tensor<8xf32>
// TUPLE:     %[[V_15:.*]] = stablehlo.multiply %[[V_14]], %[[V_12]] : tensor<8xf32>
// TUPLE:     %[[CST:.*]] = stablehlo.constant dense<4.097000e+03> : tensor<8xf32>
// TUPLE:     %[[V_16:.*]] = stablehlo.multiply %[[V_14]], %[[CST]] : tensor<8xf32>
// TUPLE:     %[[V_17:.*]] = stablehlo.subtract %[[V_16]], %[[V_14]] : tensor<8xf32>
// TUPLE:     %[[V_18:.*]] = stablehlo.subtract %[[V_16]], %[[V_17]] : tensor<8xf32>
// TUPLE:     %[[V_19:.*]] = stablehlo.subtract %[[V_14]], %[[V_18]] : tensor<8xf32>
// TUPLE:     %[[CST_0:.*]] = stablehlo.constant dense<4.097000e+03> : tensor<8xf32>
// TUPLE:     %[[V_20:.*]] = stablehlo.multiply %[[V_12]], %[[CST_0]] : tensor<8xf32>
// TUPLE:     %[[V_21:.*]] = stablehlo.subtract %[[V_20]], %[[V_12]] : tensor<8xf32>
// TUPLE:     %[[V_22:.*]] = stablehlo.subtract %[[V_20]], %[[V_21]] : tensor<8xf32>
// TUPLE:     %[[V_23:.*]] = stablehlo.subtract %[[V_12]], %[[V_22]] : tensor<8xf32>
// TUPLE:     %[[V_24:.*]] = stablehlo.multiply %[[V_18]], %[[V_22]] : tensor<8xf32>
// TUPLE:     %[[V_25:.*]] = stablehlo.multiply %[[V_18]], %[[V_23]] : tensor<8xf32>
// TUPLE:     %[[V_26:.*]] = stablehlo.multiply %[[V_19]], %[[V_22]] : tensor<8xf32>
// TUPLE:     %[[V_27:.*]] = stablehlo.multiply %[[V_19]], %[[V_23]] : tensor<8xf32>
// TUPLE:     %[[V_28:.*]] = stablehlo.subtract %[[V_24]], %[[V_15]] : tensor<8xf32>
// TUPLE:     %[[V_29:.*]] = stablehlo.add %[[V_25]], %[[V_26]] : tensor<8xf32>
// TUPLE:     %[[V_30:.*]] = stablehlo.add %[[V_28]], %[[V_29]] : tensor<8xf32>
// TUPLE:     %[[V_31:.*]] = stablehlo.add %[[V_30]], %[[V_27]] : tensor<8xf32>
// TUPLE:     %[[V_32:.*]] = stablehlo.negate %[[V_15]] : tensor<8xf32>
// TUPLE:     %[[V_33:.*]] = stablehlo.negate %[[V_31]] : tensor<8xf32>
// TUPLE:     %[[V_34:.*]] = stablehlo.add %[[V_10]], %[[V_32]] : tensor<8xf32>
// TUPLE:     %[[V_35:.*]] = stablehlo.add %[[V_34]], %[[V_33]] : tensor<8xf32>
// TUPLE:     %[[V_36:.*]] = stablehlo.add %[[V_35]], %[[V_11]] : tensor<8xf32>
// TUPLE:     %[[V_37:.*]] = stablehlo.multiply %[[V_14]], %[[V_13]] : tensor<8xf32>
// TUPLE:     %[[V_38:.*]] = stablehlo.negate %[[V_37]] : tensor<8xf32>
// TUPLE:     %[[V_39:.*]] = stablehlo.add %[[V_36]], %[[V_38]] : tensor<8xf32>
// TUPLE:     %[[V_40:.*]] = stablehlo.divide %[[V_39]], %[[V_12]] : tensor<8xf32>
// TUPLE:     %[[V_41:.*]] = stablehlo.add %[[V_14]], %[[V_40]] : tensor<8xf32>
// TUPLE:     %[[V_42:.*]] = stablehlo.subtract %[[V_41]], %[[V_14]] : tensor<8xf32>
// TUPLE:     %[[V_43:.*]] = stablehlo.subtract %[[V_40]], %[[V_42]] : tensor<8xf32>
// TUPLE:     %[[V_44:.*]] = stablehlo.tuple %[[V_41]], %[[V_43]] : tuple<tensor<8xf32>, tensor<8xf32>>
// TUPLE:     %[[V_45:.*]] = stablehlo.convert %[[V_41]] : (tensor<8xf32>) -> tensor<8xf64>
// TUPLE:     %[[V_46:.*]] = stablehlo.convert %[[V_43]] : (tensor<8xf32>) -> tensor<8xf64>
// TUPLE:     %[[V_47:.*]] = stablehlo.add %[[V_45]], %[[V_46]] : tensor<8xf64>
// TUPLE:     return %[[V_47]] : tensor<8xf64>

func.func @main() attributes {enzyme.no_multifloat} {
  %cst1 = stablehlo.constant dense<[1.0, 1.0, 0.0, 1.0, 1.1, -1.0, -1.0, 2.2]> : tensor<8xf64>
  %cst2 = stablehlo.constant dense<[3.0, -3.0, 3.0, 0.0, 3.0, 3.0, -3.0, 1.00000003]> : tensor<8xf64>
  
  %expected_mf = stablehlo.constant dense<[0.33333333333333304, -0.33333333333333304, 0.0, 0x7FF8000000000000, 0.36666666666666625, -0.33333333333333304, 0.33333333333333304, 2.1999999340000045]> : tensor<8xf64>
  
  %res = func.call @divide(%cst1, %cst2) : (tensor<8xf64>, tensor<8xf64>) -> tensor<8xf64>
  
  // Strict test against Julia MultiFloat (relaxed tolerance due to our higher accuracy on index 4)
  "check.expect_close"(%res, %expected_mf) {max_ulp_difference = 20 : ui64} : (tensor<8xf64>, tensor<8xf64>) -> ()
  
  // Approximate test against regular f64
  %expected_f64 = stablehlo.constant dense<[0.3333333333333333, -0.3333333333333333, 0.0, 0x7FF8000000000000, 0.36666666666666664, -0.3333333333333333, 0.3333333333333333, 2.1999999340000023]> : tensor<8xf64>
  "check.expect_close"(%res, %expected_f64) {max_ulp_difference = 15 : ui64} : (tensor<8xf64>, tensor<8xf64>) -> ()
  return
}

// FIRST-LABEL: func.func @main
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<8xf64>
// FIRST:     %[[CST_0:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<8xf64>
// FIRST:     %[[CST_1:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<8xf64>
// FIRST:     %[[V_0:.*]] = call @divide(%[[CST]], %[[CST_0]]) : (tensor<8xf64>, tensor<8xf64>) -> tensor<8xf64>
// FIRST:     check.expect_close %[[V_0]], %[[CST_1]], max_ulp_difference = {{[0-9]+}} : tensor<8xf64>, tensor<8xf64>
// FIRST:     %[[CST_2:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<8xf64>
// FIRST:     check.expect_close %[[V_0]], %[[CST_2]], max_ulp_difference = {{[0-9]+}} : tensor<8xf64>, tensor<8xf64>
// FIRST:     return
