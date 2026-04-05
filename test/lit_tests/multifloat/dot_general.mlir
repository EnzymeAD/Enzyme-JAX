// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck --check-prefix=FIRST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=TUPLE %s
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @dot_general(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>) -> tensor<f64> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [0] : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
  return %0 : tensor<f64>
}

// TUPLE-LABEL: {{.*}}func.func @dot_general(
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
// TUPLE:     %[[V_14:.*]] = stablehlo.reshape %[[V_10]] : (tensor<2xf32>) -> tensor<2xf32>
// TUPLE:     %[[V_15:.*]] = stablehlo.reshape %[[V_11]] : (tensor<2xf32>) -> tensor<2xf32>
// TUPLE:     %[[V_16:.*]] = stablehlo.reshape %[[V_12]] : (tensor<2xf32>) -> tensor<2xf32>
// TUPLE:     %[[V_17:.*]] = stablehlo.reshape %[[V_13]] : (tensor<2xf32>) -> tensor<2xf32>
// TUPLE:     %[[CST:.*]] = stablehlo.constant dense<4.097000e+03> : tensor<2xf32>
// TUPLE:     %[[V_18:.*]] = stablehlo.multiply %[[V_14]], %[[CST]] : tensor<2xf32>
// TUPLE:     %[[V_19:.*]] = stablehlo.subtract %[[V_18]], %[[V_14]] : tensor<2xf32>
// TUPLE:     %[[V_20:.*]] = stablehlo.subtract %[[V_18]], %[[V_19]] : tensor<2xf32>
// TUPLE:     %[[V_21:.*]] = stablehlo.subtract %[[V_14]], %[[V_20]] : tensor<2xf32>
// TUPLE:     %[[CST_0:.*]] = stablehlo.constant dense<4.097000e+03> : tensor<2xf32>
// TUPLE:     %[[V_22:.*]] = stablehlo.multiply %[[V_16]], %[[CST_0]] : tensor<2xf32>
// TUPLE:     %[[V_23:.*]] = stablehlo.subtract %[[V_22]], %[[V_16]] : tensor<2xf32>
// TUPLE:     %[[V_24:.*]] = stablehlo.subtract %[[V_22]], %[[V_23]] : tensor<2xf32>
// TUPLE:     %[[V_25:.*]] = stablehlo.subtract %[[V_16]], %[[V_24]] : tensor<2xf32>
// TUPLE:     %[[V_26:.*]] = stablehlo.dot_general %[[V_20]], %[[V_24]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// TUPLE:     %[[V_27:.*]] = stablehlo.dot_general %[[V_20]], %[[V_25]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// TUPLE:     %[[V_28:.*]] = stablehlo.dot_general %[[V_21]], %[[V_24]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// TUPLE:     %[[V_29:.*]] = stablehlo.dot_general %[[V_21]], %[[V_25]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// TUPLE:     %[[V_30:.*]] = stablehlo.dot_general %[[V_14]], %[[V_16]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// TUPLE:     %[[V_31:.*]] = stablehlo.subtract %[[V_26]], %[[V_30]] : tensor<f32>
// TUPLE:     %[[V_32:.*]] = stablehlo.add %[[V_27]], %[[V_28]] : tensor<f32>
// TUPLE:     %[[V_33:.*]] = stablehlo.add %[[V_31]], %[[V_32]] : tensor<f32>
// TUPLE:     %[[V_34:.*]] = stablehlo.add %[[V_33]], %[[V_29]] : tensor<f32>
// TUPLE:     %[[V_35:.*]] = stablehlo.dot_general %[[V_14]], %[[V_17]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// TUPLE:     %[[V_36:.*]] = stablehlo.dot_general %[[V_15]], %[[V_16]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// TUPLE:     %[[V_37:.*]] = stablehlo.add %[[V_34]], %[[V_35]] : tensor<f32>
// TUPLE:     %[[V_38:.*]] = stablehlo.add %[[V_37]], %[[V_36]] : tensor<f32>
// TUPLE:     %[[V_39:.*]] = stablehlo.add %[[V_30]], %[[V_38]] : tensor<f32>
// TUPLE:     %[[V_40:.*]] = stablehlo.subtract %[[V_39]], %[[V_30]] : tensor<f32>
// TUPLE:     %[[V_41:.*]] = stablehlo.subtract %[[V_38]], %[[V_40]] : tensor<f32>
// TUPLE:     %[[V_42:.*]] = stablehlo.tuple %[[V_39]], %[[V_41]] : tuple<tensor<f32>, tensor<f32>>
// TUPLE:     %[[V_43:.*]] = stablehlo.convert %[[V_39]] : (tensor<f32>) -> tensor<f64>
// TUPLE:     %[[V_44:.*]] = stablehlo.convert %[[V_41]] : (tensor<f32>) -> tensor<f64>
// TUPLE:     %[[V_45:.*]] = stablehlo.add %[[V_43]], %[[V_44]] : tensor<f64>
// TUPLE:     return %[[V_45]] : tensor<f64>

// FIRST-LABEL: {{.*}}func.func @dot_general(
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
// FIRST:     %[[V_18:.*]] = stablehlo.reshape %[[V_14]] : (tensor<1x2xf32>) -> tensor<2xf32>
// FIRST:     %[[V_19:.*]] = stablehlo.reshape %[[V_15]] : (tensor<1x2xf32>) -> tensor<2xf32>
// FIRST:     %[[V_20:.*]] = stablehlo.reshape %[[V_16]] : (tensor<1x2xf32>) -> tensor<2xf32>
// FIRST:     %[[V_21:.*]] = stablehlo.reshape %[[V_17]] : (tensor<1x2xf32>) -> tensor<2xf32>
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<4.097000e+03> : tensor<2xf32>
// FIRST:     %[[V_22:.*]] = stablehlo.multiply %[[V_18]], %[[CST]] : tensor<2xf32>
// FIRST:     %[[V_23:.*]] = stablehlo.subtract %[[V_22]], %[[V_18]] : tensor<2xf32>
// FIRST:     %[[V_24:.*]] = stablehlo.subtract %[[V_22]], %[[V_23]] : tensor<2xf32>
// FIRST:     %[[V_25:.*]] = stablehlo.subtract %[[V_18]], %[[V_24]] : tensor<2xf32>
// FIRST:     %[[CST_0:.*]] = stablehlo.constant dense<4.097000e+03> : tensor<2xf32>
// FIRST:     %[[V_26:.*]] = stablehlo.multiply %[[V_20]], %[[CST_0]] : tensor<2xf32>
// FIRST:     %[[V_27:.*]] = stablehlo.subtract %[[V_26]], %[[V_20]] : tensor<2xf32>
// FIRST:     %[[V_28:.*]] = stablehlo.subtract %[[V_26]], %[[V_27]] : tensor<2xf32>
// FIRST:     %[[V_29:.*]] = stablehlo.subtract %[[V_20]], %[[V_28]] : tensor<2xf32>
// FIRST:     %[[V_30:.*]] = stablehlo.dot_general %[[V_24]], %[[V_28]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// FIRST:     %[[V_31:.*]] = stablehlo.dot_general %[[V_24]], %[[V_29]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// FIRST:     %[[V_32:.*]] = stablehlo.dot_general %[[V_25]], %[[V_28]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// FIRST:     %[[V_33:.*]] = stablehlo.dot_general %[[V_25]], %[[V_29]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// FIRST:     %[[V_34:.*]] = stablehlo.dot_general %[[V_18]], %[[V_20]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// FIRST:     %[[V_35:.*]] = stablehlo.subtract %[[V_30]], %[[V_34]] : tensor<f32>
// FIRST:     %[[V_36:.*]] = stablehlo.add %[[V_31]], %[[V_32]] : tensor<f32>
// FIRST:     %[[V_37:.*]] = stablehlo.add %[[V_35]], %[[V_36]] : tensor<f32>
// FIRST:     %[[V_38:.*]] = stablehlo.add %[[V_37]], %[[V_33]] : tensor<f32>
// FIRST:     %[[V_39:.*]] = stablehlo.dot_general %[[V_18]], %[[V_21]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// FIRST:     %[[V_40:.*]] = stablehlo.dot_general %[[V_19]], %[[V_20]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// FIRST:     %[[V_41:.*]] = stablehlo.add %[[V_38]], %[[V_39]] : tensor<f32>
// FIRST:     %[[V_42:.*]] = stablehlo.add %[[V_41]], %[[V_40]] : tensor<f32>
// FIRST:     %[[V_43:.*]] = stablehlo.add %[[V_34]], %[[V_42]] : tensor<f32>
// FIRST:     %[[V_44:.*]] = stablehlo.subtract %[[V_43]], %[[V_34]] : tensor<f32>
// FIRST:     %[[V_45:.*]] = stablehlo.subtract %[[V_42]], %[[V_44]] : tensor<f32>
// FIRST:     %[[V_46:.*]] = stablehlo.reshape %[[V_43]] : (tensor<f32>) -> tensor<1xf32>
// FIRST:     %[[V_47:.*]] = stablehlo.reshape %[[V_45]] : (tensor<f32>) -> tensor<1xf32>
// FIRST:     %[[V_48:.*]] = stablehlo.concatenate %[[V_46]], %[[V_47]], dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
// FIRST:     %[[V_49:.*]] = stablehlo.convert %[[V_48]] : (tensor<2xf32>) -> tensor<2xf64>
// FIRST:     %[[CST_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// FIRST:     %[[V_50:.*]] = stablehlo.reduce(%[[V_49]] init: %[[CST_1]]) applies stablehlo.add across dimensions = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<f64>
// FIRST:     return %[[V_50]] : tensor<f64>

// LAST-LABEL: {{.*}}func.func @dot_general(
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
// LAST:     %[[V_18:.*]] = stablehlo.reshape %[[V_14]] : (tensor<2x1xf32>) -> tensor<2xf32>
// LAST:     %[[V_19:.*]] = stablehlo.reshape %[[V_15]] : (tensor<2x1xf32>) -> tensor<2xf32>
// LAST:     %[[V_20:.*]] = stablehlo.reshape %[[V_16]] : (tensor<2x1xf32>) -> tensor<2xf32>
// LAST:     %[[V_21:.*]] = stablehlo.reshape %[[V_17]] : (tensor<2x1xf32>) -> tensor<2xf32>
// LAST:     %[[CST:.*]] = stablehlo.constant dense<4.097000e+03> : tensor<2xf32>
// LAST:     %[[V_22:.*]] = stablehlo.multiply %[[V_18]], %[[CST]] : tensor<2xf32>
// LAST:     %[[V_23:.*]] = stablehlo.subtract %[[V_22]], %[[V_18]] : tensor<2xf32>
// LAST:     %[[V_24:.*]] = stablehlo.subtract %[[V_22]], %[[V_23]] : tensor<2xf32>
// LAST:     %[[V_25:.*]] = stablehlo.subtract %[[V_18]], %[[V_24]] : tensor<2xf32>
// LAST:     %[[CST_0:.*]] = stablehlo.constant dense<4.097000e+03> : tensor<2xf32>
// LAST:     %[[V_26:.*]] = stablehlo.multiply %[[V_20]], %[[CST_0]] : tensor<2xf32>
// LAST:     %[[V_27:.*]] = stablehlo.subtract %[[V_26]], %[[V_20]] : tensor<2xf32>
// LAST:     %[[V_28:.*]] = stablehlo.subtract %[[V_26]], %[[V_27]] : tensor<2xf32>
// LAST:     %[[V_29:.*]] = stablehlo.subtract %[[V_20]], %[[V_28]] : tensor<2xf32>
// LAST:     %[[V_30:.*]] = stablehlo.dot_general %[[V_24]], %[[V_28]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// LAST:     %[[V_31:.*]] = stablehlo.dot_general %[[V_24]], %[[V_29]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// LAST:     %[[V_32:.*]] = stablehlo.dot_general %[[V_25]], %[[V_28]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// LAST:     %[[V_33:.*]] = stablehlo.dot_general %[[V_25]], %[[V_29]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// LAST:     %[[V_34:.*]] = stablehlo.dot_general %[[V_18]], %[[V_20]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// LAST:     %[[V_35:.*]] = stablehlo.subtract %[[V_30]], %[[V_34]] : tensor<f32>
// LAST:     %[[V_36:.*]] = stablehlo.add %[[V_31]], %[[V_32]] : tensor<f32>
// LAST:     %[[V_37:.*]] = stablehlo.add %[[V_35]], %[[V_36]] : tensor<f32>
// LAST:     %[[V_38:.*]] = stablehlo.add %[[V_37]], %[[V_33]] : tensor<f32>
// LAST:     %[[V_39:.*]] = stablehlo.dot_general %[[V_18]], %[[V_21]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// LAST:     %[[V_40:.*]] = stablehlo.dot_general %[[V_19]], %[[V_20]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// LAST:     %[[V_41:.*]] = stablehlo.add %[[V_38]], %[[V_39]] : tensor<f32>
// LAST:     %[[V_42:.*]] = stablehlo.add %[[V_41]], %[[V_40]] : tensor<f32>
// LAST:     %[[V_43:.*]] = stablehlo.add %[[V_34]], %[[V_42]] : tensor<f32>
// LAST:     %[[V_44:.*]] = stablehlo.subtract %[[V_43]], %[[V_34]] : tensor<f32>
// LAST:     %[[V_45:.*]] = stablehlo.subtract %[[V_42]], %[[V_44]] : tensor<f32>
// LAST:     %[[V_46:.*]] = stablehlo.reshape %[[V_43]] : (tensor<f32>) -> tensor<1xf32>
// LAST:     %[[V_47:.*]] = stablehlo.reshape %[[V_45]] : (tensor<f32>) -> tensor<1xf32>
// LAST:     %[[V_48:.*]] = stablehlo.concatenate %[[V_46]], %[[V_47]], dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
// LAST:     %[[V_49:.*]] = stablehlo.convert %[[V_48]] : (tensor<2xf32>) -> tensor<2xf64>
// LAST:     %[[CST_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// LAST:     %[[V_50:.*]] = stablehlo.reduce(%[[V_49]] init: %[[CST_1]]) applies stablehlo.add across dimensions = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<f64>
// LAST:     return %[[V_50]] : tensor<f64>

// FIRST-LABEL: func.func @main

// LAST-LABEL: func.func @main

// TUPLE-LABEL: func.func @main

func.func @main() attributes {enzyme.no_multifloat} {
  %c1 = stablehlo.constant dense<[1.10000001, 2.2]> : tensor<2xf64>
  %c2 = stablehlo.constant dense<[-1.1, 1.0]> : tensor<2xf64>
  
  %expected_mf = stablehlo.constant dense<0.989999989000001> : tensor<f64>
  
  %res = func.call @dot_general(%c1, %c2) : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
  
  // Strict test against Julia MultiFloat
  "check.expect_close"(%res, %expected_mf) {max_ulp_difference = 20 : ui64} : (tensor<f64>, tensor<f64>) -> ()
  
  // Approximate test against regular f64
  %expected_f64 = stablehlo.constant dense<0.989999989> : tensor<f64>
  "check.expect_close"(%res, %expected_f64) {max_ulp_difference = 20 : ui64} : (tensor<f64>, tensor<f64>) -> ()
  return
}
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<2xf64>
// FIRST:     %[[CST_0:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<2xf64>
// FIRST:     %[[CST_1:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<f64>
// FIRST:     %[[V_0:.*]] = call @dot_general(%[[CST]], %[[CST_0]]) : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// FIRST:     check.expect_close %[[V_0]], %[[CST_1]], max_ulp_difference = 20 : tensor<f64>, tensor<f64>
// FIRST:     %[[CST_2:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<f64>
// FIRST:     check.expect_close %[[V_0]], %[[CST_2]], max_ulp_difference = 20 : tensor<f64>, tensor<f64>
// FIRST:     return
