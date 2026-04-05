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
// TUPLE:     %[[V_18:.*]] = stablehlo.dot_general %[[V_14]], %[[V_16]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// TUPLE:     %[[V_19:.*]] = stablehlo.dot_general %[[V_14]], %[[V_17]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// TUPLE:     %[[V_20:.*]] = stablehlo.dot_general %[[V_15]], %[[V_16]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// TUPLE:     %[[V_21:.*]] = stablehlo.dot_general %[[V_15]], %[[V_17]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// TUPLE:     %[[V_22:.*]] = stablehlo.add %[[V_19]], %[[V_20]] : tensor<f32>
// TUPLE:     %[[V_23:.*]] = stablehlo.add %[[V_22]], %[[V_21]] : tensor<f32>
// TUPLE:     %[[V_24:.*]] = stablehlo.add %[[V_18]], %[[V_23]] : tensor<f32>
// TUPLE:     %[[V_25:.*]] = stablehlo.subtract %[[V_24]], %[[V_18]] : tensor<f32>
// TUPLE:     %[[V_26:.*]] = stablehlo.subtract %[[V_23]], %[[V_25]] : tensor<f32>
// TUPLE:     %[[V_27:.*]] = stablehlo.tuple %[[V_24]], %[[V_26]] : tuple<tensor<f32>, tensor<f32>>
// TUPLE:     %[[V_28:.*]] = stablehlo.convert %[[V_24]] : (tensor<f32>) -> tensor<f64>
// TUPLE:     %[[V_29:.*]] = stablehlo.convert %[[V_26]] : (tensor<f32>) -> tensor<f64>
// TUPLE:     %[[V_30:.*]] = stablehlo.add %[[V_28]], %[[V_29]] : tensor<f64>
// TUPLE:     return %[[V_30]] : tensor<f64>

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
// FIRST:     %[[V_22:.*]] = stablehlo.dot_general %[[V_18]], %[[V_20]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// FIRST:     %[[V_23:.*]] = stablehlo.dot_general %[[V_18]], %[[V_21]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// FIRST:     %[[V_24:.*]] = stablehlo.dot_general %[[V_19]], %[[V_20]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// FIRST:     %[[V_25:.*]] = stablehlo.dot_general %[[V_19]], %[[V_21]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// FIRST:     %[[V_26:.*]] = stablehlo.add %[[V_23]], %[[V_24]] : tensor<f32>
// FIRST:     %[[V_27:.*]] = stablehlo.add %[[V_26]], %[[V_25]] : tensor<f32>
// FIRST:     %[[V_28:.*]] = stablehlo.add %[[V_22]], %[[V_27]] : tensor<f32>
// FIRST:     %[[V_29:.*]] = stablehlo.subtract %[[V_28]], %[[V_22]] : tensor<f32>
// FIRST:     %[[V_30:.*]] = stablehlo.subtract %[[V_27]], %[[V_29]] : tensor<f32>
// FIRST:     %[[V_31:.*]] = stablehlo.reshape %[[V_28]] : (tensor<f32>) -> tensor<1xf32>
// FIRST:     %[[V_32:.*]] = stablehlo.reshape %[[V_30]] : (tensor<f32>) -> tensor<1xf32>
// FIRST:     %[[V_33:.*]] = stablehlo.concatenate %[[V_31]], %[[V_32]], dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
// FIRST:     %[[V_34:.*]] = stablehlo.convert %[[V_33]] : (tensor<2xf32>) -> tensor<2xf64>
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// FIRST:     %[[V_35:.*]] = stablehlo.reduce(%[[V_34]] init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<f64>
// FIRST:     return %[[V_35]] : tensor<f64>

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
// LAST:     %[[V_22:.*]] = stablehlo.dot_general %[[V_18]], %[[V_20]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// LAST:     %[[V_23:.*]] = stablehlo.dot_general %[[V_18]], %[[V_21]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// LAST:     %[[V_24:.*]] = stablehlo.dot_general %[[V_19]], %[[V_20]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// LAST:     %[[V_25:.*]] = stablehlo.dot_general %[[V_19]], %[[V_21]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
// LAST:     %[[V_26:.*]] = stablehlo.add %[[V_23]], %[[V_24]] : tensor<f32>
// LAST:     %[[V_27:.*]] = stablehlo.add %[[V_26]], %[[V_25]] : tensor<f32>
// LAST:     %[[V_28:.*]] = stablehlo.add %[[V_22]], %[[V_27]] : tensor<f32>
// LAST:     %[[V_29:.*]] = stablehlo.subtract %[[V_28]], %[[V_22]] : tensor<f32>
// LAST:     %[[V_30:.*]] = stablehlo.subtract %[[V_27]], %[[V_29]] : tensor<f32>
// LAST:     %[[V_31:.*]] = stablehlo.reshape %[[V_28]] : (tensor<f32>) -> tensor<1xf32>
// LAST:     %[[V_32:.*]] = stablehlo.reshape %[[V_30]] : (tensor<f32>) -> tensor<1xf32>
// LAST:     %[[V_33:.*]] = stablehlo.concatenate %[[V_31]], %[[V_32]], dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
// LAST:     %[[V_34:.*]] = stablehlo.convert %[[V_33]] : (tensor<2xf32>) -> tensor<2xf64>
// LAST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// LAST:     %[[V_35:.*]] = stablehlo.reduce(%[[V_34]] init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<f64>
// LAST:     return %[[V_35]] : tensor<f64>

// FIRST-LABEL: func.func @main

// LAST-LABEL: func.func @main

// TUPLE-LABEL: func.func @main

func.func @main() attributes {enzyme.no_multifloat} {
  %c1 = stablehlo.constant dense<[1.10000001, 2.2]> : tensor<2xf64>
  %c2 = stablehlo.constant dense<[-1.1, 1.0]> : tensor<2xf64>
  
  %expected_mf = stablehlo.constant dense<0.989999989000001> : tensor<f64>
  
  %res = func.call @dot_general(%c1, %c2) : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
  
  // Strict test against Julia MultiFloat
  "check.expect_close"(%res, %expected_mf) {max_ulp_difference = 3 : ui64} : (tensor<f64>, tensor<f64>) -> ()
  
  // Approximate test against regular f64
  %expected_f64 = stablehlo.constant dense<0.989999989> : tensor<f64>
  "check.expect_close"(%res, %expected_f64) {max_ulp_difference = 10 : ui64} : (tensor<f64>, tensor<f64>) -> ()
  return
}
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<2xf64>
// FIRST:     %[[CST_0:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<2xf64>
// FIRST:     %[[CST_1:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<f64>
// FIRST:     %[[V_0:.*]] = call @dot_general(%[[CST]], %[[CST_0]]) : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// FIRST:     check.expect_close %[[V_0]], %[[CST_1]], max_ulp_difference = 3 : tensor<f64>, tensor<f64>
// FIRST:     %[[CST_2:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<f64>
// FIRST:     check.expect_close %[[V_0]], %[[CST_2]], max_ulp_difference = 10 : tensor<f64>, tensor<f64>
// FIRST:     return
