// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck --check-prefix=FIRST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=TUPLE %s
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @while(%arg0: tensor<f64>) -> tensor<f64> {
  %cst = stablehlo.constant dense<1.000000e+01> : tensor<f64>
  %0:2 = stablehlo.while(%iterArg0 = %arg0, %iterArg1 = %cst) : tensor<f64>, tensor<f64>
    cond {
      %1 = stablehlo.compare LT, %iterArg0, %iterArg1 : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg0, %iterArg0 : tensor<f64>
      stablehlo.return %1, %iterArg1 : tensor<f64>, tensor<f64>
    }
  return %0#0 : tensor<f64>
}

// TUPLE-LABEL: func.func @while
// TUPLE:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<f64>) -> tensor<f32>
// TUPLE:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<f32>) -> tensor<f64>
// TUPLE:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<f64>
// TUPLE:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<f64>) -> tensor<f32>
// TUPLE:     %[[V_4:.*]] = stablehlo.tuple %[[V_0]], %[[V_3]] : tuple<tensor<f32>, tensor<f32>>
// TUPLE:     %[[CST:.*]] = stablehlo.constant dense<1.000000e+01> : tensor<f32>
// TUPLE:     %[[CST_0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// TUPLE:     %[[V_5:.*]] = stablehlo.tuple %[[CST]], %[[CST_0]] : tuple<tensor<f32>, tensor<f32>>
// TUPLE:     %[[V_6:.*]] = stablehlo.get_tuple_element %[[V_4]][0] : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
// TUPLE:     %[[V_7:.*]] = stablehlo.get_tuple_element %[[V_4]][1] : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
// TUPLE:     %[[V_8:.*]] = stablehlo.get_tuple_element %[[V_5]][0] : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
// TUPLE:     %[[V_9:.*]] = stablehlo.get_tuple_element %[[V_5]][1] : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
// TUPLE:     %[[V_10:.*]]:4 = stablehlo.while(%iterArg = %[[V_6]], %iterArg_1 = %[[V_7]], %iterArg_2 = %[[V_8]], %iterArg_3 = %[[V_9]]) : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
// TUPLE:     cond {
// TUPLE:       %[[V_16:.*]] = stablehlo.compare LT, %iterArg, %iterArg_2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
// TUPLE:       %[[V_17:.*]] = stablehlo.compare EQ, %iterArg, %iterArg_2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
// TUPLE:       %[[V_18:.*]] = stablehlo.compare LT, %iterArg_1, %iterArg_3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
// TUPLE:       %[[V_19:.*]] = stablehlo.select %[[V_17]], %[[V_18]], %[[V_16]] : tensor<i1>, tensor<i1>
// TUPLE:       stablehlo.return %[[V_19]] : tensor<i1>
// TUPLE:       %[[V_16:.*]] = stablehlo.add %iterArg, %iterArg : tensor<f32>
// TUPLE:       %[[V_17:.*]] = stablehlo.subtract %[[V_16]], %iterArg : tensor<f32>
// TUPLE:       %[[V_18:.*]] = stablehlo.subtract %[[V_16]], %[[V_17]] : tensor<f32>
// TUPLE:       %[[V_19:.*]] = stablehlo.subtract %iterArg, %[[V_17]] : tensor<f32>
// TUPLE:       %[[V_20:.*]] = stablehlo.subtract %iterArg, %[[V_18]] : tensor<f32>
// TUPLE:       %[[V_21:.*]] = stablehlo.add %[[V_19]], %[[V_20]] : tensor<f32>
// TUPLE:       %[[V_22:.*]] = stablehlo.add %iterArg_1, %iterArg_1 : tensor<f32>
// TUPLE:       %[[V_23:.*]] = stablehlo.subtract %[[V_22]], %iterArg_1 : tensor<f32>
// TUPLE:       %[[V_24:.*]] = stablehlo.subtract %[[V_22]], %[[V_23]] : tensor<f32>
// TUPLE:       %[[V_25:.*]] = stablehlo.subtract %iterArg_1, %[[V_23]] : tensor<f32>
// TUPLE:       %[[V_26:.*]] = stablehlo.subtract %iterArg_1, %[[V_24]] : tensor<f32>
// TUPLE:       %[[V_27:.*]] = stablehlo.add %[[V_25]], %[[V_26]] : tensor<f32>
// TUPLE:       %[[V_28:.*]] = stablehlo.add %[[V_16]], %[[V_22]] : tensor<f32>
// TUPLE:       %[[V_29:.*]] = stablehlo.subtract %[[V_28]], %[[V_16]] : tensor<f32>
// TUPLE:       %[[V_30:.*]] = stablehlo.subtract %[[V_22]], %[[V_29]] : tensor<f32>
// TUPLE:       %[[V_31:.*]] = stablehlo.add %[[V_21]], %[[V_27]] : tensor<f32>
// TUPLE:       %[[V_32:.*]] = stablehlo.add %[[V_31]], %[[V_30]] : tensor<f32>
// TUPLE:       %[[V_33:.*]] = stablehlo.add %[[V_28]], %[[V_32]] : tensor<f32>
// TUPLE:       %[[V_34:.*]] = stablehlo.subtract %[[V_33]], %[[V_28]] : tensor<f32>
// TUPLE:       %[[V_35:.*]] = stablehlo.subtract %[[V_32]], %[[V_34]] : tensor<f32>
// TUPLE:       %[[V_36:.*]] = stablehlo.tuple %[[V_33]], %[[V_35]] : tuple<tensor<f32>, tensor<f32>>
// TUPLE:       stablehlo.return %[[V_33]], %[[V_35]], %iterArg_2, %iterArg_3 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
// TUPLE:     %[[V_11:.*]] = stablehlo.tuple %[[V_10]]#0, %[[V_10]]#1 : tuple<tensor<f32>, tensor<f32>>
// TUPLE:     %[[V_12:.*]] = stablehlo.tuple %[[V_10]]#2, %[[V_10]]#3 : tuple<tensor<f32>, tensor<f32>>
// TUPLE:     %[[V_13:.*]] = stablehlo.convert %[[V_10]]#0 : (tensor<f32>) -> tensor<f64>
// TUPLE:     %[[V_14:.*]] = stablehlo.convert %[[V_10]]#1 : (tensor<f32>) -> tensor<f64>
// TUPLE:     %[[V_15:.*]] = stablehlo.add %[[V_13]], %[[V_14]] : tensor<f64>
// TUPLE:     return %[[V_15]] : tensor<f64>

// FIRST-LABEL: func.func @while
// FIRST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<f64>) -> tensor<f32>
// FIRST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<f32>) -> tensor<f64>
// FIRST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<f64>
// FIRST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<f64>) -> tensor<f32>
// FIRST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<f32>) -> tensor<1xf32>
// FIRST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<f32>) -> tensor<1xf32>
// FIRST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<1.000000e+01> : tensor<1xf32>
// FIRST:     %[[CST_0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<1xf32>
// FIRST:     %[[V_7:.*]] = stablehlo.concatenate %[[CST]], %[[CST_0]], dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
// FIRST:     %[[V_8:.*]]:2 = stablehlo.while(%iterArg = %[[V_6]], %iterArg_2 = %[[V_7]]) : tensor<2xf32>, tensor<2xf32>
// FIRST:     cond {
// FIRST:       %[[V_11:.*]] = stablehlo.slice %iterArg [0:1] : (tensor<2xf32>) -> tensor<1xf32>
// FIRST:       %[[V_12:.*]] = stablehlo.slice %iterArg [1:2] : (tensor<2xf32>) -> tensor<1xf32>
// FIRST:       %[[V_13:.*]] = stablehlo.slice %iterArg_2 [0:1] : (tensor<2xf32>) -> tensor<1xf32>
// FIRST:       %[[V_14:.*]] = stablehlo.slice %iterArg_2 [1:2] : (tensor<2xf32>) -> tensor<1xf32>
// FIRST:       %[[V_15:.*]] = stablehlo.compare LT, %[[V_11]], %[[V_13]] : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
// FIRST:       %[[V_16:.*]] = stablehlo.compare EQ, %[[V_11]], %[[V_13]] : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
// FIRST:       %[[V_17:.*]] = stablehlo.compare LT, %[[V_12]], %[[V_14]] : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
// FIRST:       %[[V_18:.*]] = stablehlo.select %[[V_16]], %[[V_17]], %[[V_15]] : tensor<1xi1>, tensor<1xi1>
// FIRST:       %[[V_19:.*]] = stablehlo.reshape %[[V_18]] : (tensor<1xi1>) -> tensor<i1>
// FIRST:       stablehlo.return %[[V_19]] : tensor<i1>
// FIRST:       %[[V_11:.*]] = stablehlo.slice %iterArg [0:1] : (tensor<2xf32>) -> tensor<1xf32>
// FIRST:       %[[V_12:.*]] = stablehlo.slice %iterArg [1:2] : (tensor<2xf32>) -> tensor<1xf32>
// FIRST:       %[[V_13:.*]] = stablehlo.slice %iterArg [0:1] : (tensor<2xf32>) -> tensor<1xf32>
// FIRST:       %[[V_14:.*]] = stablehlo.slice %iterArg [1:2] : (tensor<2xf32>) -> tensor<1xf32>
// FIRST:       %[[V_15:.*]] = stablehlo.add %[[V_11]], %[[V_13]] : tensor<1xf32>
// FIRST:       %[[V_16:.*]] = stablehlo.subtract %[[V_15]], %[[V_13]] : tensor<1xf32>
// FIRST:       %[[V_17:.*]] = stablehlo.subtract %[[V_15]], %[[V_16]] : tensor<1xf32>
// FIRST:       %[[V_18:.*]] = stablehlo.subtract %[[V_11]], %[[V_16]] : tensor<1xf32>
// FIRST:       %[[V_19:.*]] = stablehlo.subtract %[[V_13]], %[[V_17]] : tensor<1xf32>
// FIRST:       %[[V_20:.*]] = stablehlo.add %[[V_18]], %[[V_19]] : tensor<1xf32>
// FIRST:       %[[V_21:.*]] = stablehlo.add %[[V_12]], %[[V_14]] : tensor<1xf32>
// FIRST:       %[[V_22:.*]] = stablehlo.subtract %[[V_21]], %[[V_14]] : tensor<1xf32>
// FIRST:       %[[V_23:.*]] = stablehlo.subtract %[[V_21]], %[[V_22]] : tensor<1xf32>
// FIRST:       %[[V_24:.*]] = stablehlo.subtract %[[V_12]], %[[V_22]] : tensor<1xf32>
// FIRST:       %[[V_25:.*]] = stablehlo.subtract %[[V_14]], %[[V_23]] : tensor<1xf32>
// FIRST:       %[[V_26:.*]] = stablehlo.add %[[V_24]], %[[V_25]] : tensor<1xf32>
// FIRST:       %[[V_27:.*]] = stablehlo.add %[[V_15]], %[[V_21]] : tensor<1xf32>
// FIRST:       %[[V_28:.*]] = stablehlo.subtract %[[V_27]], %[[V_15]] : tensor<1xf32>
// FIRST:       %[[V_29:.*]] = stablehlo.subtract %[[V_21]], %[[V_28]] : tensor<1xf32>
// FIRST:       %[[V_30:.*]] = stablehlo.add %[[V_20]], %[[V_26]] : tensor<1xf32>
// FIRST:       %[[V_31:.*]] = stablehlo.add %[[V_30]], %[[V_29]] : tensor<1xf32>
// FIRST:       %[[V_32:.*]] = stablehlo.add %[[V_27]], %[[V_31]] : tensor<1xf32>
// FIRST:       %[[V_33:.*]] = stablehlo.subtract %[[V_32]], %[[V_27]] : tensor<1xf32>
// FIRST:       %[[V_34:.*]] = stablehlo.subtract %[[V_31]], %[[V_33]] : tensor<1xf32>
// FIRST:       %[[V_35:.*]] = stablehlo.concatenate %[[V_32]], %[[V_34]], dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
// FIRST:       stablehlo.return %[[V_35]], %iterArg_2 : tensor<2xf32>, tensor<2xf32>
// FIRST:     %[[V_9:.*]] = stablehlo.convert %[[V_8]]#0 : (tensor<2xf32>) -> tensor<2xf64>
// FIRST:     %[[CST_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// FIRST:     %[[V_10:.*]] = stablehlo.reduce(%[[V_9]] init: %[[CST_1]]) applies stablehlo.add across dimensions = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<f64>
// FIRST:     return %[[V_10]] : tensor<f64>

// LAST-LABEL: func.func @while
// LAST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<f64>) -> tensor<f32>
// LAST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<f32>) -> tensor<f64>
// LAST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<f64>
// LAST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<f64>) -> tensor<f32>
// LAST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<f32>) -> tensor<1xf32>
// LAST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<f32>) -> tensor<1xf32>
// LAST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
// LAST:     %[[CST:.*]] = stablehlo.constant dense<1.000000e+01> : tensor<1xf32>
// LAST:     %[[CST_0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<1xf32>
// LAST:     %[[V_7:.*]] = stablehlo.concatenate %[[CST]], %[[CST_0]], dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
// LAST:     %[[V_8:.*]]:2 = stablehlo.while(%iterArg = %[[V_6]], %iterArg_2 = %[[V_7]]) : tensor<2xf32>, tensor<2xf32>
// LAST:     cond {
// LAST:       %[[V_11:.*]] = stablehlo.slice %iterArg [0:1] : (tensor<2xf32>) -> tensor<1xf32>
// LAST:       %[[V_12:.*]] = stablehlo.slice %iterArg [1:2] : (tensor<2xf32>) -> tensor<1xf32>
// LAST:       %[[V_13:.*]] = stablehlo.slice %iterArg_2 [0:1] : (tensor<2xf32>) -> tensor<1xf32>
// LAST:       %[[V_14:.*]] = stablehlo.slice %iterArg_2 [1:2] : (tensor<2xf32>) -> tensor<1xf32>
// LAST:       %[[V_15:.*]] = stablehlo.compare LT, %[[V_11]], %[[V_13]] : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
// LAST:       %[[V_16:.*]] = stablehlo.compare EQ, %[[V_11]], %[[V_13]] : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
// LAST:       %[[V_17:.*]] = stablehlo.compare LT, %[[V_12]], %[[V_14]] : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
// LAST:       %[[V_18:.*]] = stablehlo.select %[[V_16]], %[[V_17]], %[[V_15]] : tensor<1xi1>, tensor<1xi1>
// LAST:       %[[V_19:.*]] = stablehlo.reshape %[[V_18]] : (tensor<1xi1>) -> tensor<i1>
// LAST:       stablehlo.return %[[V_19]] : tensor<i1>
// LAST:       %[[V_11:.*]] = stablehlo.slice %iterArg [0:1] : (tensor<2xf32>) -> tensor<1xf32>
// LAST:       %[[V_12:.*]] = stablehlo.slice %iterArg [1:2] : (tensor<2xf32>) -> tensor<1xf32>
// LAST:       %[[V_13:.*]] = stablehlo.slice %iterArg [0:1] : (tensor<2xf32>) -> tensor<1xf32>
// LAST:       %[[V_14:.*]] = stablehlo.slice %iterArg [1:2] : (tensor<2xf32>) -> tensor<1xf32>
// LAST:       %[[V_15:.*]] = stablehlo.add %[[V_11]], %[[V_13]] : tensor<1xf32>
// LAST:       %[[V_16:.*]] = stablehlo.subtract %[[V_15]], %[[V_13]] : tensor<1xf32>
// LAST:       %[[V_17:.*]] = stablehlo.subtract %[[V_15]], %[[V_16]] : tensor<1xf32>
// LAST:       %[[V_18:.*]] = stablehlo.subtract %[[V_11]], %[[V_16]] : tensor<1xf32>
// LAST:       %[[V_19:.*]] = stablehlo.subtract %[[V_13]], %[[V_17]] : tensor<1xf32>
// LAST:       %[[V_20:.*]] = stablehlo.add %[[V_18]], %[[V_19]] : tensor<1xf32>
// LAST:       %[[V_21:.*]] = stablehlo.add %[[V_12]], %[[V_14]] : tensor<1xf32>
// LAST:       %[[V_22:.*]] = stablehlo.subtract %[[V_21]], %[[V_14]] : tensor<1xf32>
// LAST:       %[[V_23:.*]] = stablehlo.subtract %[[V_21]], %[[V_22]] : tensor<1xf32>
// LAST:       %[[V_24:.*]] = stablehlo.subtract %[[V_12]], %[[V_22]] : tensor<1xf32>
// LAST:       %[[V_25:.*]] = stablehlo.subtract %[[V_14]], %[[V_23]] : tensor<1xf32>
// LAST:       %[[V_26:.*]] = stablehlo.add %[[V_24]], %[[V_25]] : tensor<1xf32>
// LAST:       %[[V_27:.*]] = stablehlo.add %[[V_15]], %[[V_21]] : tensor<1xf32>
// LAST:       %[[V_28:.*]] = stablehlo.subtract %[[V_27]], %[[V_15]] : tensor<1xf32>
// LAST:       %[[V_29:.*]] = stablehlo.subtract %[[V_21]], %[[V_28]] : tensor<1xf32>
// LAST:       %[[V_30:.*]] = stablehlo.add %[[V_20]], %[[V_26]] : tensor<1xf32>
// LAST:       %[[V_31:.*]] = stablehlo.add %[[V_30]], %[[V_29]] : tensor<1xf32>
// LAST:       %[[V_32:.*]] = stablehlo.add %[[V_27]], %[[V_31]] : tensor<1xf32>
// LAST:       %[[V_33:.*]] = stablehlo.subtract %[[V_32]], %[[V_27]] : tensor<1xf32>
// LAST:       %[[V_34:.*]] = stablehlo.subtract %[[V_31]], %[[V_33]] : tensor<1xf32>
// LAST:       %[[V_35:.*]] = stablehlo.concatenate %[[V_32]], %[[V_34]], dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
// LAST:       stablehlo.return %[[V_35]], %iterArg_2 : tensor<2xf32>, tensor<2xf32>
// LAST:     %[[V_9:.*]] = stablehlo.convert %[[V_8]]#0 : (tensor<2xf32>) -> tensor<2xf64>
// LAST:     %[[CST_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// LAST:     %[[V_10:.*]] = stablehlo.reduce(%[[V_9]] init: %[[CST_1]]) applies stablehlo.add across dimensions = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<f64>
// LAST:     return %[[V_10]] : tensor<f64>

// FIRST-LABEL: func.func @main

// LAST-LABEL: func.func @main

// TUPLE-LABEL: func.func @main

func.func @main() attributes {enzyme.no_multifloat} {
  %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
  
  %expected = stablehlo.constant dense<1.600000e+01> : tensor<f64>
  
  %res = func.call @while(%cst) : (tensor<f64>) -> tensor<f64>
  
  // Strict test against Julia MultiFloat
  "check.expect_close"(%res, %expected) {max_ulp_difference = 0 : ui64} : (tensor<f64>, tensor<f64>) -> ()
  
  // Approximate test against regular f64
  "check.expect_close"(%res, %expected) {max_ulp_difference = 0 : ui64} : (tensor<f64>, tensor<f64>) -> ()
  return
}
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<f64>
// FIRST:     %[[CST_0:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<f64>
// FIRST:     %[[V_0:.*]] = call @while(%[[CST]]) : (tensor<f64>) -> tensor<f64>
// FIRST:     check.expect_close %[[V_0]], %[[CST_0]], max_ulp_difference = 0 : tensor<f64>, tensor<f64>
// FIRST:     check.expect_close %[[V_0]], %[[CST_0]], max_ulp_difference = 0 : tensor<f64>, tensor<f64>
// FIRST:     return
