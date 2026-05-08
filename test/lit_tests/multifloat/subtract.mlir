// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck --check-prefix=FIRST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=TUPLE %s
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @test_sub(%arg0: tensor<5xf64>, %arg1: tensor<5xf64>) -> tensor<5xf64> {
  %0 = stablehlo.subtract %arg0, %arg1 : tensor<5xf64>
  return %0 : tensor<5xf64>
}

// TUPLE-LABEL: func.func @test_sub(
// TUPLE:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<5xf64>) -> tensor<5xf32>
// TUPLE:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<5xf32>) -> tensor<5xf64>
// TUPLE:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<5xf64>
// TUPLE:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<5xf64>) -> tensor<5xf32>
// TUPLE:     %[[V_4:.*]] = stablehlo.tuple %[[V_0]], %[[V_3]] : tuple<tensor<5xf32>, tensor<5xf32>>
// TUPLE:     %[[V_5:.*]] = stablehlo.convert %arg1 : (tensor<5xf64>) -> tensor<5xf32>
// TUPLE:     %[[V_6:.*]] = stablehlo.convert %[[V_5]] : (tensor<5xf32>) -> tensor<5xf64>
// TUPLE:     %[[V_7:.*]] = stablehlo.subtract %arg1, %[[V_6]] : tensor<5xf64>
// TUPLE:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<5xf64>) -> tensor<5xf32>
// TUPLE:     %[[V_9:.*]] = stablehlo.tuple %[[V_5]], %[[V_8]] : tuple<tensor<5xf32>, tensor<5xf32>>
// TUPLE:     %[[V_10:.*]] = stablehlo.get_tuple_element %[[V_4]][0] : (tuple<tensor<5xf32>, tensor<5xf32>>) -> tensor<5xf32>
// TUPLE:     %[[V_11:.*]] = stablehlo.get_tuple_element %[[V_4]][1] : (tuple<tensor<5xf32>, tensor<5xf32>>) -> tensor<5xf32>
// TUPLE:     %[[V_12:.*]] = stablehlo.get_tuple_element %[[V_9]][0] : (tuple<tensor<5xf32>, tensor<5xf32>>) -> tensor<5xf32>
// TUPLE:     %[[V_13:.*]] = stablehlo.get_tuple_element %[[V_9]][1] : (tuple<tensor<5xf32>, tensor<5xf32>>) -> tensor<5xf32>
// TUPLE:     %[[V_14:.*]] = stablehlo.negate %[[V_12]] : tensor<5xf32>
// TUPLE:     %[[V_15:.*]] = stablehlo.negate %[[V_13]] : tensor<5xf32>
// TUPLE:     %[[V_16:.*]] = stablehlo.add %[[V_10]], %[[V_14]] : tensor<5xf32>
// TUPLE:     %[[V_17:.*]] = stablehlo.subtract %[[V_16]], %[[V_14]] : tensor<5xf32>
// TUPLE:     %[[V_18:.*]] = stablehlo.subtract %[[V_16]], %[[V_17]] : tensor<5xf32>
// TUPLE:     %[[V_19:.*]] = stablehlo.subtract %[[V_10]], %[[V_17]] : tensor<5xf32>
// TUPLE:     %[[V_20:.*]] = stablehlo.subtract %[[V_14]], %[[V_18]] : tensor<5xf32>
// TUPLE:     %[[V_21:.*]] = stablehlo.add %[[V_19]], %[[V_20]] : tensor<5xf32>
// TUPLE:     %[[V_22:.*]] = stablehlo.add %[[V_11]], %[[V_15]] : tensor<5xf32>
// TUPLE:     %[[V_23:.*]] = stablehlo.subtract %[[V_22]], %[[V_15]] : tensor<5xf32>
// TUPLE:     %[[V_24:.*]] = stablehlo.subtract %[[V_22]], %[[V_23]] : tensor<5xf32>
// TUPLE:     %[[V_25:.*]] = stablehlo.subtract %[[V_11]], %[[V_23]] : tensor<5xf32>
// TUPLE:     %[[V_26:.*]] = stablehlo.subtract %[[V_15]], %[[V_24]] : tensor<5xf32>
// TUPLE:     %[[V_27:.*]] = stablehlo.add %[[V_25]], %[[V_26]] : tensor<5xf32>
// TUPLE:     %[[V_28:.*]] = stablehlo.add %[[V_16]], %[[V_22]] : tensor<5xf32>
// TUPLE:     %[[V_29:.*]] = stablehlo.subtract %[[V_28]], %[[V_16]] : tensor<5xf32>
// TUPLE:     %[[V_30:.*]] = stablehlo.subtract %[[V_22]], %[[V_29]] : tensor<5xf32>
// TUPLE:     %[[V_31:.*]] = stablehlo.add %[[V_21]], %[[V_27]] : tensor<5xf32>
// TUPLE:     %[[V_32:.*]] = stablehlo.add %[[V_31]], %[[V_30]] : tensor<5xf32>
// TUPLE:     %[[V_33:.*]] = stablehlo.add %[[V_28]], %[[V_32]] : tensor<5xf32>
// TUPLE:     %[[V_34:.*]] = stablehlo.subtract %[[V_33]], %[[V_28]] : tensor<5xf32>
// TUPLE:     %[[V_35:.*]] = stablehlo.subtract %[[V_32]], %[[V_34]] : tensor<5xf32>
// TUPLE:     %[[V_36:.*]] = stablehlo.tuple %[[V_33]], %[[V_35]] : tuple<tensor<5xf32>, tensor<5xf32>>
// TUPLE:     %[[V_37:.*]] = stablehlo.convert %[[V_33]] : (tensor<5xf32>) -> tensor<5xf64>
// TUPLE:     %[[V_38:.*]] = stablehlo.convert %[[V_35]] : (tensor<5xf32>) -> tensor<5xf64>
// TUPLE:     %[[V_39:.*]] = stablehlo.add %[[V_37]], %[[V_38]] : tensor<5xf64>
// TUPLE:     return %[[V_39]] : tensor<5xf64>

// FIRST-LABEL: func.func @test_sub(
// FIRST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<5xf64>) -> tensor<5xf32>
// FIRST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<5xf32>) -> tensor<5xf64>
// FIRST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<5xf64>
// FIRST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<5xf64>) -> tensor<5xf32>
// FIRST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<5xf32>) -> tensor<1x5xf32>
// FIRST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<5xf32>) -> tensor<1x5xf32>
// FIRST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 0 : (tensor<1x5xf32>, tensor<1x5xf32>) -> tensor<2x5xf32>
// FIRST:     %[[V_7:.*]] = stablehlo.convert %arg1 : (tensor<5xf64>) -> tensor<5xf32>
// FIRST:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<5xf32>) -> tensor<5xf64>
// FIRST:     %[[V_9:.*]] = stablehlo.subtract %arg1, %[[V_8]] : tensor<5xf64>
// FIRST:     %[[V_10:.*]] = stablehlo.convert %[[V_9]] : (tensor<5xf64>) -> tensor<5xf32>
// FIRST:     %[[V_11:.*]] = stablehlo.reshape %[[V_7]] : (tensor<5xf32>) -> tensor<1x5xf32>
// FIRST:     %[[V_12:.*]] = stablehlo.reshape %[[V_10]] : (tensor<5xf32>) -> tensor<1x5xf32>
// FIRST:     %[[V_13:.*]] = stablehlo.concatenate %[[V_11]], %[[V_12]], dim = 0 : (tensor<1x5xf32>, tensor<1x5xf32>) -> tensor<2x5xf32>
// FIRST:     %[[V_14:.*]] = stablehlo.slice %[[V_6]] [0:1, 0:5] : (tensor<2x5xf32>) -> tensor<1x5xf32>
// FIRST:     %[[V_15:.*]] = stablehlo.slice %[[V_6]] [1:2, 0:5] : (tensor<2x5xf32>) -> tensor<1x5xf32>
// FIRST:     %[[V_16:.*]] = stablehlo.slice %[[V_13]] [0:1, 0:5] : (tensor<2x5xf32>) -> tensor<1x5xf32>
// FIRST:     %[[V_17:.*]] = stablehlo.slice %[[V_13]] [1:2, 0:5] : (tensor<2x5xf32>) -> tensor<1x5xf32>
// FIRST:     %[[V_18:.*]] = stablehlo.negate %[[V_16]] : tensor<1x5xf32>
// FIRST:     %[[V_19:.*]] = stablehlo.negate %[[V_17]] : tensor<1x5xf32>
// FIRST:     %[[V_20:.*]] = stablehlo.add %[[V_14]], %[[V_18]] : tensor<1x5xf32>
// FIRST:     %[[V_21:.*]] = stablehlo.subtract %[[V_20]], %[[V_18]] : tensor<1x5xf32>
// FIRST:     %[[V_22:.*]] = stablehlo.subtract %[[V_20]], %[[V_21]] : tensor<1x5xf32>
// FIRST:     %[[V_23:.*]] = stablehlo.subtract %[[V_14]], %[[V_21]] : tensor<1x5xf32>
// FIRST:     %[[V_24:.*]] = stablehlo.subtract %[[V_18]], %[[V_22]] : tensor<1x5xf32>
// FIRST:     %[[V_25:.*]] = stablehlo.add %[[V_23]], %[[V_24]] : tensor<1x5xf32>
// FIRST:     %[[V_26:.*]] = stablehlo.add %[[V_15]], %[[V_19]] : tensor<1x5xf32>
// FIRST:     %[[V_27:.*]] = stablehlo.subtract %[[V_26]], %[[V_19]] : tensor<1x5xf32>
// FIRST:     %[[V_28:.*]] = stablehlo.subtract %[[V_26]], %[[V_27]] : tensor<1x5xf32>
// FIRST:     %[[V_29:.*]] = stablehlo.subtract %[[V_15]], %[[V_27]] : tensor<1x5xf32>
// FIRST:     %[[V_30:.*]] = stablehlo.subtract %[[V_19]], %[[V_28]] : tensor<1x5xf32>
// FIRST:     %[[V_31:.*]] = stablehlo.add %[[V_29]], %[[V_30]] : tensor<1x5xf32>
// FIRST:     %[[V_32:.*]] = stablehlo.add %[[V_20]], %[[V_26]] : tensor<1x5xf32>
// FIRST:     %[[V_33:.*]] = stablehlo.subtract %[[V_32]], %[[V_20]] : tensor<1x5xf32>
// FIRST:     %[[V_34:.*]] = stablehlo.subtract %[[V_26]], %[[V_33]] : tensor<1x5xf32>
// FIRST:     %[[V_35:.*]] = stablehlo.add %[[V_25]], %[[V_31]] : tensor<1x5xf32>
// FIRST:     %[[V_36:.*]] = stablehlo.add %[[V_35]], %[[V_34]] : tensor<1x5xf32>
// FIRST:     %[[V_37:.*]] = stablehlo.add %[[V_32]], %[[V_36]] : tensor<1x5xf32>
// FIRST:     %[[V_38:.*]] = stablehlo.subtract %[[V_37]], %[[V_32]] : tensor<1x5xf32>
// FIRST:     %[[V_39:.*]] = stablehlo.subtract %[[V_36]], %[[V_38]] : tensor<1x5xf32>
// FIRST:     %[[V_40:.*]] = stablehlo.concatenate %[[V_37]], %[[V_39]], dim = 0 : (tensor<1x5xf32>, tensor<1x5xf32>) -> tensor<2x5xf32>
// FIRST:     %[[V_41:.*]] = stablehlo.convert %[[V_40]] : (tensor<2x5xf32>) -> tensor<2x5xf64>
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// FIRST:     %[[V_42:.*]] = stablehlo.reduce(%[[V_41]] init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x5xf64>, tensor<f64>) -> tensor<5xf64>
// FIRST:     return %[[V_42]] : tensor<5xf64>

// LAST-LABEL: func.func @test_sub(
// LAST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<5xf64>) -> tensor<5xf32>
// LAST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<5xf32>) -> tensor<5xf64>
// LAST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<5xf64>
// LAST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<5xf64>) -> tensor<5xf32>
// LAST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<5xf32>) -> tensor<5x1xf32>
// LAST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<5xf32>) -> tensor<5x1xf32>
// LAST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 1 : (tensor<5x1xf32>, tensor<5x1xf32>) -> tensor<5x2xf32>
// LAST:     %[[V_7:.*]] = stablehlo.convert %arg1 : (tensor<5xf64>) -> tensor<5xf32>
// LAST:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<5xf32>) -> tensor<5xf64>
// LAST:     %[[V_9:.*]] = stablehlo.subtract %arg1, %[[V_8]] : tensor<5xf64>
// LAST:     %[[V_10:.*]] = stablehlo.convert %[[V_9]] : (tensor<5xf64>) -> tensor<5xf32>
// LAST:     %[[V_11:.*]] = stablehlo.reshape %[[V_7]] : (tensor<5xf32>) -> tensor<5x1xf32>
// LAST:     %[[V_12:.*]] = stablehlo.reshape %[[V_10]] : (tensor<5xf32>) -> tensor<5x1xf32>
// LAST:     %[[V_13:.*]] = stablehlo.concatenate %[[V_11]], %[[V_12]], dim = 1 : (tensor<5x1xf32>, tensor<5x1xf32>) -> tensor<5x2xf32>
// LAST:     %[[V_14:.*]] = stablehlo.slice %[[V_6]] [0:5, 0:1] : (tensor<5x2xf32>) -> tensor<5x1xf32>
// LAST:     %[[V_15:.*]] = stablehlo.slice %[[V_6]] [0:5, 1:2] : (tensor<5x2xf32>) -> tensor<5x1xf32>
// LAST:     %[[V_16:.*]] = stablehlo.slice %[[V_13]] [0:5, 0:1] : (tensor<5x2xf32>) -> tensor<5x1xf32>
// LAST:     %[[V_17:.*]] = stablehlo.slice %[[V_13]] [0:5, 1:2] : (tensor<5x2xf32>) -> tensor<5x1xf32>
// LAST:     %[[V_18:.*]] = stablehlo.negate %[[V_16]] : tensor<5x1xf32>
// LAST:     %[[V_19:.*]] = stablehlo.negate %[[V_17]] : tensor<5x1xf32>
// LAST:     %[[V_20:.*]] = stablehlo.add %[[V_14]], %[[V_18]] : tensor<5x1xf32>
// LAST:     %[[V_21:.*]] = stablehlo.subtract %[[V_20]], %[[V_18]] : tensor<5x1xf32>
// LAST:     %[[V_22:.*]] = stablehlo.subtract %[[V_20]], %[[V_21]] : tensor<5x1xf32>
// LAST:     %[[V_23:.*]] = stablehlo.subtract %[[V_14]], %[[V_21]] : tensor<5x1xf32>
// LAST:     %[[V_24:.*]] = stablehlo.subtract %[[V_18]], %[[V_22]] : tensor<5x1xf32>
// LAST:     %[[V_25:.*]] = stablehlo.add %[[V_23]], %[[V_24]] : tensor<5x1xf32>
// LAST:     %[[V_26:.*]] = stablehlo.add %[[V_15]], %[[V_19]] : tensor<5x1xf32>
// LAST:     %[[V_27:.*]] = stablehlo.subtract %[[V_26]], %[[V_19]] : tensor<5x1xf32>
// LAST:     %[[V_28:.*]] = stablehlo.subtract %[[V_26]], %[[V_27]] : tensor<5x1xf32>
// LAST:     %[[V_29:.*]] = stablehlo.subtract %[[V_15]], %[[V_27]] : tensor<5x1xf32>
// LAST:     %[[V_30:.*]] = stablehlo.subtract %[[V_19]], %[[V_28]] : tensor<5x1xf32>
// LAST:     %[[V_31:.*]] = stablehlo.add %[[V_29]], %[[V_30]] : tensor<5x1xf32>
// LAST:     %[[V_32:.*]] = stablehlo.add %[[V_20]], %[[V_26]] : tensor<5x1xf32>
// LAST:     %[[V_33:.*]] = stablehlo.subtract %[[V_32]], %[[V_20]] : tensor<5x1xf32>
// LAST:     %[[V_34:.*]] = stablehlo.subtract %[[V_26]], %[[V_33]] : tensor<5x1xf32>
// LAST:     %[[V_35:.*]] = stablehlo.add %[[V_25]], %[[V_31]] : tensor<5x1xf32>
// LAST:     %[[V_36:.*]] = stablehlo.add %[[V_35]], %[[V_34]] : tensor<5x1xf32>
// LAST:     %[[V_37:.*]] = stablehlo.add %[[V_32]], %[[V_36]] : tensor<5x1xf32>
// LAST:     %[[V_38:.*]] = stablehlo.subtract %[[V_37]], %[[V_32]] : tensor<5x1xf32>
// LAST:     %[[V_39:.*]] = stablehlo.subtract %[[V_36]], %[[V_38]] : tensor<5x1xf32>
// LAST:     %[[V_40:.*]] = stablehlo.concatenate %[[V_37]], %[[V_39]], dim = 1 : (tensor<5x1xf32>, tensor<5x1xf32>) -> tensor<5x2xf32>
// LAST:     %[[V_41:.*]] = stablehlo.convert %[[V_40]] : (tensor<5x2xf32>) -> tensor<5x2xf64>
// LAST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// LAST:     %[[V_42:.*]] = stablehlo.reduce(%[[V_41]] init: %[[CST]]) applies stablehlo.add across dimensions = [1] : (tensor<5x2xf64>, tensor<f64>) -> tensor<5xf64>
// LAST:     return %[[V_42]] : tensor<5xf64>

// FIRST-LABEL: func.func @main

// LAST-LABEL: func.func @main

// TUPLE-LABEL: func.func @main

func.func @main() attributes {enzyme.no_multifloat} {
  %c1 = stablehlo.constant dense<[1.1, 1.1, -1.1, -1.1, 1.10000001]> : tensor<5xf64>
  %c2 = stablehlo.constant dense<[1.1, -1.1, 1.1, -1.1, 1.1]> : tensor<5xf64>
  
  %expected_mf = stablehlo.constant dense<[0.0, 2.1999999999999993, -2.1999999999999993, 0.0, 1.000000082740371e-08]> : tensor<5xf64>
  
  %res = func.call @test_sub(%c1, %c2) : (tensor<5xf64>, tensor<5xf64>) -> tensor<5xf64>
  
  // Strict test against Julia MultiFloat
  "check.expect_close"(%res, %expected_mf) {max_ulp_difference = 3 : ui64} : (tensor<5xf64>, tensor<5xf64>) -> ()
  
  // Approximate test against regular f64
  %expected_f64 = stablehlo.constant dense<[0.0, 2.2, -2.2, 0.0, 9.99999993922529e-09]> : tensor<5xf64>
  %diff = stablehlo.subtract %res, %expected_f64 : tensor<5xf64>
  %abs_diff = stablehlo.abs %diff : tensor<5xf64>
  %zero = stablehlo.constant dense<0.0> : tensor<5xf64>
  "check.expect_almost_eq"(%abs_diff, %zero) {tolerance = 1.0e-12 : f64} : (tensor<5xf64>, tensor<5xf64>) -> ()
  return
}
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<5xf64>
// FIRST:     %[[CST_0:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<5xf64>
// FIRST:     %[[CST_1:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<5xf64>
// FIRST:     %[[V_0:.*]] = call @test_sub(%[[CST]], %[[CST_0]]) : (tensor<5xf64>, tensor<5xf64>) -> tensor<5xf64>
// FIRST:     check.expect_close %[[V_0]], %[[CST_1]], max_ulp_difference = 3 : tensor<5xf64>, tensor<5xf64>
// FIRST:     %[[CST_2:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<5xf64>
// FIRST:     %[[V_1:.*]] = stablehlo.subtract %[[V_0]], %[[CST_2]] : tensor<5xf64>
// FIRST:     %[[V_2:.*]] = stablehlo.abs %[[V_1]] : tensor<5xf64>
// FIRST:     %[[CST_3:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<5xf64>
// FIRST:     check.expect_almost_eq %[[V_2]], %[[CST_3]], tolerance = 9.9999999999999998E-13 : tensor<5xf64>
// FIRST:     return
