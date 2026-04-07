// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck --check-prefix=FIRST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=LAST %s
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @test_concat_non_fusable(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>, %arg2: tensor<2xf64>) -> (tensor<4xf64>, tensor<2xf64>) {
  %0 = stablehlo.add %arg0, %arg1 : tensor<2xf64>
  %1 = stablehlo.add %arg0, %arg2 : tensor<2xf64>
  %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<2xf64>, tensor<2xf64>) -> tensor<4xf64>
  return %2, %0 : tensor<4xf64>, tensor<2xf64>
}

// FIRST-LABEL: func.func @test_concat_non_fusable(
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
// FIRST:     %[[V_14:.*]] = stablehlo.convert %arg2 : (tensor<2xf64>) -> tensor<2xf32>
// FIRST:     %[[V_15:.*]] = stablehlo.convert %[[V_14]] : (tensor<2xf32>) -> tensor<2xf64>
// FIRST:     %[[V_16:.*]] = stablehlo.subtract %arg2, %[[V_15]] : tensor<2xf64>
// FIRST:     %[[V_17:.*]] = stablehlo.convert %[[V_16]] : (tensor<2xf64>) -> tensor<2xf32>
// FIRST:     %[[V_18:.*]] = stablehlo.reshape %[[V_14]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_19:.*]] = stablehlo.reshape %[[V_17]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_20:.*]] = stablehlo.concatenate %[[V_18]], %[[V_19]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST:     %[[V_21:.*]] = stablehlo.slice %[[V_6]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_22:.*]] = stablehlo.slice %[[V_6]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_23:.*]] = stablehlo.slice %[[V_13]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_24:.*]] = stablehlo.slice %[[V_13]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_25:.*]] = stablehlo.add %[[V_21]], %[[V_23]] : tensor<1x2xf32>
// FIRST:     %[[V_26:.*]] = stablehlo.subtract %[[V_25]], %[[V_23]] : tensor<1x2xf32>
// FIRST:     %[[V_27:.*]] = stablehlo.subtract %[[V_25]], %[[V_26]] : tensor<1x2xf32>
// FIRST:     %[[V_28:.*]] = stablehlo.subtract %[[V_21]], %[[V_26]] : tensor<1x2xf32>
// FIRST:     %[[V_29:.*]] = stablehlo.subtract %[[V_23]], %[[V_27]] : tensor<1x2xf32>
// FIRST:     %[[V_30:.*]] = stablehlo.add %[[V_28]], %[[V_29]] : tensor<1x2xf32>
// FIRST:     %[[V_31:.*]] = stablehlo.add %[[V_22]], %[[V_24]] : tensor<1x2xf32>
// FIRST:     %[[V_32:.*]] = stablehlo.subtract %[[V_31]], %[[V_24]] : tensor<1x2xf32>
// FIRST:     %[[V_33:.*]] = stablehlo.subtract %[[V_31]], %[[V_32]] : tensor<1x2xf32>
// FIRST:     %[[V_34:.*]] = stablehlo.subtract %[[V_22]], %[[V_32]] : tensor<1x2xf32>
// FIRST:     %[[V_35:.*]] = stablehlo.subtract %[[V_24]], %[[V_33]] : tensor<1x2xf32>
// FIRST:     %[[V_36:.*]] = stablehlo.add %[[V_34]], %[[V_35]] : tensor<1x2xf32>
// FIRST:     %[[V_37:.*]] = stablehlo.add %[[V_25]], %[[V_31]] : tensor<1x2xf32>
// FIRST:     %[[V_38:.*]] = stablehlo.subtract %[[V_37]], %[[V_25]] : tensor<1x2xf32>
// FIRST:     %[[V_39:.*]] = stablehlo.subtract %[[V_31]], %[[V_38]] : tensor<1x2xf32>
// FIRST:     %[[V_40:.*]] = stablehlo.add %[[V_30]], %[[V_36]] : tensor<1x2xf32>
// FIRST:     %[[V_41:.*]] = stablehlo.add %[[V_40]], %[[V_39]] : tensor<1x2xf32>
// FIRST:     %[[V_42:.*]] = stablehlo.add %[[V_37]], %[[V_41]] : tensor<1x2xf32>
// FIRST:     %[[V_43:.*]] = stablehlo.subtract %[[V_42]], %[[V_37]] : tensor<1x2xf32>
// FIRST:     %[[V_44:.*]] = stablehlo.subtract %[[V_41]], %[[V_43]] : tensor<1x2xf32>
// FIRST:     %[[V_45:.*]] = stablehlo.concatenate %[[V_42]], %[[V_44]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST:     %[[V_46:.*]] = stablehlo.slice %[[V_6]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_47:.*]] = stablehlo.slice %[[V_6]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_48:.*]] = stablehlo.slice %[[V_20]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_49:.*]] = stablehlo.slice %[[V_20]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_50:.*]] = stablehlo.add %[[V_46]], %[[V_48]] : tensor<1x2xf32>
// FIRST:     %[[V_51:.*]] = stablehlo.subtract %[[V_50]], %[[V_48]] : tensor<1x2xf32>
// FIRST:     %[[V_52:.*]] = stablehlo.subtract %[[V_50]], %[[V_51]] : tensor<1x2xf32>
// FIRST:     %[[V_53:.*]] = stablehlo.subtract %[[V_46]], %[[V_51]] : tensor<1x2xf32>
// FIRST:     %[[V_54:.*]] = stablehlo.subtract %[[V_48]], %[[V_52]] : tensor<1x2xf32>
// FIRST:     %[[V_55:.*]] = stablehlo.add %[[V_53]], %[[V_54]] : tensor<1x2xf32>
// FIRST:     %[[V_56:.*]] = stablehlo.add %[[V_47]], %[[V_49]] : tensor<1x2xf32>
// FIRST:     %[[V_57:.*]] = stablehlo.subtract %[[V_56]], %[[V_49]] : tensor<1x2xf32>
// FIRST:     %[[V_58:.*]] = stablehlo.subtract %[[V_56]], %[[V_57]] : tensor<1x2xf32>
// FIRST:     %[[V_59:.*]] = stablehlo.subtract %[[V_47]], %[[V_57]] : tensor<1x2xf32>
// FIRST:     %[[V_60:.*]] = stablehlo.subtract %[[V_49]], %[[V_58]] : tensor<1x2xf32>
// FIRST:     %[[V_61:.*]] = stablehlo.add %[[V_59]], %[[V_60]] : tensor<1x2xf32>
// FIRST:     %[[V_62:.*]] = stablehlo.add %[[V_50]], %[[V_56]] : tensor<1x2xf32>
// FIRST:     %[[V_63:.*]] = stablehlo.subtract %[[V_62]], %[[V_50]] : tensor<1x2xf32>
// FIRST:     %[[V_64:.*]] = stablehlo.subtract %[[V_56]], %[[V_63]] : tensor<1x2xf32>
// FIRST:     %[[V_65:.*]] = stablehlo.add %[[V_55]], %[[V_61]] : tensor<1x2xf32>
// FIRST:     %[[V_66:.*]] = stablehlo.add %[[V_65]], %[[V_64]] : tensor<1x2xf32>
// FIRST:     %[[V_67:.*]] = stablehlo.add %[[V_62]], %[[V_66]] : tensor<1x2xf32>
// FIRST:     %[[V_68:.*]] = stablehlo.subtract %[[V_67]], %[[V_62]] : tensor<1x2xf32>
// FIRST:     %[[V_69:.*]] = stablehlo.subtract %[[V_66]], %[[V_68]] : tensor<1x2xf32>
// FIRST:     %[[V_70:.*]] = stablehlo.concatenate %[[V_67]], %[[V_69]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST:     %[[V_71:.*]] = stablehlo.concatenate %[[V_45]], %[[V_70]], dim = 1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
// FIRST:     %[[V_72:.*]] = stablehlo.convert %[[V_71]] : (tensor<2x4xf32>) -> tensor<2x4xf64>
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// FIRST:     %[[V_73:.*]] = stablehlo.reduce(%[[V_72]] init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x4xf64>, tensor<f64>) -> tensor<4xf64>
// FIRST:     %[[V_74:.*]] = stablehlo.convert %[[V_45]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
// FIRST:     %[[V_75:.*]] = stablehlo.reduce(%[[V_74]] init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
// FIRST:     return %[[V_73]], %[[V_75]] : tensor<4xf64>, tensor<2xf64>

// LAST-LABEL: func.func @test_concat_non_fusable(
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
// LAST:     %[[V_14:.*]] = stablehlo.convert %arg2 : (tensor<2xf64>) -> tensor<2xf32>
// LAST:     %[[V_15:.*]] = stablehlo.convert %[[V_14]] : (tensor<2xf32>) -> tensor<2xf64>
// LAST:     %[[V_16:.*]] = stablehlo.subtract %arg2, %[[V_15]] : tensor<2xf64>
// LAST:     %[[V_17:.*]] = stablehlo.convert %[[V_16]] : (tensor<2xf64>) -> tensor<2xf32>
// LAST:     %[[V_18:.*]] = stablehlo.reshape %[[V_14]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_19:.*]] = stablehlo.reshape %[[V_17]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_20:.*]] = stablehlo.concatenate %[[V_18]], %[[V_19]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// LAST:     %[[V_21:.*]] = stablehlo.slice %[[V_6]] [0:2, 0:1] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_22:.*]] = stablehlo.slice %[[V_6]] [0:2, 1:2] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_23:.*]] = stablehlo.slice %[[V_13]] [0:2, 0:1] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_24:.*]] = stablehlo.slice %[[V_13]] [0:2, 1:2] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_25:.*]] = stablehlo.add %[[V_21]], %[[V_23]] : tensor<2x1xf32>
// LAST:     %[[V_26:.*]] = stablehlo.subtract %[[V_25]], %[[V_23]] : tensor<2x1xf32>
// LAST:     %[[V_27:.*]] = stablehlo.subtract %[[V_25]], %[[V_26]] : tensor<2x1xf32>
// LAST:     %[[V_28:.*]] = stablehlo.subtract %[[V_21]], %[[V_26]] : tensor<2x1xf32>
// LAST:     %[[V_29:.*]] = stablehlo.subtract %[[V_23]], %[[V_27]] : tensor<2x1xf32>
// LAST:     %[[V_30:.*]] = stablehlo.add %[[V_28]], %[[V_29]] : tensor<2x1xf32>
// LAST:     %[[V_31:.*]] = stablehlo.add %[[V_22]], %[[V_24]] : tensor<2x1xf32>
// LAST:     %[[V_32:.*]] = stablehlo.subtract %[[V_31]], %[[V_24]] : tensor<2x1xf32>
// LAST:     %[[V_33:.*]] = stablehlo.subtract %[[V_31]], %[[V_32]] : tensor<2x1xf32>
// LAST:     %[[V_34:.*]] = stablehlo.subtract %[[V_22]], %[[V_32]] : tensor<2x1xf32>
// LAST:     %[[V_35:.*]] = stablehlo.subtract %[[V_24]], %[[V_33]] : tensor<2x1xf32>
// LAST:     %[[V_36:.*]] = stablehlo.add %[[V_34]], %[[V_35]] : tensor<2x1xf32>
// LAST:     %[[V_37:.*]] = stablehlo.add %[[V_25]], %[[V_31]] : tensor<2x1xf32>
// LAST:     %[[V_38:.*]] = stablehlo.subtract %[[V_37]], %[[V_25]] : tensor<2x1xf32>
// LAST:     %[[V_39:.*]] = stablehlo.subtract %[[V_31]], %[[V_38]] : tensor<2x1xf32>
// LAST:     %[[V_40:.*]] = stablehlo.add %[[V_30]], %[[V_36]] : tensor<2x1xf32>
// LAST:     %[[V_41:.*]] = stablehlo.add %[[V_40]], %[[V_39]] : tensor<2x1xf32>
// LAST:     %[[V_42:.*]] = stablehlo.add %[[V_37]], %[[V_41]] : tensor<2x1xf32>
// LAST:     %[[V_43:.*]] = stablehlo.subtract %[[V_42]], %[[V_37]] : tensor<2x1xf32>
// LAST:     %[[V_44:.*]] = stablehlo.subtract %[[V_41]], %[[V_43]] : tensor<2x1xf32>
// LAST:     %[[V_45:.*]] = stablehlo.concatenate %[[V_42]], %[[V_44]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// LAST:     %[[V_46:.*]] = stablehlo.slice %[[V_6]] [0:2, 0:1] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_47:.*]] = stablehlo.slice %[[V_6]] [0:2, 1:2] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_48:.*]] = stablehlo.slice %[[V_20]] [0:2, 0:1] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_49:.*]] = stablehlo.slice %[[V_20]] [0:2, 1:2] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_50:.*]] = stablehlo.add %[[V_46]], %[[V_48]] : tensor<2x1xf32>
// LAST:     %[[V_51:.*]] = stablehlo.subtract %[[V_50]], %[[V_48]] : tensor<2x1xf32>
// LAST:     %[[V_52:.*]] = stablehlo.subtract %[[V_50]], %[[V_51]] : tensor<2x1xf32>
// LAST:     %[[V_53:.*]] = stablehlo.subtract %[[V_46]], %[[V_51]] : tensor<2x1xf32>
// LAST:     %[[V_54:.*]] = stablehlo.subtract %[[V_48]], %[[V_52]] : tensor<2x1xf32>
// LAST:     %[[V_55:.*]] = stablehlo.add %[[V_53]], %[[V_54]] : tensor<2x1xf32>
// LAST:     %[[V_56:.*]] = stablehlo.add %[[V_47]], %[[V_49]] : tensor<2x1xf32>
// LAST:     %[[V_57:.*]] = stablehlo.subtract %[[V_56]], %[[V_49]] : tensor<2x1xf32>
// LAST:     %[[V_58:.*]] = stablehlo.subtract %[[V_56]], %[[V_57]] : tensor<2x1xf32>
// LAST:     %[[V_59:.*]] = stablehlo.subtract %[[V_47]], %[[V_57]] : tensor<2x1xf32>
// LAST:     %[[V_60:.*]] = stablehlo.subtract %[[V_49]], %[[V_58]] : tensor<2x1xf32>
// LAST:     %[[V_61:.*]] = stablehlo.add %[[V_59]], %[[V_60]] : tensor<2x1xf32>
// LAST:     %[[V_62:.*]] = stablehlo.add %[[V_50]], %[[V_56]] : tensor<2x1xf32>
// LAST:     %[[V_63:.*]] = stablehlo.subtract %[[V_62]], %[[V_50]] : tensor<2x1xf32>
// LAST:     %[[V_64:.*]] = stablehlo.subtract %[[V_56]], %[[V_63]] : tensor<2x1xf32>
// LAST:     %[[V_65:.*]] = stablehlo.add %[[V_55]], %[[V_61]] : tensor<2x1xf32>
// LAST:     %[[V_66:.*]] = stablehlo.add %[[V_65]], %[[V_64]] : tensor<2x1xf32>
// LAST:     %[[V_67:.*]] = stablehlo.add %[[V_62]], %[[V_66]] : tensor<2x1xf32>
// LAST:     %[[V_68:.*]] = stablehlo.subtract %[[V_67]], %[[V_62]] : tensor<2x1xf32>
// LAST:     %[[V_69:.*]] = stablehlo.subtract %[[V_66]], %[[V_68]] : tensor<2x1xf32>
// LAST:     %[[V_70:.*]] = stablehlo.concatenate %[[V_67]], %[[V_69]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// LAST:     %[[V_71:.*]] = stablehlo.concatenate %[[V_45]], %[[V_70]], dim = 0 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<4x2xf32>
// LAST:     %[[V_72:.*]] = stablehlo.convert %[[V_71]] : (tensor<4x2xf32>) -> tensor<4x2xf64>
// LAST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// LAST:     %[[V_73:.*]] = stablehlo.reduce(%[[V_72]] init: %[[CST]]) applies stablehlo.add across dimensions = [1] : (tensor<4x2xf64>, tensor<f64>) -> tensor<4xf64>
// LAST:     %[[V_74:.*]] = stablehlo.convert %[[V_45]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
// LAST:     %[[V_75:.*]] = stablehlo.reduce(%[[V_74]] init: %[[CST]]) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
// LAST:     return %[[V_73]], %[[V_75]] : tensor<4xf64>, tensor<2xf64>

// FIRST-LABEL: func.func @main

// LAST-LABEL: func.func @main

func.func @main() attributes {enzyme.no_multifloat} {
  %cst1 = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf64>
  %cst2 = stablehlo.constant dense<[3.0, 4.0]> : tensor<2xf64>
  %cst3 = stablehlo.constant dense<[5.0, 6.0]> : tensor<2xf64>
  
  %exp_concat = stablehlo.constant dense<[4.0, 6.0, 6.0, 8.0]> : tensor<4xf64>
  %exp_add = stablehlo.constant dense<[4.0, 6.0]> : tensor<2xf64>
  
  %res:2 = func.call @test_concat_non_fusable(%cst1, %cst2, %cst3) : (tensor<2xf64>, tensor<2xf64>, tensor<2xf64>) -> (tensor<4xf64>, tensor<2xf64>)
  
  // Strict tests against Julia MultiFloat
  "check.expect_close"(%res#0, %exp_concat) {max_ulp_difference = 0 : ui64} : (tensor<4xf64>, tensor<4xf64>) -> ()
  "check.expect_close"(%res#1, %exp_add) {max_ulp_difference = 0 : ui64} : (tensor<2xf64>, tensor<2xf64>) -> ()
  
  // Approximate tests against regular f64
  "check.expect_close"(%res#0, %exp_concat) {max_ulp_difference = 0 : ui64} : (tensor<4xf64>, tensor<4xf64>) -> ()
  "check.expect_close"(%res#1, %exp_add) {max_ulp_difference = 0 : ui64} : (tensor<2xf64>, tensor<2xf64>) -> ()
  return
}
// FIRST:     %[[CST_MAIN:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<2xf64>
// FIRST:     %[[CST_MAIN_0:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<2xf64>
// FIRST:     %[[CST_MAIN_1:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<2xf64>
// FIRST:     %[[CST_MAIN_2:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<4xf64>
// FIRST:     %[[CST_MAIN_3:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<2xf64>
// FIRST:     %[[V_MAIN_0:.*]]:2 = call @test_concat_non_fusable(%[[CST_MAIN]], %[[CST_MAIN_0]], %[[CST_MAIN_1]]) : (tensor<2xf64>, tensor<2xf64>, tensor<2xf64>) -> (tensor<4xf64>, tensor<2xf64>)
// FIRST:     check.expect_close %[[V_MAIN_0]]#0, %[[CST_MAIN_2]], max_ulp_difference = 0 : tensor<4xf64>, tensor<4xf64>
// FIRST:     check.expect_close %[[V_MAIN_0]]#1, %[[CST_MAIN_3]], max_ulp_difference = 0 : tensor<2xf64>, tensor<2xf64>
// FIRST:     check.expect_close %[[V_MAIN_0]]#0, %[[CST_MAIN_2]], max_ulp_difference = 0 : tensor<4xf64>, tensor<4xf64>
// FIRST:     check.expect_close %[[V_MAIN_0]]#1, %[[CST_MAIN_3]], max_ulp_difference = 0 : tensor<2xf64>, tensor<2xf64>
// FIRST:     return
