// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck --check-prefix=FIRST %s
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @test_concat_fusable(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>) -> tensor<4xf64> {
  // FIRST-LABEL: func.func @test_concat_fusable
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
// FIRST:     %[[V_18:.*]] = stablehlo.add %[[V_14]], %[[V_16]] : tensor<1x2xf32>
// FIRST:     %[[V_19:.*]] = stablehlo.subtract %[[V_18]], %[[V_16]] : tensor<1x2xf32>
// FIRST:     %[[V_20:.*]] = stablehlo.subtract %[[V_18]], %[[V_19]] : tensor<1x2xf32>
// FIRST:     %[[V_21:.*]] = stablehlo.subtract %[[V_14]], %[[V_19]] : tensor<1x2xf32>
// FIRST:     %[[V_22:.*]] = stablehlo.subtract %[[V_16]], %[[V_20]] : tensor<1x2xf32>
// FIRST:     %[[V_23:.*]] = stablehlo.add %[[V_21]], %[[V_22]] : tensor<1x2xf32>
// FIRST:     %[[V_24:.*]] = stablehlo.add %[[V_15]], %[[V_17]] : tensor<1x2xf32>
// FIRST:     %[[V_25:.*]] = stablehlo.subtract %[[V_24]], %[[V_17]] : tensor<1x2xf32>
// FIRST:     %[[V_26:.*]] = stablehlo.subtract %[[V_24]], %[[V_25]] : tensor<1x2xf32>
// FIRST:     %[[V_27:.*]] = stablehlo.subtract %[[V_15]], %[[V_25]] : tensor<1x2xf32>
// FIRST:     %[[V_28:.*]] = stablehlo.subtract %[[V_17]], %[[V_26]] : tensor<1x2xf32>
// FIRST:     %[[V_29:.*]] = stablehlo.add %[[V_27]], %[[V_28]] : tensor<1x2xf32>
// FIRST:     %[[V_30:.*]] = stablehlo.add %[[V_18]], %[[V_24]] : tensor<1x2xf32>
// FIRST:     %[[V_31:.*]] = stablehlo.subtract %[[V_30]], %[[V_18]] : tensor<1x2xf32>
// FIRST:     %[[V_32:.*]] = stablehlo.subtract %[[V_24]], %[[V_31]] : tensor<1x2xf32>
// FIRST:     %[[V_33:.*]] = stablehlo.add %[[V_23]], %[[V_29]] : tensor<1x2xf32>
// FIRST:     %[[V_34:.*]] = stablehlo.add %[[V_33]], %[[V_32]] : tensor<1x2xf32>
// FIRST:     %[[V_35:.*]] = stablehlo.add %[[V_30]], %[[V_34]] : tensor<1x2xf32>
// FIRST:     %[[V_36:.*]] = stablehlo.subtract %[[V_35]], %[[V_30]] : tensor<1x2xf32>
// FIRST:     %[[V_37:.*]] = stablehlo.subtract %[[V_34]], %[[V_36]] : tensor<1x2xf32>
// FIRST:     %[[V_38:.*]] = stablehlo.concatenate %[[V_35]], %[[V_37]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST:     %[[V_39:.*]] = stablehlo.slice %[[V_6]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_40:.*]] = stablehlo.slice %[[V_6]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_41:.*]] = stablehlo.slice %[[V_13]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_42:.*]] = stablehlo.slice %[[V_13]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_43:.*]] = stablehlo.add %[[V_39]], %[[V_41]] : tensor<1x2xf32>
// FIRST:     %[[V_44:.*]] = stablehlo.subtract %[[V_43]], %[[V_41]] : tensor<1x2xf32>
// FIRST:     %[[V_45:.*]] = stablehlo.subtract %[[V_43]], %[[V_44]] : tensor<1x2xf32>
// FIRST:     %[[V_46:.*]] = stablehlo.subtract %[[V_39]], %[[V_44]] : tensor<1x2xf32>
// FIRST:     %[[V_47:.*]] = stablehlo.subtract %[[V_41]], %[[V_45]] : tensor<1x2xf32>
// FIRST:     %[[V_48:.*]] = stablehlo.add %[[V_46]], %[[V_47]] : tensor<1x2xf32>
// FIRST:     %[[V_49:.*]] = stablehlo.add %[[V_40]], %[[V_42]] : tensor<1x2xf32>
// FIRST:     %[[V_50:.*]] = stablehlo.subtract %[[V_49]], %[[V_42]] : tensor<1x2xf32>
// FIRST:     %[[V_51:.*]] = stablehlo.subtract %[[V_49]], %[[V_50]] : tensor<1x2xf32>
// FIRST:     %[[V_52:.*]] = stablehlo.subtract %[[V_40]], %[[V_50]] : tensor<1x2xf32>
// FIRST:     %[[V_53:.*]] = stablehlo.subtract %[[V_42]], %[[V_51]] : tensor<1x2xf32>
// FIRST:     %[[V_54:.*]] = stablehlo.add %[[V_52]], %[[V_53]] : tensor<1x2xf32>
// FIRST:     %[[V_55:.*]] = stablehlo.add %[[V_43]], %[[V_49]] : tensor<1x2xf32>
// FIRST:     %[[V_56:.*]] = stablehlo.subtract %[[V_55]], %[[V_43]] : tensor<1x2xf32>
// FIRST:     %[[V_57:.*]] = stablehlo.subtract %[[V_49]], %[[V_56]] : tensor<1x2xf32>
// FIRST:     %[[V_58:.*]] = stablehlo.add %[[V_48]], %[[V_54]] : tensor<1x2xf32>
// FIRST:     %[[V_59:.*]] = stablehlo.add %[[V_58]], %[[V_57]] : tensor<1x2xf32>
// FIRST:     %[[V_60:.*]] = stablehlo.add %[[V_55]], %[[V_59]] : tensor<1x2xf32>
// FIRST:     %[[V_61:.*]] = stablehlo.subtract %[[V_60]], %[[V_55]] : tensor<1x2xf32>
// FIRST:     %[[V_62:.*]] = stablehlo.subtract %[[V_59]], %[[V_61]] : tensor<1x2xf32>
// FIRST:     %[[V_63:.*]] = stablehlo.concatenate %[[V_60]], %[[V_62]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST:     %[[V_64:.*]] = stablehlo.concatenate %[[V_6]], %[[V_6]], dim = 1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
// FIRST:     %[[V_65:.*]] = stablehlo.concatenate %[[V_13]], %[[V_13]], dim = 1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
// FIRST:     %[[V_66:.*]] = stablehlo.slice %[[V_64]] [0:1, 0:4] : (tensor<2x4xf32>) -> tensor<1x4xf32>
// FIRST:     %[[V_67:.*]] = stablehlo.slice %[[V_64]] [1:2, 0:4] : (tensor<2x4xf32>) -> tensor<1x4xf32>
// FIRST:     %[[V_68:.*]] = stablehlo.slice %[[V_65]] [0:1, 0:4] : (tensor<2x4xf32>) -> tensor<1x4xf32>
// FIRST:     %[[V_69:.*]] = stablehlo.slice %[[V_65]] [1:2, 0:4] : (tensor<2x4xf32>) -> tensor<1x4xf32>
// FIRST:     %[[V_70:.*]] = stablehlo.add %[[V_66]], %[[V_68]] : tensor<1x4xf32>
// FIRST:     %[[V_71:.*]] = stablehlo.subtract %[[V_70]], %[[V_68]] : tensor<1x4xf32>
// FIRST:     %[[V_72:.*]] = stablehlo.subtract %[[V_70]], %[[V_71]] : tensor<1x4xf32>
// FIRST:     %[[V_73:.*]] = stablehlo.subtract %[[V_66]], %[[V_71]] : tensor<1x4xf32>
// FIRST:     %[[V_74:.*]] = stablehlo.subtract %[[V_68]], %[[V_72]] : tensor<1x4xf32>
// FIRST:     %[[V_75:.*]] = stablehlo.add %[[V_73]], %[[V_74]] : tensor<1x4xf32>
// FIRST:     %[[V_76:.*]] = stablehlo.add %[[V_67]], %[[V_69]] : tensor<1x4xf32>
// FIRST:     %[[V_77:.*]] = stablehlo.subtract %[[V_76]], %[[V_69]] : tensor<1x4xf32>
// FIRST:     %[[V_78:.*]] = stablehlo.subtract %[[V_76]], %[[V_77]] : tensor<1x4xf32>
// FIRST:     %[[V_79:.*]] = stablehlo.subtract %[[V_67]], %[[V_77]] : tensor<1x4xf32>
// FIRST:     %[[V_80:.*]] = stablehlo.subtract %[[V_69]], %[[V_78]] : tensor<1x4xf32>
// FIRST:     %[[V_81:.*]] = stablehlo.add %[[V_79]], %[[V_80]] : tensor<1x4xf32>
// FIRST:     %[[V_82:.*]] = stablehlo.add %[[V_70]], %[[V_76]] : tensor<1x4xf32>
// FIRST:     %[[V_83:.*]] = stablehlo.subtract %[[V_82]], %[[V_70]] : tensor<1x4xf32>
// FIRST:     %[[V_84:.*]] = stablehlo.subtract %[[V_76]], %[[V_83]] : tensor<1x4xf32>
// FIRST:     %[[V_85:.*]] = stablehlo.add %[[V_75]], %[[V_81]] : tensor<1x4xf32>
// FIRST:     %[[V_86:.*]] = stablehlo.add %[[V_85]], %[[V_84]] : tensor<1x4xf32>
// FIRST:     %[[V_87:.*]] = stablehlo.add %[[V_82]], %[[V_86]] : tensor<1x4xf32>
// FIRST:     %[[V_88:.*]] = stablehlo.subtract %[[V_87]], %[[V_82]] : tensor<1x4xf32>
// FIRST:     %[[V_89:.*]] = stablehlo.subtract %[[V_86]], %[[V_88]] : tensor<1x4xf32>
// FIRST:     %[[V_90:.*]] = stablehlo.concatenate %[[V_87]], %[[V_89]], dim = 0 : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<2x4xf32>
// FIRST:     %[[V_91:.*]] = stablehlo.convert %[[V_90]] : (tensor<2x4xf32>) -> tensor<2x4xf64>
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// FIRST:     %[[V_92:.*]] = stablehlo.reduce(%[[V_91]] init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x4xf64>, tensor<f64>) -> tensor<4xf64>
// FIRST:     return %[[V_92]] : tensor<4xf64>
  %0 = stablehlo.add %arg0, %arg1 : tensor<2xf64>
  %1 = stablehlo.add %arg0, %arg1 : tensor<2xf64>
  %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<2xf64>, tensor<2xf64>) -> tensor<4xf64>
  return %2 : tensor<4xf64>
}

func.func @main() attributes {enzyme.no_multifloat} {
  %cst1 = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf64>
  %cst2 = stablehlo.constant dense<[3.0, 4.0]> : tensor<2xf64>
  
  %expected = stablehlo.constant dense<[4.0, 6.0, 4.0, 6.0]> : tensor<4xf64>
  
  %res = func.call @test_concat_fusable(%cst1, %cst2) : (tensor<2xf64>, tensor<2xf64>) -> tensor<4xf64>
  
  // Strict test against Julia MultiFloat
  "check.expect_close"(%res, %expected) {max_ulp_difference = 0 : ui64} : (tensor<4xf64>, tensor<4xf64>) -> ()
  
  // Approximate test against regular f64
  "check.expect_close"(%res, %expected) {max_ulp_difference = 0 : ui64} : (tensor<4xf64>, tensor<4xf64>) -> ()
  return
}

// FIRST-LABEL: func.func @main
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
// FIRST:     %[[CST_0:.*]] = stablehlo.constant dense<[3.000000e+00, 4.000000e+00]> : tensor<2xf64>
// FIRST:     %[[CST_1:.*]] = stablehlo.constant dense<[4.000000e+00, 6.000000e+00, 4.000000e+00, 6.000000e+00]> : tensor<4xf64>
// FIRST:     %[[V_0:.*]] = call @test_concat_fusable(%[[CST]], %[[CST_0]]) : (tensor<2xf64>, tensor<2xf64>) -> tensor<4xf64>
// FIRST:     check.expect_close %[[V_0]], %[[CST_1]], max_ulp_difference = 0 : tensor<4xf64>, tensor<4xf64>
// FIRST:     check.expect_close %[[V_0]], %[[CST_1]], max_ulp_difference = 0 : tensor<4xf64>, tensor<4xf64>
// FIRST:     return
