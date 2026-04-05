// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32" %s | FileCheck %s

func.func @test_combine_add(%arg0: tensor<1xf64>, %arg1: tensor<1xf64>, %arg2: tensor<1xf64>, %arg3: tensor<1xf64>) -> tensor<2xf64> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<1xf64>
  %1 = stablehlo.add %arg2, %arg3 : tensor<1xf64>
  %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
  return %2 : tensor<2xf64>
}

// CHECK-LABEL: func.func @test_combine_add
// CHECK:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<1xf64>) -> tensor<1xf32>
// CHECK:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<1xf32>) -> tensor<1xf64>
// CHECK:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<1xf64>
// CHECK:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<1xf64>) -> tensor<1xf32>
// CHECK:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<1xf32>) -> tensor<1x1xf32>
// CHECK:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<1xf32>) -> tensor<1x1xf32>
// CHECK:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<2x1xf32>
// CHECK:     %[[V_7:.*]] = stablehlo.convert %arg1 : (tensor<1xf64>) -> tensor<1xf32>
// CHECK:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<1xf32>) -> tensor<1xf64>
// CHECK:     %[[V_9:.*]] = stablehlo.subtract %arg1, %[[V_8]] : tensor<1xf64>
// CHECK:     %[[V_10:.*]] = stablehlo.convert %[[V_9]] : (tensor<1xf64>) -> tensor<1xf32>
// CHECK:     %[[V_11:.*]] = stablehlo.reshape %[[V_7]] : (tensor<1xf32>) -> tensor<1x1xf32>
// CHECK:     %[[V_12:.*]] = stablehlo.reshape %[[V_10]] : (tensor<1xf32>) -> tensor<1x1xf32>
// CHECK:     %[[V_13:.*]] = stablehlo.concatenate %[[V_11]], %[[V_12]], dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<2x1xf32>
// CHECK:     %[[V_14:.*]] = stablehlo.convert %arg2 : (tensor<1xf64>) -> tensor<1xf32>
// CHECK:     %[[V_15:.*]] = stablehlo.convert %[[V_14]] : (tensor<1xf32>) -> tensor<1xf64>
// CHECK:     %[[V_16:.*]] = stablehlo.subtract %arg2, %[[V_15]] : tensor<1xf64>
// CHECK:     %[[V_17:.*]] = stablehlo.convert %[[V_16]] : (tensor<1xf64>) -> tensor<1xf32>
// CHECK:     %[[V_18:.*]] = stablehlo.reshape %[[V_14]] : (tensor<1xf32>) -> tensor<1x1xf32>
// CHECK:     %[[V_19:.*]] = stablehlo.reshape %[[V_17]] : (tensor<1xf32>) -> tensor<1x1xf32>
// CHECK:     %[[V_20:.*]] = stablehlo.concatenate %[[V_18]], %[[V_19]], dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<2x1xf32>
// CHECK:     %[[V_21:.*]] = stablehlo.convert %arg3 : (tensor<1xf64>) -> tensor<1xf32>
// CHECK:     %[[V_22:.*]] = stablehlo.convert %[[V_21]] : (tensor<1xf32>) -> tensor<1xf64>
// CHECK:     %[[V_23:.*]] = stablehlo.subtract %arg3, %[[V_22]] : tensor<1xf64>
// CHECK:     %[[V_24:.*]] = stablehlo.convert %[[V_23]] : (tensor<1xf64>) -> tensor<1xf32>
// CHECK:     %[[V_25:.*]] = stablehlo.reshape %[[V_21]] : (tensor<1xf32>) -> tensor<1x1xf32>
// CHECK:     %[[V_26:.*]] = stablehlo.reshape %[[V_24]] : (tensor<1xf32>) -> tensor<1x1xf32>
// CHECK:     %[[V_27:.*]] = stablehlo.concatenate %[[V_25]], %[[V_26]], dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<2x1xf32>
// CHECK:     %[[V_28:.*]] = stablehlo.slice %[[V_6]] [0:1, 0:1] : (tensor<2x1xf32>) -> tensor<1x1xf32>
// CHECK:     %[[V_29:.*]] = stablehlo.slice %[[V_6]] [1:2, 0:1] : (tensor<2x1xf32>) -> tensor<1x1xf32>
// CHECK:     %[[V_30:.*]] = stablehlo.slice %[[V_13]] [0:1, 0:1] : (tensor<2x1xf32>) -> tensor<1x1xf32>
// CHECK:     %[[V_31:.*]] = stablehlo.slice %[[V_13]] [1:2, 0:1] : (tensor<2x1xf32>) -> tensor<1x1xf32>
// CHECK:     %[[V_32:.*]] = stablehlo.add %[[V_28]], %[[V_30]] : tensor<1x1xf32>
// CHECK:     %[[V_33:.*]] = stablehlo.subtract %[[V_32]], %[[V_30]] : tensor<1x1xf32>
// CHECK:     %[[V_34:.*]] = stablehlo.subtract %[[V_32]], %[[V_33]] : tensor<1x1xf32>
// CHECK:     %[[V_35:.*]] = stablehlo.subtract %[[V_28]], %[[V_33]] : tensor<1x1xf32>
// CHECK:     %[[V_36:.*]] = stablehlo.subtract %[[V_30]], %[[V_34]] : tensor<1x1xf32>
// CHECK:     %[[V_37:.*]] = stablehlo.add %[[V_35]], %[[V_36]] : tensor<1x1xf32>
// CHECK:     %[[V_38:.*]] = stablehlo.add %[[V_29]], %[[V_31]] : tensor<1x1xf32>
// CHECK:     %[[V_39:.*]] = stablehlo.subtract %[[V_38]], %[[V_31]] : tensor<1x1xf32>
// CHECK:     %[[V_40:.*]] = stablehlo.subtract %[[V_38]], %[[V_39]] : tensor<1x1xf32>
// CHECK:     %[[V_41:.*]] = stablehlo.subtract %[[V_29]], %[[V_39]] : tensor<1x1xf32>
// CHECK:     %[[V_42:.*]] = stablehlo.subtract %[[V_31]], %[[V_40]] : tensor<1x1xf32>
// CHECK:     %[[V_43:.*]] = stablehlo.add %[[V_41]], %[[V_42]] : tensor<1x1xf32>
// CHECK:     %[[V_44:.*]] = stablehlo.add %[[V_32]], %[[V_38]] : tensor<1x1xf32>
// CHECK:     %[[V_45:.*]] = stablehlo.subtract %[[V_44]], %[[V_32]] : tensor<1x1xf32>
// CHECK:     %[[V_46:.*]] = stablehlo.subtract %[[V_38]], %[[V_45]] : tensor<1x1xf32>
// CHECK:     %[[V_47:.*]] = stablehlo.add %[[V_37]], %[[V_43]] : tensor<1x1xf32>
// CHECK:     %[[V_48:.*]] = stablehlo.add %[[V_47]], %[[V_46]] : tensor<1x1xf32>
// CHECK:     %[[V_49:.*]] = stablehlo.add %[[V_44]], %[[V_48]] : tensor<1x1xf32>
// CHECK:     %[[V_50:.*]] = stablehlo.subtract %[[V_49]], %[[V_44]] : tensor<1x1xf32>
// CHECK:     %[[V_51:.*]] = stablehlo.subtract %[[V_48]], %[[V_50]] : tensor<1x1xf32>
// CHECK:     %[[V_52:.*]] = stablehlo.concatenate %[[V_49]], %[[V_51]], dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<2x1xf32>
// CHECK:     %[[V_53:.*]] = stablehlo.slice %[[V_20]] [0:1, 0:1] : (tensor<2x1xf32>) -> tensor<1x1xf32>
// CHECK:     %[[V_54:.*]] = stablehlo.slice %[[V_20]] [1:2, 0:1] : (tensor<2x1xf32>) -> tensor<1x1xf32>
// CHECK:     %[[V_55:.*]] = stablehlo.slice %[[V_27]] [0:1, 0:1] : (tensor<2x1xf32>) -> tensor<1x1xf32>
// CHECK:     %[[V_56:.*]] = stablehlo.slice %[[V_27]] [1:2, 0:1] : (tensor<2x1xf32>) -> tensor<1x1xf32>
// CHECK:     %[[V_57:.*]] = stablehlo.add %[[V_53]], %[[V_55]] : tensor<1x1xf32>
// CHECK:     %[[V_58:.*]] = stablehlo.subtract %[[V_57]], %[[V_55]] : tensor<1x1xf32>
// CHECK:     %[[V_59:.*]] = stablehlo.subtract %[[V_57]], %[[V_58]] : tensor<1x1xf32>
// CHECK:     %[[V_60:.*]] = stablehlo.subtract %[[V_53]], %[[V_58]] : tensor<1x1xf32>
// CHECK:     %[[V_61:.*]] = stablehlo.subtract %[[V_55]], %[[V_59]] : tensor<1x1xf32>
// CHECK:     %[[V_62:.*]] = stablehlo.add %[[V_60]], %[[V_61]] : tensor<1x1xf32>
// CHECK:     %[[V_63:.*]] = stablehlo.add %[[V_54]], %[[V_56]] : tensor<1x1xf32>
// CHECK:     %[[V_64:.*]] = stablehlo.subtract %[[V_63]], %[[V_56]] : tensor<1x1xf32>
// CHECK:     %[[V_65:.*]] = stablehlo.subtract %[[V_63]], %[[V_64]] : tensor<1x1xf32>
// CHECK:     %[[V_66:.*]] = stablehlo.subtract %[[V_54]], %[[V_64]] : tensor<1x1xf32>
// CHECK:     %[[V_67:.*]] = stablehlo.subtract %[[V_56]], %[[V_65]] : tensor<1x1xf32>
// CHECK:     %[[V_68:.*]] = stablehlo.add %[[V_66]], %[[V_67]] : tensor<1x1xf32>
// CHECK:     %[[V_69:.*]] = stablehlo.add %[[V_57]], %[[V_63]] : tensor<1x1xf32>
// CHECK:     %[[V_70:.*]] = stablehlo.subtract %[[V_69]], %[[V_57]] : tensor<1x1xf32>
// CHECK:     %[[V_71:.*]] = stablehlo.subtract %[[V_63]], %[[V_70]] : tensor<1x1xf32>
// CHECK:     %[[V_72:.*]] = stablehlo.add %[[V_62]], %[[V_68]] : tensor<1x1xf32>
// CHECK:     %[[V_73:.*]] = stablehlo.add %[[V_72]], %[[V_71]] : tensor<1x1xf32>
// CHECK:     %[[V_74:.*]] = stablehlo.add %[[V_69]], %[[V_73]] : tensor<1x1xf32>
// CHECK:     %[[V_75:.*]] = stablehlo.subtract %[[V_74]], %[[V_69]] : tensor<1x1xf32>
// CHECK:     %[[V_76:.*]] = stablehlo.subtract %[[V_73]], %[[V_75]] : tensor<1x1xf32>
// CHECK:     %[[V_77:.*]] = stablehlo.concatenate %[[V_74]], %[[V_76]], dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<2x1xf32>
// CHECK:     %[[V_78:.*]] = stablehlo.concatenate %[[V_6]], %[[V_20]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// CHECK:     %[[V_79:.*]] = stablehlo.concatenate %[[V_13]], %[[V_27]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// CHECK:     %[[V_80:.*]] = stablehlo.slice %[[V_78]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// CHECK:     %[[V_81:.*]] = stablehlo.slice %[[V_78]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// CHECK:     %[[V_82:.*]] = stablehlo.slice %[[V_79]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// CHECK:     %[[V_83:.*]] = stablehlo.slice %[[V_79]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// CHECK:     %[[V_84:.*]] = stablehlo.add %[[V_80]], %[[V_82]] : tensor<1x2xf32>
// CHECK:     %[[V_85:.*]] = stablehlo.subtract %[[V_84]], %[[V_82]] : tensor<1x2xf32>
// CHECK:     %[[V_86:.*]] = stablehlo.subtract %[[V_84]], %[[V_85]] : tensor<1x2xf32>
// CHECK:     %[[V_87:.*]] = stablehlo.subtract %[[V_80]], %[[V_85]] : tensor<1x2xf32>
// CHECK:     %[[V_88:.*]] = stablehlo.subtract %[[V_82]], %[[V_86]] : tensor<1x2xf32>
// CHECK:     %[[V_89:.*]] = stablehlo.add %[[V_87]], %[[V_88]] : tensor<1x2xf32>
// CHECK:     %[[V_90:.*]] = stablehlo.add %[[V_81]], %[[V_83]] : tensor<1x2xf32>
// CHECK:     %[[V_91:.*]] = stablehlo.subtract %[[V_90]], %[[V_83]] : tensor<1x2xf32>
// CHECK:     %[[V_92:.*]] = stablehlo.subtract %[[V_90]], %[[V_91]] : tensor<1x2xf32>
// CHECK:     %[[V_93:.*]] = stablehlo.subtract %[[V_81]], %[[V_91]] : tensor<1x2xf32>
// CHECK:     %[[V_94:.*]] = stablehlo.subtract %[[V_83]], %[[V_92]] : tensor<1x2xf32>
// CHECK:     %[[V_95:.*]] = stablehlo.add %[[V_93]], %[[V_94]] : tensor<1x2xf32>
// CHECK:     %[[V_96:.*]] = stablehlo.add %[[V_84]], %[[V_90]] : tensor<1x2xf32>
// CHECK:     %[[V_97:.*]] = stablehlo.subtract %[[V_96]], %[[V_84]] : tensor<1x2xf32>
// CHECK:     %[[V_98:.*]] = stablehlo.subtract %[[V_90]], %[[V_97]] : tensor<1x2xf32>
// CHECK:     %[[V_99:.*]] = stablehlo.add %[[V_89]], %[[V_95]] : tensor<1x2xf32>
// CHECK:     %[[V_100:.*]] = stablehlo.add %[[V_99]], %[[V_98]] : tensor<1x2xf32>
// CHECK:     %[[V_101:.*]] = stablehlo.add %[[V_96]], %[[V_100]] : tensor<1x2xf32>
// CHECK:     %[[V_102:.*]] = stablehlo.subtract %[[V_101]], %[[V_96]] : tensor<1x2xf32>
// CHECK:     %[[V_103:.*]] = stablehlo.subtract %[[V_100]], %[[V_102]] : tensor<1x2xf32>
// CHECK:     %[[V_104:.*]] = stablehlo.concatenate %[[V_101]], %[[V_103]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK:     %[[V_105:.*]] = stablehlo.convert %[[V_104]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
// CHECK:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK:     %[[V_106:.*]] = stablehlo.reduce(%[[V_105]] init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
// CHECK:     return %[[V_106]] : tensor<2xf64>

