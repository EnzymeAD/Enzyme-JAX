// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck --check-prefix=FIRST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=TUPLE %s
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @test_combine_add(%arg0: tensor<1xf64>, %arg1: tensor<1xf64>, %arg2: tensor<1xf64>, %arg3: tensor<1xf64>) -> tensor<2xf64> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<1xf64>
  %1 = stablehlo.add %arg2, %arg3 : tensor<1xf64>
  %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
  return %2 : tensor<2xf64>
}

// FIRST-LABEL: func.func @test_combine_add
// FIRST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<1xf64>) -> tensor<1xf32>
// FIRST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<1xf32>) -> tensor<1xf64>
// FIRST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<1xf64>
// FIRST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<1xf64>) -> tensor<1xf32>
// FIRST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<1xf32>) -> tensor<1x1xf32>
// FIRST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<1xf32>) -> tensor<1x1xf32>
// FIRST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<2x1xf32>
// FIRST:     %[[V_7:.*]] = stablehlo.convert %arg1 : (tensor<1xf64>) -> tensor<1xf32>
// FIRST:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<1xf32>) -> tensor<1xf64>
// FIRST:     %[[V_9:.*]] = stablehlo.subtract %arg1, %[[V_8]] : tensor<1xf64>
// FIRST:     %[[V_10:.*]] = stablehlo.convert %[[V_9]] : (tensor<1xf64>) -> tensor<1xf32>
// FIRST:     %[[V_11:.*]] = stablehlo.reshape %[[V_7]] : (tensor<1xf32>) -> tensor<1x1xf32>
// FIRST:     %[[V_12:.*]] = stablehlo.reshape %[[V_10]] : (tensor<1xf32>) -> tensor<1x1xf32>
// FIRST:     %[[V_13:.*]] = stablehlo.concatenate %[[V_11]], %[[V_12]], dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<2x1xf32>
// FIRST:     %[[V_14:.*]] = stablehlo.convert %arg2 : (tensor<1xf64>) -> tensor<1xf32>
// FIRST:     %[[V_15:.*]] = stablehlo.convert %[[V_14]] : (tensor<1xf32>) -> tensor<1xf64>
// FIRST:     %[[V_16:.*]] = stablehlo.subtract %arg2, %[[V_15]] : tensor<1xf64>
// FIRST:     %[[V_17:.*]] = stablehlo.convert %[[V_16]] : (tensor<1xf64>) -> tensor<1xf32>
// FIRST:     %[[V_18:.*]] = stablehlo.reshape %[[V_14]] : (tensor<1xf32>) -> tensor<1x1xf32>
// FIRST:     %[[V_19:.*]] = stablehlo.reshape %[[V_17]] : (tensor<1xf32>) -> tensor<1x1xf32>
// FIRST:     %[[V_20:.*]] = stablehlo.concatenate %[[V_18]], %[[V_19]], dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<2x1xf32>
// FIRST:     %[[V_21:.*]] = stablehlo.convert %arg3 : (tensor<1xf64>) -> tensor<1xf32>
// FIRST:     %[[V_22:.*]] = stablehlo.convert %[[V_21]] : (tensor<1xf32>) -> tensor<1xf64>
// FIRST:     %[[V_23:.*]] = stablehlo.subtract %arg3, %[[V_22]] : tensor<1xf64>
// FIRST:     %[[V_24:.*]] = stablehlo.convert %[[V_23]] : (tensor<1xf64>) -> tensor<1xf32>
// FIRST:     %[[V_25:.*]] = stablehlo.reshape %[[V_21]] : (tensor<1xf32>) -> tensor<1x1xf32>
// FIRST:     %[[V_26:.*]] = stablehlo.reshape %[[V_24]] : (tensor<1xf32>) -> tensor<1x1xf32>
// FIRST:     %[[V_27:.*]] = stablehlo.concatenate %[[V_25]], %[[V_26]], dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<2x1xf32>
// FIRST:     %[[V_28:.*]] = stablehlo.slice %[[V_6]] [0:1, 0:1] : (tensor<2x1xf32>) -> tensor<1x1xf32>
// FIRST:     %[[V_29:.*]] = stablehlo.slice %[[V_6]] [1:2, 0:1] : (tensor<2x1xf32>) -> tensor<1x1xf32>
// FIRST:     %[[V_30:.*]] = stablehlo.slice %[[V_13]] [0:1, 0:1] : (tensor<2x1xf32>) -> tensor<1x1xf32>
// FIRST:     %[[V_31:.*]] = stablehlo.slice %[[V_13]] [1:2, 0:1] : (tensor<2x1xf32>) -> tensor<1x1xf32>
// FIRST:     %[[V_32:.*]] = stablehlo.add %[[V_28]], %[[V_30]] : tensor<1x1xf32>
// FIRST:     %[[V_33:.*]] = stablehlo.subtract %[[V_32]], %[[V_30]] : tensor<1x1xf32>
// FIRST:     %[[V_34:.*]] = stablehlo.subtract %[[V_32]], %[[V_33]] : tensor<1x1xf32>
// FIRST:     %[[V_35:.*]] = stablehlo.subtract %[[V_28]], %[[V_33]] : tensor<1x1xf32>
// FIRST:     %[[V_36:.*]] = stablehlo.subtract %[[V_30]], %[[V_34]] : tensor<1x1xf32>
// FIRST:     %[[V_37:.*]] = stablehlo.add %[[V_35]], %[[V_36]] : tensor<1x1xf32>
// FIRST:     %[[V_38:.*]] = stablehlo.add %[[V_29]], %[[V_31]] : tensor<1x1xf32>
// FIRST:     %[[V_39:.*]] = stablehlo.subtract %[[V_38]], %[[V_31]] : tensor<1x1xf32>
// FIRST:     %[[V_40:.*]] = stablehlo.subtract %[[V_38]], %[[V_39]] : tensor<1x1xf32>
// FIRST:     %[[V_41:.*]] = stablehlo.subtract %[[V_29]], %[[V_39]] : tensor<1x1xf32>
// FIRST:     %[[V_42:.*]] = stablehlo.subtract %[[V_31]], %[[V_40]] : tensor<1x1xf32>
// FIRST:     %[[V_43:.*]] = stablehlo.add %[[V_41]], %[[V_42]] : tensor<1x1xf32>
// FIRST:     %[[V_44:.*]] = stablehlo.add %[[V_32]], %[[V_38]] : tensor<1x1xf32>
// FIRST:     %[[V_45:.*]] = stablehlo.subtract %[[V_44]], %[[V_32]] : tensor<1x1xf32>
// FIRST:     %[[V_46:.*]] = stablehlo.subtract %[[V_38]], %[[V_45]] : tensor<1x1xf32>
// FIRST:     %[[V_47:.*]] = stablehlo.add %[[V_37]], %[[V_43]] : tensor<1x1xf32>
// FIRST:     %[[V_48:.*]] = stablehlo.add %[[V_47]], %[[V_46]] : tensor<1x1xf32>
// FIRST:     %[[V_49:.*]] = stablehlo.add %[[V_44]], %[[V_48]] : tensor<1x1xf32>
// FIRST:     %[[V_50:.*]] = stablehlo.subtract %[[V_49]], %[[V_44]] : tensor<1x1xf32>
// FIRST:     %[[V_51:.*]] = stablehlo.subtract %[[V_48]], %[[V_50]] : tensor<1x1xf32>
// FIRST:     %[[V_52:.*]] = stablehlo.concatenate %[[V_49]], %[[V_51]], dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<2x1xf32>
// FIRST:     %[[V_53:.*]] = stablehlo.slice %[[V_20]] [0:1, 0:1] : (tensor<2x1xf32>) -> tensor<1x1xf32>
// FIRST:     %[[V_54:.*]] = stablehlo.slice %[[V_20]] [1:2, 0:1] : (tensor<2x1xf32>) -> tensor<1x1xf32>
// FIRST:     %[[V_55:.*]] = stablehlo.slice %[[V_27]] [0:1, 0:1] : (tensor<2x1xf32>) -> tensor<1x1xf32>
// FIRST:     %[[V_56:.*]] = stablehlo.slice %[[V_27]] [1:2, 0:1] : (tensor<2x1xf32>) -> tensor<1x1xf32>
// FIRST:     %[[V_57:.*]] = stablehlo.add %[[V_53]], %[[V_55]] : tensor<1x1xf32>
// FIRST:     %[[V_58:.*]] = stablehlo.subtract %[[V_57]], %[[V_55]] : tensor<1x1xf32>
// FIRST:     %[[V_59:.*]] = stablehlo.subtract %[[V_57]], %[[V_58]] : tensor<1x1xf32>
// FIRST:     %[[V_60:.*]] = stablehlo.subtract %[[V_53]], %[[V_58]] : tensor<1x1xf32>
// FIRST:     %[[V_61:.*]] = stablehlo.subtract %[[V_55]], %[[V_59]] : tensor<1x1xf32>
// FIRST:     %[[V_62:.*]] = stablehlo.add %[[V_60]], %[[V_61]] : tensor<1x1xf32>
// FIRST:     %[[V_63:.*]] = stablehlo.add %[[V_54]], %[[V_56]] : tensor<1x1xf32>
// FIRST:     %[[V_64:.*]] = stablehlo.subtract %[[V_63]], %[[V_56]] : tensor<1x1xf32>
// FIRST:     %[[V_65:.*]] = stablehlo.subtract %[[V_63]], %[[V_64]] : tensor<1x1xf32>
// FIRST:     %[[V_66:.*]] = stablehlo.subtract %[[V_54]], %[[V_64]] : tensor<1x1xf32>
// FIRST:     %[[V_67:.*]] = stablehlo.subtract %[[V_56]], %[[V_65]] : tensor<1x1xf32>
// FIRST:     %[[V_68:.*]] = stablehlo.add %[[V_66]], %[[V_67]] : tensor<1x1xf32>
// FIRST:     %[[V_69:.*]] = stablehlo.add %[[V_57]], %[[V_63]] : tensor<1x1xf32>
// FIRST:     %[[V_70:.*]] = stablehlo.subtract %[[V_69]], %[[V_57]] : tensor<1x1xf32>
// FIRST:     %[[V_71:.*]] = stablehlo.subtract %[[V_63]], %[[V_70]] : tensor<1x1xf32>
// FIRST:     %[[V_72:.*]] = stablehlo.add %[[V_62]], %[[V_68]] : tensor<1x1xf32>
// FIRST:     %[[V_73:.*]] = stablehlo.add %[[V_72]], %[[V_71]] : tensor<1x1xf32>
// FIRST:     %[[V_74:.*]] = stablehlo.add %[[V_69]], %[[V_73]] : tensor<1x1xf32>
// FIRST:     %[[V_75:.*]] = stablehlo.subtract %[[V_74]], %[[V_69]] : tensor<1x1xf32>
// FIRST:     %[[V_76:.*]] = stablehlo.subtract %[[V_73]], %[[V_75]] : tensor<1x1xf32>
// FIRST:     %[[V_77:.*]] = stablehlo.concatenate %[[V_74]], %[[V_76]], dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<2x1xf32>
// FIRST:     %[[V_78:.*]] = stablehlo.concatenate %[[V_6]], %[[V_20]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// FIRST:     %[[V_79:.*]] = stablehlo.concatenate %[[V_13]], %[[V_27]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// FIRST:     %[[V_80:.*]] = stablehlo.slice %[[V_78]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_81:.*]] = stablehlo.slice %[[V_78]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_82:.*]] = stablehlo.slice %[[V_79]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_83:.*]] = stablehlo.slice %[[V_79]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_84:.*]] = stablehlo.add %[[V_80]], %[[V_82]] : tensor<1x2xf32>
// FIRST:     %[[V_85:.*]] = stablehlo.subtract %[[V_84]], %[[V_82]] : tensor<1x2xf32>
// FIRST:     %[[V_86:.*]] = stablehlo.subtract %[[V_84]], %[[V_85]] : tensor<1x2xf32>
// FIRST:     %[[V_87:.*]] = stablehlo.subtract %[[V_80]], %[[V_85]] : tensor<1x2xf32>
// FIRST:     %[[V_88:.*]] = stablehlo.subtract %[[V_82]], %[[V_86]] : tensor<1x2xf32>
// FIRST:     %[[V_89:.*]] = stablehlo.add %[[V_87]], %[[V_88]] : tensor<1x2xf32>
// FIRST:     %[[V_90:.*]] = stablehlo.add %[[V_81]], %[[V_83]] : tensor<1x2xf32>
// FIRST:     %[[V_91:.*]] = stablehlo.subtract %[[V_90]], %[[V_83]] : tensor<1x2xf32>
// FIRST:     %[[V_92:.*]] = stablehlo.subtract %[[V_90]], %[[V_91]] : tensor<1x2xf32>
// FIRST:     %[[V_93:.*]] = stablehlo.subtract %[[V_81]], %[[V_91]] : tensor<1x2xf32>
// FIRST:     %[[V_94:.*]] = stablehlo.subtract %[[V_83]], %[[V_92]] : tensor<1x2xf32>
// FIRST:     %[[V_95:.*]] = stablehlo.add %[[V_93]], %[[V_94]] : tensor<1x2xf32>
// FIRST:     %[[V_96:.*]] = stablehlo.add %[[V_84]], %[[V_90]] : tensor<1x2xf32>
// FIRST:     %[[V_97:.*]] = stablehlo.subtract %[[V_96]], %[[V_84]] : tensor<1x2xf32>
// FIRST:     %[[V_98:.*]] = stablehlo.subtract %[[V_90]], %[[V_97]] : tensor<1x2xf32>
// FIRST:     %[[V_99:.*]] = stablehlo.add %[[V_89]], %[[V_95]] : tensor<1x2xf32>
// FIRST:     %[[V_100:.*]] = stablehlo.add %[[V_99]], %[[V_98]] : tensor<1x2xf32>
// FIRST:     %[[V_101:.*]] = stablehlo.add %[[V_96]], %[[V_100]] : tensor<1x2xf32>
// FIRST:     %[[V_102:.*]] = stablehlo.subtract %[[V_101]], %[[V_96]] : tensor<1x2xf32>
// FIRST:     %[[V_103:.*]] = stablehlo.subtract %[[V_100]], %[[V_102]] : tensor<1x2xf32>
// FIRST:     %[[V_104:.*]] = stablehlo.concatenate %[[V_101]], %[[V_103]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST:     %[[V_105:.*]] = stablehlo.convert %[[V_104]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// FIRST:     %[[V_106:.*]] = stablehlo.reduce(%[[V_105]] init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
// FIRST:     return %[[V_106]] : tensor<2xf64>

// LAST-LABEL: func.func @test_combine_add
// LAST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<1xf64>) -> tensor<1xf32>
// LAST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<1xf32>) -> tensor<1xf64>
// LAST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<1xf64>
// LAST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<1xf64>) -> tensor<1xf32>
// LAST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<1xf32>) -> tensor<1x1xf32>
// LAST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<1xf32>) -> tensor<1x1xf32>
// LAST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 1 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x2xf32>
// LAST:     %[[V_7:.*]] = stablehlo.convert %arg1 : (tensor<1xf64>) -> tensor<1xf32>
// LAST:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<1xf32>) -> tensor<1xf64>
// LAST:     %[[V_9:.*]] = stablehlo.subtract %arg1, %[[V_8]] : tensor<1xf64>
// LAST:     %[[V_10:.*]] = stablehlo.convert %[[V_9]] : (tensor<1xf64>) -> tensor<1xf32>
// LAST:     %[[V_11:.*]] = stablehlo.reshape %[[V_7]] : (tensor<1xf32>) -> tensor<1x1xf32>
// LAST:     %[[V_12:.*]] = stablehlo.reshape %[[V_10]] : (tensor<1xf32>) -> tensor<1x1xf32>
// LAST:     %[[V_13:.*]] = stablehlo.concatenate %[[V_11]], %[[V_12]], dim = 1 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x2xf32>
// LAST:     %[[V_14:.*]] = stablehlo.convert %arg2 : (tensor<1xf64>) -> tensor<1xf32>
// LAST:     %[[V_15:.*]] = stablehlo.convert %[[V_14]] : (tensor<1xf32>) -> tensor<1xf64>
// LAST:     %[[V_16:.*]] = stablehlo.subtract %arg2, %[[V_15]] : tensor<1xf64>
// LAST:     %[[V_17:.*]] = stablehlo.convert %[[V_16]] : (tensor<1xf64>) -> tensor<1xf32>
// LAST:     %[[V_18:.*]] = stablehlo.reshape %[[V_14]] : (tensor<1xf32>) -> tensor<1x1xf32>
// LAST:     %[[V_19:.*]] = stablehlo.reshape %[[V_17]] : (tensor<1xf32>) -> tensor<1x1xf32>
// LAST:     %[[V_20:.*]] = stablehlo.concatenate %[[V_18]], %[[V_19]], dim = 1 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x2xf32>
// LAST:     %[[V_21:.*]] = stablehlo.convert %arg3 : (tensor<1xf64>) -> tensor<1xf32>
// LAST:     %[[V_22:.*]] = stablehlo.convert %[[V_21]] : (tensor<1xf32>) -> tensor<1xf64>
// LAST:     %[[V_23:.*]] = stablehlo.subtract %arg3, %[[V_22]] : tensor<1xf64>
// LAST:     %[[V_24:.*]] = stablehlo.convert %[[V_23]] : (tensor<1xf64>) -> tensor<1xf32>
// LAST:     %[[V_25:.*]] = stablehlo.reshape %[[V_21]] : (tensor<1xf32>) -> tensor<1x1xf32>
// LAST:     %[[V_26:.*]] = stablehlo.reshape %[[V_24]] : (tensor<1xf32>) -> tensor<1x1xf32>
// LAST:     %[[V_27:.*]] = stablehlo.concatenate %[[V_25]], %[[V_26]], dim = 1 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x2xf32>
// LAST:     %[[V_28:.*]] = stablehlo.slice %[[V_6]] [0:1, 0:1] : (tensor<1x2xf32>) -> tensor<1x1xf32>
// LAST:     %[[V_29:.*]] = stablehlo.slice %[[V_6]] [0:1, 1:2] : (tensor<1x2xf32>) -> tensor<1x1xf32>
// LAST:     %[[V_30:.*]] = stablehlo.slice %[[V_13]] [0:1, 0:1] : (tensor<1x2xf32>) -> tensor<1x1xf32>
// LAST:     %[[V_31:.*]] = stablehlo.slice %[[V_13]] [0:1, 1:2] : (tensor<1x2xf32>) -> tensor<1x1xf32>
// LAST:     %[[V_32:.*]] = stablehlo.add %[[V_28]], %[[V_30]] : tensor<1x1xf32>
// LAST:     %[[V_33:.*]] = stablehlo.subtract %[[V_32]], %[[V_30]] : tensor<1x1xf32>
// LAST:     %[[V_34:.*]] = stablehlo.subtract %[[V_32]], %[[V_33]] : tensor<1x1xf32>
// LAST:     %[[V_35:.*]] = stablehlo.subtract %[[V_28]], %[[V_33]] : tensor<1x1xf32>
// LAST:     %[[V_36:.*]] = stablehlo.subtract %[[V_30]], %[[V_34]] : tensor<1x1xf32>
// LAST:     %[[V_37:.*]] = stablehlo.add %[[V_35]], %[[V_36]] : tensor<1x1xf32>
// LAST:     %[[V_38:.*]] = stablehlo.add %[[V_29]], %[[V_31]] : tensor<1x1xf32>
// LAST:     %[[V_39:.*]] = stablehlo.subtract %[[V_38]], %[[V_31]] : tensor<1x1xf32>
// LAST:     %[[V_40:.*]] = stablehlo.subtract %[[V_38]], %[[V_39]] : tensor<1x1xf32>
// LAST:     %[[V_41:.*]] = stablehlo.subtract %[[V_29]], %[[V_39]] : tensor<1x1xf32>
// LAST:     %[[V_42:.*]] = stablehlo.subtract %[[V_31]], %[[V_40]] : tensor<1x1xf32>
// LAST:     %[[V_43:.*]] = stablehlo.add %[[V_41]], %[[V_42]] : tensor<1x1xf32>
// LAST:     %[[V_44:.*]] = stablehlo.add %[[V_32]], %[[V_38]] : tensor<1x1xf32>
// LAST:     %[[V_45:.*]] = stablehlo.subtract %[[V_44]], %[[V_32]] : tensor<1x1xf32>
// LAST:     %[[V_46:.*]] = stablehlo.subtract %[[V_38]], %[[V_45]] : tensor<1x1xf32>
// LAST:     %[[V_47:.*]] = stablehlo.add %[[V_37]], %[[V_43]] : tensor<1x1xf32>
// LAST:     %[[V_48:.*]] = stablehlo.add %[[V_47]], %[[V_46]] : tensor<1x1xf32>
// LAST:     %[[V_49:.*]] = stablehlo.add %[[V_44]], %[[V_48]] : tensor<1x1xf32>
// LAST:     %[[V_50:.*]] = stablehlo.subtract %[[V_49]], %[[V_44]] : tensor<1x1xf32>
// LAST:     %[[V_51:.*]] = stablehlo.subtract %[[V_48]], %[[V_50]] : tensor<1x1xf32>
// LAST:     %[[V_52:.*]] = stablehlo.concatenate %[[V_49]], %[[V_51]], dim = 1 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x2xf32>
// LAST:     %[[V_53:.*]] = stablehlo.slice %[[V_20]] [0:1, 0:1] : (tensor<1x2xf32>) -> tensor<1x1xf32>
// LAST:     %[[V_54:.*]] = stablehlo.slice %[[V_20]] [0:1, 1:2] : (tensor<1x2xf32>) -> tensor<1x1xf32>
// LAST:     %[[V_55:.*]] = stablehlo.slice %[[V_27]] [0:1, 0:1] : (tensor<1x2xf32>) -> tensor<1x1xf32>
// LAST:     %[[V_56:.*]] = stablehlo.slice %[[V_27]] [0:1, 1:2] : (tensor<1x2xf32>) -> tensor<1x1xf32>
// LAST:     %[[V_57:.*]] = stablehlo.add %[[V_53]], %[[V_55]] : tensor<1x1xf32>
// LAST:     %[[V_58:.*]] = stablehlo.subtract %[[V_57]], %[[V_55]] : tensor<1x1xf32>
// LAST:     %[[V_59:.*]] = stablehlo.subtract %[[V_57]], %[[V_58]] : tensor<1x1xf32>
// LAST:     %[[V_60:.*]] = stablehlo.subtract %[[V_53]], %[[V_58]] : tensor<1x1xf32>
// LAST:     %[[V_61:.*]] = stablehlo.subtract %[[V_55]], %[[V_59]] : tensor<1x1xf32>
// LAST:     %[[V_62:.*]] = stablehlo.add %[[V_60]], %[[V_61]] : tensor<1x1xf32>
// LAST:     %[[V_63:.*]] = stablehlo.add %[[V_54]], %[[V_56]] : tensor<1x1xf32>
// LAST:     %[[V_64:.*]] = stablehlo.subtract %[[V_63]], %[[V_56]] : tensor<1x1xf32>
// LAST:     %[[V_65:.*]] = stablehlo.subtract %[[V_63]], %[[V_64]] : tensor<1x1xf32>
// LAST:     %[[V_66:.*]] = stablehlo.subtract %[[V_54]], %[[V_64]] : tensor<1x1xf32>
// LAST:     %[[V_67:.*]] = stablehlo.subtract %[[V_56]], %[[V_65]] : tensor<1x1xf32>
// LAST:     %[[V_68:.*]] = stablehlo.add %[[V_66]], %[[V_67]] : tensor<1x1xf32>
// LAST:     %[[V_69:.*]] = stablehlo.add %[[V_57]], %[[V_63]] : tensor<1x1xf32>
// LAST:     %[[V_70:.*]] = stablehlo.subtract %[[V_69]], %[[V_57]] : tensor<1x1xf32>
// LAST:     %[[V_71:.*]] = stablehlo.subtract %[[V_63]], %[[V_70]] : tensor<1x1xf32>
// LAST:     %[[V_72:.*]] = stablehlo.add %[[V_62]], %[[V_68]] : tensor<1x1xf32>
// LAST:     %[[V_73:.*]] = stablehlo.add %[[V_72]], %[[V_71]] : tensor<1x1xf32>
// LAST:     %[[V_74:.*]] = stablehlo.add %[[V_69]], %[[V_73]] : tensor<1x1xf32>
// LAST:     %[[V_75:.*]] = stablehlo.subtract %[[V_74]], %[[V_69]] : tensor<1x1xf32>
// LAST:     %[[V_76:.*]] = stablehlo.subtract %[[V_73]], %[[V_75]] : tensor<1x1xf32>
// LAST:     %[[V_77:.*]] = stablehlo.concatenate %[[V_74]], %[[V_76]], dim = 1 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x2xf32>
// LAST:     %[[V_78:.*]] = stablehlo.concatenate %[[V_6]], %[[V_20]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// LAST:     %[[V_79:.*]] = stablehlo.concatenate %[[V_13]], %[[V_27]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// LAST:     %[[V_80:.*]] = stablehlo.slice %[[V_78]] [0:2, 0:1] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_81:.*]] = stablehlo.slice %[[V_78]] [0:2, 1:2] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_82:.*]] = stablehlo.slice %[[V_79]] [0:2, 0:1] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_83:.*]] = stablehlo.slice %[[V_79]] [0:2, 1:2] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_84:.*]] = stablehlo.add %[[V_80]], %[[V_82]] : tensor<2x1xf32>
// LAST:     %[[V_85:.*]] = stablehlo.subtract %[[V_84]], %[[V_82]] : tensor<2x1xf32>
// LAST:     %[[V_86:.*]] = stablehlo.subtract %[[V_84]], %[[V_85]] : tensor<2x1xf32>
// LAST:     %[[V_87:.*]] = stablehlo.subtract %[[V_80]], %[[V_85]] : tensor<2x1xf32>
// LAST:     %[[V_88:.*]] = stablehlo.subtract %[[V_82]], %[[V_86]] : tensor<2x1xf32>
// LAST:     %[[V_89:.*]] = stablehlo.add %[[V_87]], %[[V_88]] : tensor<2x1xf32>
// LAST:     %[[V_90:.*]] = stablehlo.add %[[V_81]], %[[V_83]] : tensor<2x1xf32>
// LAST:     %[[V_91:.*]] = stablehlo.subtract %[[V_90]], %[[V_83]] : tensor<2x1xf32>
// LAST:     %[[V_92:.*]] = stablehlo.subtract %[[V_90]], %[[V_91]] : tensor<2x1xf32>
// LAST:     %[[V_93:.*]] = stablehlo.subtract %[[V_81]], %[[V_91]] : tensor<2x1xf32>
// LAST:     %[[V_94:.*]] = stablehlo.subtract %[[V_83]], %[[V_92]] : tensor<2x1xf32>
// LAST:     %[[V_95:.*]] = stablehlo.add %[[V_93]], %[[V_94]] : tensor<2x1xf32>
// LAST:     %[[V_96:.*]] = stablehlo.add %[[V_84]], %[[V_90]] : tensor<2x1xf32>
// LAST:     %[[V_97:.*]] = stablehlo.subtract %[[V_96]], %[[V_84]] : tensor<2x1xf32>
// LAST:     %[[V_98:.*]] = stablehlo.subtract %[[V_90]], %[[V_97]] : tensor<2x1xf32>
// LAST:     %[[V_99:.*]] = stablehlo.add %[[V_89]], %[[V_95]] : tensor<2x1xf32>
// LAST:     %[[V_100:.*]] = stablehlo.add %[[V_99]], %[[V_98]] : tensor<2x1xf32>
// LAST:     %[[V_101:.*]] = stablehlo.add %[[V_96]], %[[V_100]] : tensor<2x1xf32>
// LAST:     %[[V_102:.*]] = stablehlo.subtract %[[V_101]], %[[V_96]] : tensor<2x1xf32>
// LAST:     %[[V_103:.*]] = stablehlo.subtract %[[V_100]], %[[V_102]] : tensor<2x1xf32>
// LAST:     %[[V_104:.*]] = stablehlo.concatenate %[[V_101]], %[[V_103]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// LAST:     %[[V_105:.*]] = stablehlo.convert %[[V_104]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
// LAST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// LAST:     %[[V_106:.*]] = stablehlo.reduce(%[[V_105]] init: %[[CST]]) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
// LAST:     return %[[V_106]] : tensor<2xf64>

// TUPLE-LABEL: func.func @test_combine_add
// TUPLE:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<1xf64>) -> tensor<1xf32>
// TUPLE:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<1xf32>) -> tensor<1xf64>
// TUPLE:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<1xf64>
// TUPLE:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<1xf64>) -> tensor<1xf32>
// TUPLE:     %[[V_4:.*]] = stablehlo.tuple %[[V_0]], %[[V_3]] : tuple<tensor<1xf32>, tensor<1xf32>>
// TUPLE:     %[[V_5:.*]] = stablehlo.convert %arg1 : (tensor<1xf64>) -> tensor<1xf32>
// TUPLE:     %[[V_6:.*]] = stablehlo.convert %[[V_5]] : (tensor<1xf32>) -> tensor<1xf64>
// TUPLE:     %[[V_7:.*]] = stablehlo.subtract %arg1, %[[V_6]] : tensor<1xf64>
// TUPLE:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<1xf64>) -> tensor<1xf32>
// TUPLE:     %[[V_9:.*]] = stablehlo.tuple %[[V_5]], %[[V_8]] : tuple<tensor<1xf32>, tensor<1xf32>>
// TUPLE:     %[[V_10:.*]] = stablehlo.convert %arg2 : (tensor<1xf64>) -> tensor<1xf32>
// TUPLE:     %[[V_11:.*]] = stablehlo.convert %[[V_10]] : (tensor<1xf32>) -> tensor<1xf64>
// TUPLE:     %[[V_12:.*]] = stablehlo.subtract %arg2, %[[V_11]] : tensor<1xf64>
// TUPLE:     %[[V_13:.*]] = stablehlo.convert %[[V_12]] : (tensor<1xf64>) -> tensor<1xf32>
// TUPLE:     %[[V_14:.*]] = stablehlo.tuple %[[V_10]], %[[V_13]] : tuple<tensor<1xf32>, tensor<1xf32>>
// TUPLE:     %[[V_15:.*]] = stablehlo.convert %arg3 : (tensor<1xf64>) -> tensor<1xf32>
// TUPLE:     %[[V_16:.*]] = stablehlo.convert %[[V_15]] : (tensor<1xf32>) -> tensor<1xf64>
// TUPLE:     %[[V_17:.*]] = stablehlo.subtract %arg3, %[[V_16]] : tensor<1xf64>
// TUPLE:     %[[V_18:.*]] = stablehlo.convert %[[V_17]] : (tensor<1xf64>) -> tensor<1xf32>
// TUPLE:     %[[V_19:.*]] = stablehlo.tuple %[[V_15]], %[[V_18]] : tuple<tensor<1xf32>, tensor<1xf32>>
// TUPLE:     %[[V_20:.*]] = stablehlo.get_tuple_element %[[V_4]][0] : (tuple<tensor<1xf32>, tensor<1xf32>>) -> tensor<1xf32>
// TUPLE:     %[[V_21:.*]] = stablehlo.get_tuple_element %[[V_4]][1] : (tuple<tensor<1xf32>, tensor<1xf32>>) -> tensor<1xf32>
// TUPLE:     %[[V_22:.*]] = stablehlo.get_tuple_element %[[V_9]][0] : (tuple<tensor<1xf32>, tensor<1xf32>>) -> tensor<1xf32>
// TUPLE:     %[[V_23:.*]] = stablehlo.get_tuple_element %[[V_9]][1] : (tuple<tensor<1xf32>, tensor<1xf32>>) -> tensor<1xf32>
// TUPLE:     %[[V_24:.*]] = stablehlo.add %[[V_20]], %[[V_22]] : tensor<1xf32>
// TUPLE:     %[[V_25:.*]] = stablehlo.subtract %[[V_24]], %[[V_22]] : tensor<1xf32>
// TUPLE:     %[[V_26:.*]] = stablehlo.subtract %[[V_24]], %[[V_25]] : tensor<1xf32>
// TUPLE:     %[[V_27:.*]] = stablehlo.subtract %[[V_20]], %[[V_25]] : tensor<1xf32>
// TUPLE:     %[[V_28:.*]] = stablehlo.subtract %[[V_22]], %[[V_26]] : tensor<1xf32>
// TUPLE:     %[[V_29:.*]] = stablehlo.add %[[V_27]], %[[V_28]] : tensor<1xf32>
// TUPLE:     %[[V_30:.*]] = stablehlo.add %[[V_21]], %[[V_23]] : tensor<1xf32>
// TUPLE:     %[[V_31:.*]] = stablehlo.subtract %[[V_30]], %[[V_23]] : tensor<1xf32>
// TUPLE:     %[[V_32:.*]] = stablehlo.subtract %[[V_30]], %[[V_31]] : tensor<1xf32>
// TUPLE:     %[[V_33:.*]] = stablehlo.subtract %[[V_21]], %[[V_31]] : tensor<1xf32>
// TUPLE:     %[[V_34:.*]] = stablehlo.subtract %[[V_23]], %[[V_32]] : tensor<1xf32>
// TUPLE:     %[[V_35:.*]] = stablehlo.add %[[V_33]], %[[V_34]] : tensor<1xf32>
// TUPLE:     %[[V_36:.*]] = stablehlo.add %[[V_24]], %[[V_30]] : tensor<1xf32>
// TUPLE:     %[[V_37:.*]] = stablehlo.subtract %[[V_36]], %[[V_24]] : tensor<1xf32>
// TUPLE:     %[[V_38:.*]] = stablehlo.subtract %[[V_30]], %[[V_37]] : tensor<1xf32>
// TUPLE:     %[[V_39:.*]] = stablehlo.add %[[V_29]], %[[V_35]] : tensor<1xf32>
// TUPLE:     %[[V_40:.*]] = stablehlo.add %[[V_39]], %[[V_38]] : tensor<1xf32>
// TUPLE:     %[[V_41:.*]] = stablehlo.add %[[V_36]], %[[V_40]] : tensor<1xf32>
// TUPLE:     %[[V_42:.*]] = stablehlo.subtract %[[V_41]], %[[V_36]] : tensor<1xf32>
// TUPLE:     %[[V_43:.*]] = stablehlo.subtract %[[V_40]], %[[V_42]] : tensor<1xf32>
// TUPLE:     %[[V_44:.*]] = stablehlo.tuple %[[V_41]], %[[V_43]] : tuple<tensor<1xf32>, tensor<1xf32>>
// TUPLE:     %[[V_45:.*]] = stablehlo.get_tuple_element %[[V_14]][0] : (tuple<tensor<1xf32>, tensor<1xf32>>) -> tensor<1xf32>
// TUPLE:     %[[V_46:.*]] = stablehlo.get_tuple_element %[[V_14]][1] : (tuple<tensor<1xf32>, tensor<1xf32>>) -> tensor<1xf32>
// TUPLE:     %[[V_47:.*]] = stablehlo.get_tuple_element %[[V_19]][0] : (tuple<tensor<1xf32>, tensor<1xf32>>) -> tensor<1xf32>
// TUPLE:     %[[V_48:.*]] = stablehlo.get_tuple_element %[[V_19]][1] : (tuple<tensor<1xf32>, tensor<1xf32>>) -> tensor<1xf32>
// TUPLE:     %[[V_49:.*]] = stablehlo.add %[[V_45]], %[[V_47]] : tensor<1xf32>
// TUPLE:     %[[V_50:.*]] = stablehlo.subtract %[[V_49]], %[[V_47]] : tensor<1xf32>
// TUPLE:     %[[V_51:.*]] = stablehlo.subtract %[[V_49]], %[[V_50]] : tensor<1xf32>
// TUPLE:     %[[V_52:.*]] = stablehlo.subtract %[[V_45]], %[[V_50]] : tensor<1xf32>
// TUPLE:     %[[V_53:.*]] = stablehlo.subtract %[[V_47]], %[[V_51]] : tensor<1xf32>
// TUPLE:     %[[V_54:.*]] = stablehlo.add %[[V_52]], %[[V_53]] : tensor<1xf32>
// TUPLE:     %[[V_55:.*]] = stablehlo.add %[[V_46]], %[[V_48]] : tensor<1xf32>
// TUPLE:     %[[V_56:.*]] = stablehlo.subtract %[[V_55]], %[[V_48]] : tensor<1xf32>
// TUPLE:     %[[V_57:.*]] = stablehlo.subtract %[[V_55]], %[[V_56]] : tensor<1xf32>
// TUPLE:     %[[V_58:.*]] = stablehlo.subtract %[[V_46]], %[[V_56]] : tensor<1xf32>
// TUPLE:     %[[V_59:.*]] = stablehlo.subtract %[[V_48]], %[[V_57]] : tensor<1xf32>
// TUPLE:     %[[V_60:.*]] = stablehlo.add %[[V_58]], %[[V_59]] : tensor<1xf32>
// TUPLE:     %[[V_61:.*]] = stablehlo.add %[[V_49]], %[[V_55]] : tensor<1xf32>
// TUPLE:     %[[V_62:.*]] = stablehlo.subtract %[[V_61]], %[[V_49]] : tensor<1xf32>
// TUPLE:     %[[V_63:.*]] = stablehlo.subtract %[[V_55]], %[[V_62]] : tensor<1xf32>
// TUPLE:     %[[V_64:.*]] = stablehlo.add %[[V_54]], %[[V_60]] : tensor<1xf32>
// TUPLE:     %[[V_65:.*]] = stablehlo.add %[[V_64]], %[[V_63]] : tensor<1xf32>
// TUPLE:     %[[V_66:.*]] = stablehlo.add %[[V_61]], %[[V_65]] : tensor<1xf32>
// TUPLE:     %[[V_67:.*]] = stablehlo.subtract %[[V_66]], %[[V_61]] : tensor<1xf32>
// TUPLE:     %[[V_68:.*]] = stablehlo.subtract %[[V_65]], %[[V_67]] : tensor<1xf32>
// TUPLE:     %[[V_69:.*]] = stablehlo.tuple %[[V_66]], %[[V_68]] : tuple<tensor<1xf32>, tensor<1xf32>>
// TUPLE:     %[[V_70:.*]] = stablehlo.get_tuple_element %[[V_4]][0] : (tuple<tensor<1xf32>, tensor<1xf32>>) -> tensor<1xf32>
// TUPLE:     %[[V_71:.*]] = stablehlo.get_tuple_element %[[V_4]][1] : (tuple<tensor<1xf32>, tensor<1xf32>>) -> tensor<1xf32>
// TUPLE:     %[[V_72:.*]] = stablehlo.get_tuple_element %[[V_14]][0] : (tuple<tensor<1xf32>, tensor<1xf32>>) -> tensor<1xf32>
// TUPLE:     %[[V_73:.*]] = stablehlo.get_tuple_element %[[V_14]][1] : (tuple<tensor<1xf32>, tensor<1xf32>>) -> tensor<1xf32>
// TUPLE:     %[[V_74:.*]] = stablehlo.concatenate %[[V_70]], %[[V_72]], dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
// TUPLE:     %[[V_75:.*]] = stablehlo.concatenate %[[V_71]], %[[V_73]], dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
// TUPLE:     %[[V_76:.*]] = stablehlo.tuple %[[V_74]], %[[V_75]] : tuple<tensor<2xf32>, tensor<2xf32>>
// TUPLE:     %[[V_77:.*]] = stablehlo.get_tuple_element %[[V_9]][0] : (tuple<tensor<1xf32>, tensor<1xf32>>) -> tensor<1xf32>
// TUPLE:     %[[V_78:.*]] = stablehlo.get_tuple_element %[[V_9]][1] : (tuple<tensor<1xf32>, tensor<1xf32>>) -> tensor<1xf32>
// TUPLE:     %[[V_79:.*]] = stablehlo.get_tuple_element %[[V_19]][0] : (tuple<tensor<1xf32>, tensor<1xf32>>) -> tensor<1xf32>
// TUPLE:     %[[V_80:.*]] = stablehlo.get_tuple_element %[[V_19]][1] : (tuple<tensor<1xf32>, tensor<1xf32>>) -> tensor<1xf32>
// TUPLE:     %[[V_81:.*]] = stablehlo.concatenate %[[V_77]], %[[V_79]], dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
// TUPLE:     %[[V_82:.*]] = stablehlo.concatenate %[[V_78]], %[[V_80]], dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
// TUPLE:     %[[V_83:.*]] = stablehlo.tuple %[[V_81]], %[[V_82]] : tuple<tensor<2xf32>, tensor<2xf32>>
// TUPLE:     %[[V_84:.*]] = stablehlo.add %[[V_74]], %[[V_81]] : tensor<2xf32>
// TUPLE:     %[[V_85:.*]] = stablehlo.subtract %[[V_84]], %[[V_81]] : tensor<2xf32>
// TUPLE:     %[[V_86:.*]] = stablehlo.subtract %[[V_84]], %[[V_85]] : tensor<2xf32>
// TUPLE:     %[[V_87:.*]] = stablehlo.subtract %[[V_74]], %[[V_85]] : tensor<2xf32>
// TUPLE:     %[[V_88:.*]] = stablehlo.subtract %[[V_81]], %[[V_86]] : tensor<2xf32>
// TUPLE:     %[[V_89:.*]] = stablehlo.add %[[V_87]], %[[V_88]] : tensor<2xf32>
// TUPLE:     %[[V_90:.*]] = stablehlo.add %[[V_75]], %[[V_82]] : tensor<2xf32>
// TUPLE:     %[[V_91:.*]] = stablehlo.subtract %[[V_90]], %[[V_82]] : tensor<2xf32>
// TUPLE:     %[[V_92:.*]] = stablehlo.subtract %[[V_90]], %[[V_91]] : tensor<2xf32>
// TUPLE:     %[[V_93:.*]] = stablehlo.subtract %[[V_75]], %[[V_91]] : tensor<2xf32>
// TUPLE:     %[[V_94:.*]] = stablehlo.subtract %[[V_82]], %[[V_92]] : tensor<2xf32>
// TUPLE:     %[[V_95:.*]] = stablehlo.add %[[V_93]], %[[V_94]] : tensor<2xf32>
// TUPLE:     %[[V_96:.*]] = stablehlo.add %[[V_84]], %[[V_90]] : tensor<2xf32>
// TUPLE:     %[[V_97:.*]] = stablehlo.subtract %[[V_96]], %[[V_84]] : tensor<2xf32>
// TUPLE:     %[[V_98:.*]] = stablehlo.subtract %[[V_90]], %[[V_97]] : tensor<2xf32>
// TUPLE:     %[[V_99:.*]] = stablehlo.add %[[V_89]], %[[V_95]] : tensor<2xf32>
// TUPLE:     %[[V_100:.*]] = stablehlo.add %[[V_99]], %[[V_98]] : tensor<2xf32>
// TUPLE:     %[[V_101:.*]] = stablehlo.add %[[V_96]], %[[V_100]] : tensor<2xf32>
// TUPLE:     %[[V_102:.*]] = stablehlo.subtract %[[V_101]], %[[V_96]] : tensor<2xf32>
// TUPLE:     %[[V_103:.*]] = stablehlo.subtract %[[V_100]], %[[V_102]] : tensor<2xf32>
// TUPLE:     %[[V_104:.*]] = stablehlo.tuple %[[V_101]], %[[V_103]] : tuple<tensor<2xf32>, tensor<2xf32>>
// TUPLE:     %[[V_105:.*]] = stablehlo.convert %[[V_101]] : (tensor<2xf32>) -> tensor<2xf64>
// TUPLE:     %[[V_106:.*]] = stablehlo.convert %[[V_103]] : (tensor<2xf32>) -> tensor<2xf64>
// TUPLE:     %[[V_107:.*]] = stablehlo.add %[[V_105]], %[[V_106]] : tensor<2xf64>
// TUPLE:     return %[[V_107]] : tensor<2xf64>

func.func @main() attributes {enzyme.no_multifloat} {
  %c1 = stablehlo.constant dense<[1.1]> : tensor<1xf64>
  %c2 = stablehlo.constant dense<[2.2]> : tensor<1xf64>
  %c3 = stablehlo.constant dense<[3.3]> : tensor<1xf64>
  %c4 = stablehlo.constant dense<[4.4]> : tensor<1xf64>
  
  %expected = stablehlo.constant dense<[3.3000000000000003, 7.7]> : tensor<2xf64>
  
  %res = func.call @test_combine_add(%c1, %c2, %c3, %c4) : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
  
  "check.expect_almost_eq"(%res, %expected) : (tensor<2xf64>, tensor<2xf64>) -> ()
  return
}

// FIRST-LABEL: func.func @main
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<1.100000e+00> : tensor<1xf64>
// FIRST:     %[[CST_0:.*]] = stablehlo.constant dense<2.200000e+00> : tensor<1xf64>
// FIRST:     %[[CST_1:.*]] = stablehlo.constant dense<3.300000e+00> : tensor<1xf64>
// FIRST:     %[[CST_2:.*]] = stablehlo.constant dense<4.400000e+00> : tensor<1xf64>
// FIRST:     %[[CST_3:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<2xf64>
// FIRST:     %[[V_0:.*]] = call @test_combine_add(%[[CST]], %[[CST_0]], %[[CST_1]], %[[CST_2]]) : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
// FIRST:     check.expect_almost_eq %[[V_0]], %[[CST_3]] : tensor<2xf64>
// FIRST:     return
