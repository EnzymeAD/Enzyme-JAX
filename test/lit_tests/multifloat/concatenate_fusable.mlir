// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32" %s | FileCheck %s

func.func @main(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>) -> tensor<4xf64> {
  // CHECK-LABEL: @main
// CHECK:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// CHECK:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<2xf32>) -> tensor<2xf64>
// CHECK:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<2xf64>
// CHECK:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<2xf64>) -> tensor<2xf32>
// CHECK:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK:     %[[V_7:.*]] = stablehlo.convert %arg1 : (tensor<2xf64>) -> tensor<2xf32>
// CHECK:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<2xf32>) -> tensor<2xf64>
// CHECK:     %[[V_9:.*]] = stablehlo.subtract %arg1, %[[V_8]] : tensor<2xf64>
// CHECK:     %[[V_10:.*]] = stablehlo.convert %[[V_9]] : (tensor<2xf64>) -> tensor<2xf32>
// CHECK:     %[[V_11:.*]] = stablehlo.reshape %[[V_7]] : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK:     %[[V_12:.*]] = stablehlo.reshape %[[V_10]] : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK:     %[[V_13:.*]] = stablehlo.concatenate %[[V_11]], %[[V_12]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK:     %[[V_14:.*]] = stablehlo.slice %[[V_6]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// CHECK:     %[[V_15:.*]] = stablehlo.slice %[[V_6]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// CHECK:     %[[V_16:.*]] = stablehlo.slice %[[V_13]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// CHECK:     %[[V_17:.*]] = stablehlo.slice %[[V_13]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// CHECK:     %[[V_18:.*]] = stablehlo.add %[[V_14]], %[[V_16]] : tensor<1x2xf32>
// CHECK:     %[[V_19:.*]] = stablehlo.subtract %[[V_18]], %[[V_16]] : tensor<1x2xf32>
// CHECK:     %[[V_20:.*]] = stablehlo.subtract %[[V_18]], %[[V_19]] : tensor<1x2xf32>
// CHECK:     %[[V_21:.*]] = stablehlo.subtract %[[V_14]], %[[V_19]] : tensor<1x2xf32>
// CHECK:     %[[V_22:.*]] = stablehlo.subtract %[[V_16]], %[[V_20]] : tensor<1x2xf32>
// CHECK:     %[[V_23:.*]] = stablehlo.add %[[V_21]], %[[V_22]] : tensor<1x2xf32>
// CHECK:     %[[V_24:.*]] = stablehlo.add %[[V_15]], %[[V_17]] : tensor<1x2xf32>
// CHECK:     %[[V_25:.*]] = stablehlo.subtract %[[V_24]], %[[V_17]] : tensor<1x2xf32>
// CHECK:     %[[V_26:.*]] = stablehlo.subtract %[[V_24]], %[[V_25]] : tensor<1x2xf32>
// CHECK:     %[[V_27:.*]] = stablehlo.subtract %[[V_15]], %[[V_25]] : tensor<1x2xf32>
// CHECK:     %[[V_28:.*]] = stablehlo.subtract %[[V_17]], %[[V_26]] : tensor<1x2xf32>
// CHECK:     %[[V_29:.*]] = stablehlo.add %[[V_27]], %[[V_28]] : tensor<1x2xf32>
// CHECK:     %[[V_30:.*]] = stablehlo.add %[[V_18]], %[[V_24]] : tensor<1x2xf32>
// CHECK:     %[[V_31:.*]] = stablehlo.subtract %[[V_30]], %[[V_18]] : tensor<1x2xf32>
// CHECK:     %[[V_32:.*]] = stablehlo.subtract %[[V_24]], %[[V_31]] : tensor<1x2xf32>
// CHECK:     %[[V_33:.*]] = stablehlo.add %[[V_23]], %[[V_29]] : tensor<1x2xf32>
// CHECK:     %[[V_34:.*]] = stablehlo.add %[[V_33]], %[[V_32]] : tensor<1x2xf32>
// CHECK:     %[[V_35:.*]] = stablehlo.add %[[V_30]], %[[V_34]] : tensor<1x2xf32>
// CHECK:     %[[V_36:.*]] = stablehlo.subtract %[[V_35]], %[[V_30]] : tensor<1x2xf32>
// CHECK:     %[[V_37:.*]] = stablehlo.subtract %[[V_34]], %[[V_36]] : tensor<1x2xf32>
// CHECK:     %[[V_38:.*]] = stablehlo.concatenate %[[V_35]], %[[V_37]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK:     %[[V_39:.*]] = stablehlo.slice %[[V_6]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// CHECK:     %[[V_40:.*]] = stablehlo.slice %[[V_6]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// CHECK:     %[[V_41:.*]] = stablehlo.slice %[[V_13]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// CHECK:     %[[V_42:.*]] = stablehlo.slice %[[V_13]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// CHECK:     %[[V_43:.*]] = stablehlo.add %[[V_39]], %[[V_41]] : tensor<1x2xf32>
// CHECK:     %[[V_44:.*]] = stablehlo.subtract %[[V_43]], %[[V_41]] : tensor<1x2xf32>
// CHECK:     %[[V_45:.*]] = stablehlo.subtract %[[V_43]], %[[V_44]] : tensor<1x2xf32>
// CHECK:     %[[V_46:.*]] = stablehlo.subtract %[[V_39]], %[[V_44]] : tensor<1x2xf32>
// CHECK:     %[[V_47:.*]] = stablehlo.subtract %[[V_41]], %[[V_45]] : tensor<1x2xf32>
// CHECK:     %[[V_48:.*]] = stablehlo.add %[[V_46]], %[[V_47]] : tensor<1x2xf32>
// CHECK:     %[[V_49:.*]] = stablehlo.add %[[V_40]], %[[V_42]] : tensor<1x2xf32>
// CHECK:     %[[V_50:.*]] = stablehlo.subtract %[[V_49]], %[[V_42]] : tensor<1x2xf32>
// CHECK:     %[[V_51:.*]] = stablehlo.subtract %[[V_49]], %[[V_50]] : tensor<1x2xf32>
// CHECK:     %[[V_52:.*]] = stablehlo.subtract %[[V_40]], %[[V_50]] : tensor<1x2xf32>
// CHECK:     %[[V_53:.*]] = stablehlo.subtract %[[V_42]], %[[V_51]] : tensor<1x2xf32>
// CHECK:     %[[V_54:.*]] = stablehlo.add %[[V_52]], %[[V_53]] : tensor<1x2xf32>
// CHECK:     %[[V_55:.*]] = stablehlo.add %[[V_43]], %[[V_49]] : tensor<1x2xf32>
// CHECK:     %[[V_56:.*]] = stablehlo.subtract %[[V_55]], %[[V_43]] : tensor<1x2xf32>
// CHECK:     %[[V_57:.*]] = stablehlo.subtract %[[V_49]], %[[V_56]] : tensor<1x2xf32>
// CHECK:     %[[V_58:.*]] = stablehlo.add %[[V_48]], %[[V_54]] : tensor<1x2xf32>
// CHECK:     %[[V_59:.*]] = stablehlo.add %[[V_58]], %[[V_57]] : tensor<1x2xf32>
// CHECK:     %[[V_60:.*]] = stablehlo.add %[[V_55]], %[[V_59]] : tensor<1x2xf32>
// CHECK:     %[[V_61:.*]] = stablehlo.subtract %[[V_60]], %[[V_55]] : tensor<1x2xf32>
// CHECK:     %[[V_62:.*]] = stablehlo.subtract %[[V_59]], %[[V_61]] : tensor<1x2xf32>
// CHECK:     %[[V_63:.*]] = stablehlo.concatenate %[[V_60]], %[[V_62]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK:     %[[V_64:.*]] = stablehlo.concatenate %[[V_6]], %[[V_6]], dim = 1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
// CHECK:     %[[V_65:.*]] = stablehlo.concatenate %[[V_13]], %[[V_13]], dim = 1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
// CHECK:     %[[V_66:.*]] = stablehlo.slice %[[V_64]] [0:1, 0:4] : (tensor<2x4xf32>) -> tensor<1x4xf32>
// CHECK:     %[[V_67:.*]] = stablehlo.slice %[[V_64]] [1:2, 0:4] : (tensor<2x4xf32>) -> tensor<1x4xf32>
// CHECK:     %[[V_68:.*]] = stablehlo.slice %[[V_65]] [0:1, 0:4] : (tensor<2x4xf32>) -> tensor<1x4xf32>
// CHECK:     %[[V_69:.*]] = stablehlo.slice %[[V_65]] [1:2, 0:4] : (tensor<2x4xf32>) -> tensor<1x4xf32>
// CHECK:     %[[V_70:.*]] = stablehlo.add %[[V_66]], %[[V_68]] : tensor<1x4xf32>
// CHECK:     %[[V_71:.*]] = stablehlo.subtract %[[V_70]], %[[V_68]] : tensor<1x4xf32>
// CHECK:     %[[V_72:.*]] = stablehlo.subtract %[[V_70]], %[[V_71]] : tensor<1x4xf32>
// CHECK:     %[[V_73:.*]] = stablehlo.subtract %[[V_66]], %[[V_71]] : tensor<1x4xf32>
// CHECK:     %[[V_74:.*]] = stablehlo.subtract %[[V_68]], %[[V_72]] : tensor<1x4xf32>
// CHECK:     %[[V_75:.*]] = stablehlo.add %[[V_73]], %[[V_74]] : tensor<1x4xf32>
// CHECK:     %[[V_76:.*]] = stablehlo.add %[[V_67]], %[[V_69]] : tensor<1x4xf32>
// CHECK:     %[[V_77:.*]] = stablehlo.subtract %[[V_76]], %[[V_69]] : tensor<1x4xf32>
// CHECK:     %[[V_78:.*]] = stablehlo.subtract %[[V_76]], %[[V_77]] : tensor<1x4xf32>
// CHECK:     %[[V_79:.*]] = stablehlo.subtract %[[V_67]], %[[V_77]] : tensor<1x4xf32>
// CHECK:     %[[V_80:.*]] = stablehlo.subtract %[[V_69]], %[[V_78]] : tensor<1x4xf32>
// CHECK:     %[[V_81:.*]] = stablehlo.add %[[V_79]], %[[V_80]] : tensor<1x4xf32>
// CHECK:     %[[V_82:.*]] = stablehlo.add %[[V_70]], %[[V_76]] : tensor<1x4xf32>
// CHECK:     %[[V_83:.*]] = stablehlo.subtract %[[V_82]], %[[V_70]] : tensor<1x4xf32>
// CHECK:     %[[V_84:.*]] = stablehlo.subtract %[[V_76]], %[[V_83]] : tensor<1x4xf32>
// CHECK:     %[[V_85:.*]] = stablehlo.add %[[V_75]], %[[V_81]] : tensor<1x4xf32>
// CHECK:     %[[V_86:.*]] = stablehlo.add %[[V_85]], %[[V_84]] : tensor<1x4xf32>
// CHECK:     %[[V_87:.*]] = stablehlo.add %[[V_82]], %[[V_86]] : tensor<1x4xf32>
// CHECK:     %[[V_88:.*]] = stablehlo.subtract %[[V_87]], %[[V_82]] : tensor<1x4xf32>
// CHECK:     %[[V_89:.*]] = stablehlo.subtract %[[V_86]], %[[V_88]] : tensor<1x4xf32>
// CHECK:     %[[V_90:.*]] = stablehlo.concatenate %[[V_87]], %[[V_89]], dim = 0 : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<2x4xf32>
// CHECK:     %[[V_91:.*]] = stablehlo.convert %[[V_90]] : (tensor<2x4xf32>) -> tensor<2x4xf64>
// CHECK:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK:     %[[V_92:.*]] = stablehlo.reduce(%[[V_91]] init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x4xf64>, tensor<f64>) -> tensor<4xf64>
// CHECK:     return %[[V_92]] : tensor<4xf64>
  %0 = stablehlo.add %arg0, %arg1 : tensor<2xf64>
  %1 = stablehlo.add %arg0, %arg1 : tensor<2xf64>
  %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<2xf64>, tensor<2xf64>) -> tensor<4xf64>
  return %2 : tensor<4xf64>
}
