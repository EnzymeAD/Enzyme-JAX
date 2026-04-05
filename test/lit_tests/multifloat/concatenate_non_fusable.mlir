// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | FileCheck %s

func.func @non_fusable_concat(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>, %arg2: tensor<2xf64>) -> (tensor<4xf64>, tensor<2xf64>) {
  // CHECK-LABEL: @non_fusable_concat
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
// CHECK:     %[[V_14:.*]] = stablehlo.convert %arg2 : (tensor<2xf64>) -> tensor<2xf32>
// CHECK:     %[[V_15:.*]] = stablehlo.convert %[[V_14]] : (tensor<2xf32>) -> tensor<2xf64>
// CHECK:     %[[V_16:.*]] = stablehlo.subtract %arg2, %[[V_15]] : tensor<2xf64>
// CHECK:     %[[V_17:.*]] = stablehlo.convert %[[V_16]] : (tensor<2xf64>) -> tensor<2xf32>
// CHECK:     %[[V_18:.*]] = stablehlo.reshape %[[V_14]] : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK:     %[[V_19:.*]] = stablehlo.reshape %[[V_17]] : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK:     %[[V_20:.*]] = stablehlo.concatenate %[[V_18]], %[[V_19]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK:     %[[V_21:.*]] = stablehlo.slice %[[V_6]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// CHECK:     %[[V_22:.*]] = stablehlo.slice %[[V_6]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// CHECK:     %[[V_23:.*]] = stablehlo.slice %[[V_13]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// CHECK:     %[[V_24:.*]] = stablehlo.slice %[[V_13]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// CHECK:     %[[V_25:.*]] = stablehlo.add %[[V_21]], %[[V_23]] : tensor<1x2xf32>
// CHECK:     %[[V_26:.*]] = stablehlo.subtract %[[V_25]], %[[V_23]] : tensor<1x2xf32>
// CHECK:     %[[V_27:.*]] = stablehlo.subtract %[[V_25]], %[[V_26]] : tensor<1x2xf32>
// CHECK:     %[[V_28:.*]] = stablehlo.subtract %[[V_21]], %[[V_26]] : tensor<1x2xf32>
// CHECK:     %[[V_29:.*]] = stablehlo.subtract %[[V_23]], %[[V_27]] : tensor<1x2xf32>
// CHECK:     %[[V_30:.*]] = stablehlo.add %[[V_28]], %[[V_29]] : tensor<1x2xf32>
// CHECK:     %[[V_31:.*]] = stablehlo.add %[[V_22]], %[[V_24]] : tensor<1x2xf32>
// CHECK:     %[[V_32:.*]] = stablehlo.subtract %[[V_31]], %[[V_24]] : tensor<1x2xf32>
// CHECK:     %[[V_33:.*]] = stablehlo.subtract %[[V_31]], %[[V_32]] : tensor<1x2xf32>
// CHECK:     %[[V_34:.*]] = stablehlo.subtract %[[V_22]], %[[V_32]] : tensor<1x2xf32>
// CHECK:     %[[V_35:.*]] = stablehlo.subtract %[[V_24]], %[[V_33]] : tensor<1x2xf32>
// CHECK:     %[[V_36:.*]] = stablehlo.add %[[V_34]], %[[V_35]] : tensor<1x2xf32>
// CHECK:     %[[V_37:.*]] = stablehlo.add %[[V_25]], %[[V_31]] : tensor<1x2xf32>
// CHECK:     %[[V_38:.*]] = stablehlo.subtract %[[V_37]], %[[V_25]] : tensor<1x2xf32>
// CHECK:     %[[V_39:.*]] = stablehlo.subtract %[[V_31]], %[[V_38]] : tensor<1x2xf32>
// CHECK:     %[[V_40:.*]] = stablehlo.add %[[V_30]], %[[V_36]] : tensor<1x2xf32>
// CHECK:     %[[V_41:.*]] = stablehlo.add %[[V_40]], %[[V_39]] : tensor<1x2xf32>
// CHECK:     %[[V_42:.*]] = stablehlo.add %[[V_37]], %[[V_41]] : tensor<1x2xf32>
// CHECK:     %[[V_43:.*]] = stablehlo.subtract %[[V_42]], %[[V_37]] : tensor<1x2xf32>
// CHECK:     %[[V_44:.*]] = stablehlo.subtract %[[V_41]], %[[V_43]] : tensor<1x2xf32>
// CHECK:     %[[V_45:.*]] = stablehlo.concatenate %[[V_42]], %[[V_44]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK:     %[[V_46:.*]] = stablehlo.slice %[[V_6]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// CHECK:     %[[V_47:.*]] = stablehlo.slice %[[V_6]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// CHECK:     %[[V_48:.*]] = stablehlo.slice %[[V_20]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// CHECK:     %[[V_49:.*]] = stablehlo.slice %[[V_20]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// CHECK:     %[[V_50:.*]] = stablehlo.add %[[V_46]], %[[V_48]] : tensor<1x2xf32>
// CHECK:     %[[V_51:.*]] = stablehlo.subtract %[[V_50]], %[[V_48]] : tensor<1x2xf32>
// CHECK:     %[[V_52:.*]] = stablehlo.subtract %[[V_50]], %[[V_51]] : tensor<1x2xf32>
// CHECK:     %[[V_53:.*]] = stablehlo.subtract %[[V_46]], %[[V_51]] : tensor<1x2xf32>
// CHECK:     %[[V_54:.*]] = stablehlo.subtract %[[V_48]], %[[V_52]] : tensor<1x2xf32>
// CHECK:     %[[V_55:.*]] = stablehlo.add %[[V_53]], %[[V_54]] : tensor<1x2xf32>
// CHECK:     %[[V_56:.*]] = stablehlo.add %[[V_47]], %[[V_49]] : tensor<1x2xf32>
// CHECK:     %[[V_57:.*]] = stablehlo.subtract %[[V_56]], %[[V_49]] : tensor<1x2xf32>
// CHECK:     %[[V_58:.*]] = stablehlo.subtract %[[V_56]], %[[V_57]] : tensor<1x2xf32>
// CHECK:     %[[V_59:.*]] = stablehlo.subtract %[[V_47]], %[[V_57]] : tensor<1x2xf32>
// CHECK:     %[[V_60:.*]] = stablehlo.subtract %[[V_49]], %[[V_58]] : tensor<1x2xf32>
// CHECK:     %[[V_61:.*]] = stablehlo.add %[[V_59]], %[[V_60]] : tensor<1x2xf32>
// CHECK:     %[[V_62:.*]] = stablehlo.add %[[V_50]], %[[V_56]] : tensor<1x2xf32>
// CHECK:     %[[V_63:.*]] = stablehlo.subtract %[[V_62]], %[[V_50]] : tensor<1x2xf32>
// CHECK:     %[[V_64:.*]] = stablehlo.subtract %[[V_56]], %[[V_63]] : tensor<1x2xf32>
// CHECK:     %[[V_65:.*]] = stablehlo.add %[[V_55]], %[[V_61]] : tensor<1x2xf32>
// CHECK:     %[[V_66:.*]] = stablehlo.add %[[V_65]], %[[V_64]] : tensor<1x2xf32>
// CHECK:     %[[V_67:.*]] = stablehlo.add %[[V_62]], %[[V_66]] : tensor<1x2xf32>
// CHECK:     %[[V_68:.*]] = stablehlo.subtract %[[V_67]], %[[V_62]] : tensor<1x2xf32>
// CHECK:     %[[V_69:.*]] = stablehlo.subtract %[[V_66]], %[[V_68]] : tensor<1x2xf32>
// CHECK:     %[[V_70:.*]] = stablehlo.concatenate %[[V_67]], %[[V_69]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK:     %[[V_71:.*]] = stablehlo.concatenate %[[V_45]], %[[V_70]], dim = 1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
// CHECK:     %[[V_72:.*]] = stablehlo.convert %[[V_71]] : (tensor<2x4xf32>) -> tensor<2x4xf64>
// CHECK:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK:     %[[V_73:.*]] = stablehlo.reduce(%[[V_72]] init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x4xf64>, tensor<f64>) -> tensor<4xf64>
// CHECK:     %[[V_74:.*]] = stablehlo.convert %[[V_45]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
// CHECK:     %[[V_75:.*]] = stablehlo.reduce(%[[V_74]] init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>

  %0 = stablehlo.add %arg0, %arg1 : tensor<2xf64>
  %1 = stablehlo.add %arg0, %arg2 : tensor<2xf64>
  
  // %0 is used in concatenate AND returned!
  %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<2xf64>, tensor<2xf64>) -> tensor<4xf64>
  
  return %2, %0 : tensor<4xf64>, tensor<2xf64>
}
