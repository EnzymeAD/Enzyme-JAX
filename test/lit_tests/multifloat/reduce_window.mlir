// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @reduce_window(%arg0: tensor<4x2x2xf64>) -> tensor<4x2x2xf64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{
    base_dilations = array<i64: 1, 1, 1>,
    padding = dense<[[3, 0], [0, 0], [0, 0]]> : tensor<3x2xi64>,
    window_dilations = array<i64: 1, 1, 1>,
    window_dimensions = array<i64: 4, 1, 1>,
    window_strides = array<i64: 1, 1, 1>
  }> ({
  ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
    %1 = stablehlo.add %arg1, %arg2 : tensor<f64>
    stablehlo.return %1 : tensor<f64>
  }) : (tensor<4x2x2xf64>, tensor<f64>) -> tensor<4x2x2xf64>
  return %0 : tensor<4x2x2xf64>
}

// CHECK-LABEL: func.func @reduce_window
// CHECK:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<4x2x2xf64>) -> tensor<4x2x2xf32>
// CHECK:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<4x2x2xf32>) -> tensor<4x2x2xf64>
// CHECK:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<4x2x2xf64>
// CHECK:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<4x2x2xf64>) -> tensor<4x2x2xf32>
// CHECK:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<4x2x2xf32>) -> tensor<1x4x2x2xf32>
// CHECK:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<4x2x2xf32>) -> tensor<1x4x2x2xf32>
// CHECK:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 0 : (tensor<1x4x2x2xf32>, tensor<1x4x2x2xf32>) -> tensor<2x4x2x2xf32>
// CHECK:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<1xf32>
// CHECK:     %[[CST_0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<1xf32>
// CHECK:     %[[V_7:.*]] = stablehlo.concatenate %[[CST]], %[[CST_0]], dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
// CHECK:     %[[CST_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:     %[[V_8:.*]] = stablehlo.pad %[[V_6]], %[[CST_1]], low = [0, 3, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 0, 0] : (tensor<2x4x2x2xf32>, tensor<f32>) -> tensor<2x7x2x2xf32>
// CHECK:     %[[V_9:.*]] = stablehlo.slice %[[V_8]] [0:2, 0:4, 0:2, 0:2] : (tensor<2x7x2x2xf32>) -> tensor<2x4x2x2xf32>
// CHECK:     %[[V_10:.*]] = stablehlo.slice %[[V_8]] [0:2, 1:5, 0:2, 0:2] : (tensor<2x7x2x2xf32>) -> tensor<2x4x2x2xf32>
// CHECK:     %[[V_11:.*]] = stablehlo.slice %[[V_9]] [0:1, 0:4, 0:2, 0:2] : (tensor<2x4x2x2xf32>) -> tensor<1x4x2x2xf32>
// CHECK:     %[[V_12:.*]] = stablehlo.slice %[[V_9]] [1:2, 0:4, 0:2, 0:2] : (tensor<2x4x2x2xf32>) -> tensor<1x4x2x2xf32>
// CHECK:     %[[V_13:.*]] = stablehlo.slice %[[V_10]] [0:1, 0:4, 0:2, 0:2] : (tensor<2x4x2x2xf32>) -> tensor<1x4x2x2xf32>
// CHECK:     %[[V_14:.*]] = stablehlo.slice %[[V_10]] [1:2, 0:4, 0:2, 0:2] : (tensor<2x4x2x2xf32>) -> tensor<1x4x2x2xf32>
// CHECK:     %[[V_15:.*]] = stablehlo.add %[[V_11]], %[[V_13]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_16:.*]] = stablehlo.subtract %[[V_15]], %[[V_13]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_17:.*]] = stablehlo.subtract %[[V_15]], %[[V_16]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_18:.*]] = stablehlo.subtract %[[V_11]], %[[V_16]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_19:.*]] = stablehlo.subtract %[[V_13]], %[[V_17]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_20:.*]] = stablehlo.add %[[V_18]], %[[V_19]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_21:.*]] = stablehlo.add %[[V_12]], %[[V_14]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_22:.*]] = stablehlo.subtract %[[V_21]], %[[V_14]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_23:.*]] = stablehlo.subtract %[[V_21]], %[[V_22]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_24:.*]] = stablehlo.subtract %[[V_12]], %[[V_22]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_25:.*]] = stablehlo.subtract %[[V_14]], %[[V_23]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_26:.*]] = stablehlo.add %[[V_24]], %[[V_25]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_27:.*]] = stablehlo.add %[[V_15]], %[[V_21]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_28:.*]] = stablehlo.subtract %[[V_27]], %[[V_15]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_29:.*]] = stablehlo.subtract %[[V_21]], %[[V_28]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_30:.*]] = stablehlo.add %[[V_20]], %[[V_26]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_31:.*]] = stablehlo.add %[[V_30]], %[[V_29]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_32:.*]] = stablehlo.add %[[V_27]], %[[V_31]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_33:.*]] = stablehlo.subtract %[[V_32]], %[[V_27]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_34:.*]] = stablehlo.subtract %[[V_31]], %[[V_33]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_35:.*]] = stablehlo.concatenate %[[V_32]], %[[V_34]], dim = 0 : (tensor<1x4x2x2xf32>, tensor<1x4x2x2xf32>) -> tensor<2x4x2x2xf32>
// CHECK:     %[[V_36:.*]] = stablehlo.slice %[[V_8]] [0:2, 2:6, 0:2, 0:2] : (tensor<2x7x2x2xf32>) -> tensor<2x4x2x2xf32>
// CHECK:     %[[V_37:.*]] = stablehlo.slice %[[V_36]] [0:1, 0:4, 0:2, 0:2] : (tensor<2x4x2x2xf32>) -> tensor<1x4x2x2xf32>
// CHECK:     %[[V_38:.*]] = stablehlo.slice %[[V_36]] [1:2, 0:4, 0:2, 0:2] : (tensor<2x4x2x2xf32>) -> tensor<1x4x2x2xf32>
// CHECK:     %[[V_39:.*]] = stablehlo.add %[[V_32]], %[[V_37]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_40:.*]] = stablehlo.subtract %[[V_39]], %[[V_37]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_41:.*]] = stablehlo.subtract %[[V_39]], %[[V_40]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_42:.*]] = stablehlo.subtract %[[V_32]], %[[V_40]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_43:.*]] = stablehlo.subtract %[[V_37]], %[[V_41]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_44:.*]] = stablehlo.add %[[V_42]], %[[V_43]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_45:.*]] = stablehlo.add %[[V_34]], %[[V_38]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_46:.*]] = stablehlo.subtract %[[V_45]], %[[V_38]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_47:.*]] = stablehlo.subtract %[[V_45]], %[[V_46]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_48:.*]] = stablehlo.subtract %[[V_34]], %[[V_46]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_49:.*]] = stablehlo.subtract %[[V_38]], %[[V_47]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_50:.*]] = stablehlo.add %[[V_48]], %[[V_49]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_51:.*]] = stablehlo.add %[[V_39]], %[[V_45]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_52:.*]] = stablehlo.subtract %[[V_51]], %[[V_39]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_53:.*]] = stablehlo.subtract %[[V_45]], %[[V_52]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_54:.*]] = stablehlo.add %[[V_44]], %[[V_50]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_55:.*]] = stablehlo.add %[[V_54]], %[[V_53]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_56:.*]] = stablehlo.add %[[V_51]], %[[V_55]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_57:.*]] = stablehlo.subtract %[[V_56]], %[[V_51]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_58:.*]] = stablehlo.subtract %[[V_55]], %[[V_57]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_59:.*]] = stablehlo.concatenate %[[V_56]], %[[V_58]], dim = 0 : (tensor<1x4x2x2xf32>, tensor<1x4x2x2xf32>) -> tensor<2x4x2x2xf32>
// CHECK:     %[[V_60:.*]] = stablehlo.slice %[[V_8]] [0:2, 3:7, 0:2, 0:2] : (tensor<2x7x2x2xf32>) -> tensor<2x4x2x2xf32>
// CHECK:     %[[V_61:.*]] = stablehlo.slice %[[V_60]] [0:1, 0:4, 0:2, 0:2] : (tensor<2x4x2x2xf32>) -> tensor<1x4x2x2xf32>
// CHECK:     %[[V_62:.*]] = stablehlo.slice %[[V_60]] [1:2, 0:4, 0:2, 0:2] : (tensor<2x4x2x2xf32>) -> tensor<1x4x2x2xf32>
// CHECK:     %[[V_63:.*]] = stablehlo.add %[[V_56]], %[[V_61]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_64:.*]] = stablehlo.subtract %[[V_63]], %[[V_61]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_65:.*]] = stablehlo.subtract %[[V_63]], %[[V_64]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_66:.*]] = stablehlo.subtract %[[V_56]], %[[V_64]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_67:.*]] = stablehlo.subtract %[[V_61]], %[[V_65]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_68:.*]] = stablehlo.add %[[V_66]], %[[V_67]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_69:.*]] = stablehlo.add %[[V_58]], %[[V_62]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_70:.*]] = stablehlo.subtract %[[V_69]], %[[V_62]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_71:.*]] = stablehlo.subtract %[[V_69]], %[[V_70]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_72:.*]] = stablehlo.subtract %[[V_58]], %[[V_70]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_73:.*]] = stablehlo.subtract %[[V_62]], %[[V_71]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_74:.*]] = stablehlo.add %[[V_72]], %[[V_73]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_75:.*]] = stablehlo.add %[[V_63]], %[[V_69]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_76:.*]] = stablehlo.subtract %[[V_75]], %[[V_63]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_77:.*]] = stablehlo.subtract %[[V_69]], %[[V_76]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_78:.*]] = stablehlo.add %[[V_68]], %[[V_74]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_79:.*]] = stablehlo.add %[[V_78]], %[[V_77]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_80:.*]] = stablehlo.add %[[V_75]], %[[V_79]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_81:.*]] = stablehlo.subtract %[[V_80]], %[[V_75]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_82:.*]] = stablehlo.subtract %[[V_79]], %[[V_81]] : tensor<1x4x2x2xf32>
// CHECK:     %[[V_83:.*]] = stablehlo.concatenate %[[V_80]], %[[V_82]], dim = 0 : (tensor<1x4x2x2xf32>, tensor<1x4x2x2xf32>) -> tensor<2x4x2x2xf32>
// CHECK:     %[[V_84:.*]] = stablehlo.convert %[[V_83]] : (tensor<2x4x2x2xf32>) -> tensor<2x4x2x2xf64>
// CHECK:     %[[CST_2:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK:     %[[V_85:.*]] = stablehlo.reduce(%[[V_84]] init: %[[CST_2]]) applies stablehlo.add across dimensions = [0] : (tensor<2x4x2x2xf64>, tensor<f64>) -> tensor<4x2x2xf64>
// CHECK:     return %[[V_85]] : tensor<4x2x2xf64>

func.func @main() attributes {enzyme.no_multifloat} {
  %cst = stablehlo.constant dense<[[[1.0, 2.0], [3.0, 4.0]],
                                  [[5.0, 6.0], [7.0, 8.0]],
                                  [[9.0, 10.0], [11.0, 12.0]],
                                  [[13.0, 14.0], [15.0, 16.0]]]> : tensor<4x2x2xf64>
                                  
  %expected = stablehlo.constant dense<[[[1.0, 2.0], [3.0, 4.0]],
                                       [[6.0, 8.0], [10.0, 12.0]],
                                       [[15.0, 18.0], [21.0, 24.0]],
                                       [[28.0, 32.0], [36.0, 40.0]]]> : tensor<4x2x2xf64>
                                       
  %res = func.call @reduce_window(%cst) : (tensor<4x2x2xf64>) -> tensor<4x2x2xf64>
  
  "check.expect_almost_eq"(%res, %expected) : (tensor<4x2x2xf64>, tensor<4x2x2xf64>) -> ()
  return
}

// CHECK-LABEL: func.func @main
// CHECK:     %[[CST:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<4x2x2xf64>
// CHECK:     %[[CST_0:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<4x2x2xf64>
// CHECK:     %[[V_0:.*]] = call @reduce_window(%[[CST]]) : (tensor<4x2x2xf64>) -> tensor<4x2x2xf64>
// CHECK:     check.expect_almost_eq %[[V_0]], %[[CST_0]] : tensor<4x2x2xf64>
// CHECK:     return
