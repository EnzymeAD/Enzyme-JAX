// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first dot-general-to-reduce=false" %s | FileCheck --check-prefix=FIRST_OZAKI %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last dot-general-to-reduce=false" %s | FileCheck --check-prefix=LAST_OZAKI %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first dot-general-to-reduce=false" %s | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last dot-general-to-reduce=false" %s | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @dot_mini_first(%arg0: tensor<2xf64>, %arg1: tensor<2x2xf64>) -> tensor<2xf64> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [0] : (tensor<2xf64>, tensor<2x2xf64>) -> tensor<2xf64>
  return %0 : tensor<2xf64>
}

// FIRST_OZAKI-LABEL: func.func @dot_mini_first
// FIRST_OZAKI: %[[V0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// FIRST_OZAKI: %[[V1:.*]] = stablehlo.convert %[[V0]] : (tensor<2xf32>) -> tensor<2xf64>
// FIRST_OZAKI: %[[V2:.*]] = stablehlo.subtract %arg0, %[[V1]] : tensor<2xf64>
// FIRST_OZAKI: %[[V3:.*]] = stablehlo.convert %[[V2]] : (tensor<2xf64>) -> tensor<2xf32>
// FIRST_OZAKI: %[[V4:.*]] = stablehlo.reshape %[[V0]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST_OZAKI: %[[V5:.*]] = stablehlo.reshape %[[V3]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST_OZAKI: %[[V6:.*]] = stablehlo.concatenate %[[V4]], %[[V5]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST_OZAKI: %[[V7:.*]] = stablehlo.convert %arg1 : (tensor<2x2xf64>) -> tensor<2x2xf32>
// FIRST_OZAKI: %[[V8:.*]] = stablehlo.convert %[[V7]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
// FIRST_OZAKI: %[[V9:.*]] = stablehlo.subtract %arg1, %[[V8]] : tensor<2x2xf64>
// FIRST_OZAKI: %[[V10:.*]] = stablehlo.convert %[[V9]] : (tensor<2x2xf64>) -> tensor<2x2xf32>
// FIRST_OZAKI: %[[V11:.*]] = stablehlo.reshape %[[V7]] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
// FIRST_OZAKI: %[[V12:.*]] = stablehlo.reshape %[[V10]] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
// FIRST_OZAKI: %[[V13:.*]] = stablehlo.concatenate %[[V11]], %[[V12]], dim = 0 : (tensor<1x2x2xf32>, tensor<1x2x2xf32>) -> tensor<2x2x2xf32>
// FIRST_OZAKI: %[[V14:.*]] = stablehlo.slice %[[V6]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST_OZAKI: %[[V15:.*]] = stablehlo.slice %[[V6]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST_OZAKI: %[[V16:.*]] = stablehlo.slice %[[V13]] [0:1, 0:2, 0:2] : (tensor<2x2x2xf32>) -> tensor<1x2x2xf32>
// FIRST_OZAKI: %[[V17:.*]] = stablehlo.slice %[[V13]] [1:2, 0:2, 0:2] : (tensor<2x2x2xf32>) -> tensor<1x2x2xf32>
// FIRST_OZAKI: %[[V18:.*]] = stablehlo.reshape %[[V14]] : (tensor<1x2xf32>) -> tensor<2xf32>
// FIRST_OZAKI: %[[V19:.*]] = stablehlo.reshape %[[V15]] : (tensor<1x2xf32>) -> tensor<2xf32>
// FIRST_OZAKI: %[[V20:.*]] = stablehlo.reshape %[[V16]] : (tensor<1x2x2xf32>) -> tensor<2x2xf32>
// FIRST_OZAKI: %[[V21:.*]] = stablehlo.reshape %[[V17]] : (tensor<1x2x2xf32>) -> tensor<2x2xf32>
// FIRST_OZAKI: %[[V22:.*]] = stablehlo.abs %[[V18]] : tensor<2xf32>
// FIRST_OZAKI: %[[Vcst:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// FIRST_OZAKI: %[[V23:.*]] = stablehlo.reduce(%[[V22]] init: %[[Vcst]]) applies stablehlo.maximum across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
// FIRST_OZAKI: %[[V24:.*]] = stablehlo.abs %[[V20]] : tensor<2x2xf32>
// FIRST_OZAKI: %[[Vcst_0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// FIRST_OZAKI: %[[V25:.*]] = stablehlo.reduce(%[[V24]] init: %[[Vcst_0]]) applies stablehlo.maximum across dimensions = [0, 1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<f32>
// FIRST_OZAKI: %[[Vcst_1:.*]] = stablehlo.constant dense<0.693147182> : tensor<f32>
// FIRST_OZAKI: %[[V26:.*]] = stablehlo.log %[[V23]] : tensor<f32>
// FIRST_OZAKI: %[[V27:.*]] = stablehlo.divide %[[V26]], %[[Vcst_1]] : tensor<f32>
// FIRST_OZAKI: %[[V28:.*]] = stablehlo.ceil %[[V27]] : tensor<f32>
// FIRST_OZAKI: %[[V29:.*]] = stablehlo.multiply %[[V28]], %[[Vcst_1]] : tensor<f32>
// FIRST_OZAKI: %[[V30:.*]] = stablehlo.exponential %[[V29]] : tensor<f32>
// FIRST_OZAKI: %[[V31:.*]] = stablehlo.broadcast_in_dim %[[V30]], dims = [] : (tensor<f32>) -> tensor<2xf32>
// FIRST_OZAKI: %[[V32:.*]] = stablehlo.log %[[V25]] : tensor<f32>
// FIRST_OZAKI: %[[V33:.*]] = stablehlo.divide %[[V32]], %[[Vcst_1]] : tensor<f32>
// FIRST_OZAKI: %[[V34:.*]] = stablehlo.ceil %[[V33]] : tensor<f32>
// FIRST_OZAKI: %[[V35:.*]] = stablehlo.multiply %[[V34]], %[[Vcst_1]] : tensor<f32>
// FIRST_OZAKI: %[[V36:.*]] = stablehlo.exponential %[[V35]] : tensor<f32>
// FIRST_OZAKI: %[[V37:.*]] = stablehlo.broadcast_in_dim %[[V36]], dims = [] : (tensor<f32>) -> tensor<2x2xf32>
// FIRST_OZAKI: %[[V38:.*]] = stablehlo.divide %[[V18]], %[[V31]] : tensor<2xf32>
// FIRST_OZAKI: %[[V39:.*]] = stablehlo.divide %[[V20]], %[[V37]] : tensor<2x2xf32>
// FIRST_OZAKI: %[[Vcst_2:.*]] = stablehlo.constant dense<4.096000e+03> : tensor<2xf32>
// FIRST_OZAKI: %[[V40:.*]] = stablehlo.multiply %[[V38]], %[[Vcst_2]] : tensor<2xf32>
// FIRST_OZAKI: %[[V41:.*]] = stablehlo.floor %[[V40]] : tensor<2xf32>
// FIRST_OZAKI: %[[V42:.*]] = stablehlo.subtract %[[V40]], %[[V41]] : tensor<2xf32>
// FIRST_OZAKI: %[[Vcst_3:.*]] = stablehlo.constant dense<4.096000e+03> : tensor<2x2xf32>
// FIRST_OZAKI: %[[V43:.*]] = stablehlo.multiply %[[V39]], %[[Vcst_3]] : tensor<2x2xf32>
// FIRST_OZAKI: %[[V44:.*]] = stablehlo.floor %[[V43]] : tensor<2x2xf32>
// FIRST_OZAKI: %[[V45:.*]] = stablehlo.subtract %[[V43]], %[[V44]] : tensor<2x2xf32>
// FIRST_OZAKI: %[[V46:.*]] = stablehlo.dot_general %[[V41]], %[[V44]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2x2xf32>) -> tensor<2xf32>
// FIRST_OZAKI: %[[V47:.*]] = stablehlo.dot_general %[[V41]], %[[V45]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2x2xf32>) -> tensor<2xf32>
// FIRST_OZAKI: %[[V48:.*]] = stablehlo.dot_general %[[V42]], %[[V44]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2x2xf32>) -> tensor<2xf32>
// FIRST_OZAKI: %[[V49:.*]] = stablehlo.dot_general %[[V42]], %[[V45]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2x2xf32>) -> tensor<2xf32>
// FIRST_OZAKI: %[[V50:.*]] = stablehlo.multiply %[[V30]], %[[V36]] : tensor<f32>
// FIRST_OZAKI: %[[Vcst_4:.*]] = stablehlo.constant dense<0x4B800000> : tensor<f32>
// FIRST_OZAKI: %[[V51:.*]] = stablehlo.divide %[[V50]], %[[Vcst_4]] : tensor<f32>
// FIRST_OZAKI: %[[V52:.*]] = stablehlo.broadcast_in_dim %[[V51]], dims = [] : (tensor<f32>) -> tensor<2xf32>
// FIRST_OZAKI: %[[V53:.*]] = stablehlo.multiply %[[V46]], %[[V52]] : tensor<2xf32>
// FIRST_OZAKI: %[[V54:.*]] = stablehlo.multiply %[[V47]], %[[V52]] : tensor<2xf32>
// FIRST_OZAKI: %[[V55:.*]] = stablehlo.multiply %[[V48]], %[[V52]] : tensor<2xf32>
// FIRST_OZAKI: %[[V56:.*]] = stablehlo.multiply %[[V49]], %[[V52]] : tensor<2xf32>
// FIRST_OZAKI: %[[V57:.*]] = stablehlo.reshape %[[V53]] : (tensor<2xf32>) -> tensor<2x1xf32>
// FIRST_OZAKI: %[[V58:.*]] = stablehlo.reshape %[[V54]] : (tensor<2xf32>) -> tensor<2x1xf32>
// FIRST_OZAKI: %[[V59:.*]] = stablehlo.reshape %[[V55]] : (tensor<2xf32>) -> tensor<2x1xf32>
// FIRST_OZAKI: %[[V60:.*]] = stablehlo.reshape %[[V56]] : (tensor<2xf32>) -> tensor<2x1xf32>
// FIRST_OZAKI: %[[V61:.*]] = stablehlo.concatenate %[[V57]], %[[V58]], %[[V59]], %[[V60]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x4xf32>
// FIRST_OZAKI: %[[Vcst_5:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// FIRST_OZAKI: %[[V62:.*]] = stablehlo.reduce(%[[V61]] init: %[[Vcst_5]]) applies stablehlo.add across dimensions = [1] : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
// FIRST_OZAKI: %[[V63:.*]] = stablehlo.dot_general %[[V18]], %[[V21]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2x2xf32>) -> tensor<2xf32>
// FIRST_OZAKI: %[[V64:.*]] = stablehlo.dot_general %[[V19]], %[[V20]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2x2xf32>) -> tensor<2xf32>
// FIRST_OZAKI: %[[V65:.*]] = stablehlo.add %[[V62]], %[[V63]] : tensor<2xf32>
// FIRST_OZAKI: %[[V66:.*]] = stablehlo.subtract %[[V65]], %[[V63]] : tensor<2xf32>
// FIRST_OZAKI: %[[V67:.*]] = stablehlo.subtract %[[V65]], %[[V66]] : tensor<2xf32>
// FIRST_OZAKI: %[[V68:.*]] = stablehlo.subtract %[[V62]], %[[V66]] : tensor<2xf32>
// FIRST_OZAKI: %[[V69:.*]] = stablehlo.subtract %[[V63]], %[[V67]] : tensor<2xf32>
// FIRST_OZAKI: %[[V70:.*]] = stablehlo.add %[[V68]], %[[V69]] : tensor<2xf32>
// FIRST_OZAKI: %[[V71:.*]] = stablehlo.add %[[V65]], %[[V64]] : tensor<2xf32>
// FIRST_OZAKI: %[[V72:.*]] = stablehlo.subtract %[[V71]], %[[V64]] : tensor<2xf32>
// FIRST_OZAKI: %[[V73:.*]] = stablehlo.subtract %[[V71]], %[[V72]] : tensor<2xf32>
// FIRST_OZAKI: %[[V74:.*]] = stablehlo.subtract %[[V65]], %[[V72]] : tensor<2xf32>
// FIRST_OZAKI: %[[V75:.*]] = stablehlo.subtract %[[V64]], %[[V73]] : tensor<2xf32>
// FIRST_OZAKI: %[[V76:.*]] = stablehlo.add %[[V74]], %[[V75]] : tensor<2xf32>
// FIRST_OZAKI: %[[V77:.*]] = stablehlo.add %[[V70]], %[[V76]] : tensor<2xf32>
// FIRST_OZAKI: %[[V78:.*]] = stablehlo.add %[[V71]], %[[V77]] : tensor<2xf32>
// FIRST_OZAKI: %[[V79:.*]] = stablehlo.subtract %[[V78]], %[[V77]] : tensor<2xf32>
// FIRST_OZAKI: %[[V80:.*]] = stablehlo.subtract %[[V78]], %[[V79]] : tensor<2xf32>
// FIRST_OZAKI: %[[V81:.*]] = stablehlo.subtract %[[V71]], %[[V79]] : tensor<2xf32>
// FIRST_OZAKI: %[[V82:.*]] = stablehlo.subtract %[[V77]], %[[V80]] : tensor<2xf32>
// FIRST_OZAKI: %[[V83:.*]] = stablehlo.add %[[V81]], %[[V82]] : tensor<2xf32>
// FIRST_OZAKI: %[[V84:.*]] = stablehlo.reshape %[[V78]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST_OZAKI: %[[V85:.*]] = stablehlo.reshape %[[V83]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST_OZAKI: %[[V86:.*]] = stablehlo.concatenate %[[V84]], %[[V85]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST_OZAKI: %[[V87:.*]] = stablehlo.convert %[[V86]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
// FIRST_OZAKI: %[[Vcst_6:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// FIRST_OZAKI: %[[V88:.*]] = stablehlo.reduce(%[[V87]] init: %[[Vcst_6]]) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
// FIRST_OZAKI: return %[[V88]] : tensor<2xf64>

func.func @dot_mini_last(%arg0: tensor<2xf64>, %arg1: tensor<2x2xf64>) -> tensor<2xf64> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [0] : (tensor<2xf64>, tensor<2x2xf64>) -> tensor<2xf64>
  return %0 : tensor<2xf64>
}

// LAST_OZAKI-LABEL: func.func @dot_mini_last
// LAST_OZAKI: %[[V0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// LAST_OZAKI: %[[V1:.*]] = stablehlo.convert %[[V0]] : (tensor<2xf32>) -> tensor<2xf64>
// LAST_OZAKI: %[[V2:.*]] = stablehlo.subtract %arg0, %[[V1]] : tensor<2xf64>
// LAST_OZAKI: %[[V3:.*]] = stablehlo.convert %[[V2]] : (tensor<2xf64>) -> tensor<2xf32>
// LAST_OZAKI: %[[V4:.*]] = stablehlo.reshape %[[V0]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST_OZAKI: %[[V5:.*]] = stablehlo.reshape %[[V3]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST_OZAKI: %[[V6:.*]] = stablehlo.concatenate %[[V4]], %[[V5]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// LAST_OZAKI: %[[V7:.*]] = stablehlo.convert %arg1 : (tensor<2x2xf64>) -> tensor<2x2xf32>
// LAST_OZAKI: %[[V8:.*]] = stablehlo.convert %[[V7]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
// LAST_OZAKI: %[[V9:.*]] = stablehlo.subtract %arg1, %[[V8]] : tensor<2x2xf64>
// LAST_OZAKI: %[[V10:.*]] = stablehlo.convert %[[V9]] : (tensor<2x2xf64>) -> tensor<2x2xf32>
// LAST_OZAKI: %[[V11:.*]] = stablehlo.reshape %[[V7]] : (tensor<2x2xf32>) -> tensor<2x2x1xf32>
// LAST_OZAKI: %[[V12:.*]] = stablehlo.reshape %[[V10]] : (tensor<2x2xf32>) -> tensor<2x2x1xf32>
// LAST_OZAKI: %[[V13:.*]] = stablehlo.concatenate %[[V11]], %[[V12]], dim = 2 : (tensor<2x2x1xf32>, tensor<2x2x1xf32>) -> tensor<2x2x2xf32>
// LAST_OZAKI: %[[V14:.*]] = stablehlo.slice %[[V6]] [0:2, 0:1] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// LAST_OZAKI: %[[V15:.*]] = stablehlo.slice %[[V6]] [0:2, 1:2] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// LAST_OZAKI: %[[V16:.*]] = stablehlo.slice %[[V13]] [0:2, 0:2, 0:1] : (tensor<2x2x2xf32>) -> tensor<2x2x1xf32>
// LAST_OZAKI: %[[V17:.*]] = stablehlo.slice %[[V13]] [0:2, 0:2, 1:2] : (tensor<2x2x2xf32>) -> tensor<2x2x1xf32>
// LAST_OZAKI: %[[V18:.*]] = stablehlo.reshape %[[V14]] : (tensor<2x1xf32>) -> tensor<2xf32>
// LAST_OZAKI: %[[V19:.*]] = stablehlo.reshape %[[V15]] : (tensor<2x1xf32>) -> tensor<2xf32>
// LAST_OZAKI: %[[V20:.*]] = stablehlo.reshape %[[V16]] : (tensor<2x2x1xf32>) -> tensor<2x2xf32>
// LAST_OZAKI: %[[V21:.*]] = stablehlo.reshape %[[V17]] : (tensor<2x2x1xf32>) -> tensor<2x2xf32>
// LAST_OZAKI: %[[V22:.*]] = stablehlo.abs %[[V18]] : tensor<2xf32>
// LAST_OZAKI: %[[Vcst:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// LAST_OZAKI: %[[V23:.*]] = stablehlo.reduce(%[[V22]] init: %[[Vcst]]) applies stablehlo.maximum across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
// LAST_OZAKI: %[[V24:.*]] = stablehlo.abs %[[V20]] : tensor<2x2xf32>
// LAST_OZAKI: %[[Vcst_0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// LAST_OZAKI: %[[V25:.*]] = stablehlo.reduce(%[[V24]] init: %[[Vcst_0]]) applies stablehlo.maximum across dimensions = [0, 1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<f32>
// LAST_OZAKI: %[[Vcst_1:.*]] = stablehlo.constant dense<0.693147182> : tensor<f32>
// LAST_OZAKI: %[[V26:.*]] = stablehlo.log %[[V23]] : tensor<f32>
// LAST_OZAKI: %[[V27:.*]] = stablehlo.divide %[[V26]], %[[Vcst_1]] : tensor<f32>
// LAST_OZAKI: %[[V28:.*]] = stablehlo.ceil %[[V27]] : tensor<f32>
// LAST_OZAKI: %[[V29:.*]] = stablehlo.multiply %[[V28]], %[[Vcst_1]] : tensor<f32>
// LAST_OZAKI: %[[V30:.*]] = stablehlo.exponential %[[V29]] : tensor<f32>
// LAST_OZAKI: %[[V31:.*]] = stablehlo.broadcast_in_dim %[[V30]], dims = [] : (tensor<f32>) -> tensor<2xf32>
// LAST_OZAKI: %[[V32:.*]] = stablehlo.log %[[V25]] : tensor<f32>
// LAST_OZAKI: %[[V33:.*]] = stablehlo.divide %[[V32]], %[[Vcst_1]] : tensor<f32>
// LAST_OZAKI: %[[V34:.*]] = stablehlo.ceil %[[V33]] : tensor<f32>
// LAST_OZAKI: %[[V35:.*]] = stablehlo.multiply %[[V34]], %[[Vcst_1]] : tensor<f32>
// LAST_OZAKI: %[[V36:.*]] = stablehlo.exponential %[[V35]] : tensor<f32>
// LAST_OZAKI: %[[V37:.*]] = stablehlo.broadcast_in_dim %[[V36]], dims = [] : (tensor<f32>) -> tensor<2x2xf32>
// LAST_OZAKI: %[[V38:.*]] = stablehlo.divide %[[V18]], %[[V31]] : tensor<2xf32>
// LAST_OZAKI: %[[V39:.*]] = stablehlo.divide %[[V20]], %[[V37]] : tensor<2x2xf32>
// LAST_OZAKI: %[[Vcst_2:.*]] = stablehlo.constant dense<4.096000e+03> : tensor<2xf32>
// LAST_OZAKI: %[[V40:.*]] = stablehlo.multiply %[[V38]], %[[Vcst_2]] : tensor<2xf32>
// LAST_OZAKI: %[[V41:.*]] = stablehlo.floor %[[V40]] : tensor<2xf32>
// LAST_OZAKI: %[[V42:.*]] = stablehlo.subtract %[[V40]], %[[V41]] : tensor<2xf32>
// LAST_OZAKI: %[[Vcst_3:.*]] = stablehlo.constant dense<4.096000e+03> : tensor<2x2xf32>
// LAST_OZAKI: %[[V43:.*]] = stablehlo.multiply %[[V39]], %[[Vcst_3]] : tensor<2x2xf32>
// LAST_OZAKI: %[[V44:.*]] = stablehlo.floor %[[V43]] : tensor<2x2xf32>
// LAST_OZAKI: %[[V45:.*]] = stablehlo.subtract %[[V43]], %[[V44]] : tensor<2x2xf32>
// LAST_OZAKI: %[[V46:.*]] = stablehlo.dot_general %[[V41]], %[[V44]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2x2xf32>) -> tensor<2xf32>
// LAST_OZAKI: %[[V47:.*]] = stablehlo.dot_general %[[V41]], %[[V45]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2x2xf32>) -> tensor<2xf32>
// LAST_OZAKI: %[[V48:.*]] = stablehlo.dot_general %[[V42]], %[[V44]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2x2xf32>) -> tensor<2xf32>
// LAST_OZAKI: %[[V49:.*]] = stablehlo.dot_general %[[V42]], %[[V45]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2x2xf32>) -> tensor<2xf32>
// LAST_OZAKI: %[[V50:.*]] = stablehlo.multiply %[[V30]], %[[V36]] : tensor<f32>
// LAST_OZAKI: %[[Vcst_4:.*]] = stablehlo.constant dense<0x4B800000> : tensor<f32>
// LAST_OZAKI: %[[V51:.*]] = stablehlo.divide %[[V50]], %[[Vcst_4]] : tensor<f32>
// LAST_OZAKI: %[[V52:.*]] = stablehlo.broadcast_in_dim %[[V51]], dims = [] : (tensor<f32>) -> tensor<2xf32>
// LAST_OZAKI: %[[V53:.*]] = stablehlo.multiply %[[V46]], %[[V52]] : tensor<2xf32>
// LAST_OZAKI: %[[V54:.*]] = stablehlo.multiply %[[V47]], %[[V52]] : tensor<2xf32>
// LAST_OZAKI: %[[V55:.*]] = stablehlo.multiply %[[V48]], %[[V52]] : tensor<2xf32>
// LAST_OZAKI: %[[V56:.*]] = stablehlo.multiply %[[V49]], %[[V52]] : tensor<2xf32>
// LAST_OZAKI: %[[V57:.*]] = stablehlo.reshape %[[V53]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST_OZAKI: %[[V58:.*]] = stablehlo.reshape %[[V54]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST_OZAKI: %[[V59:.*]] = stablehlo.reshape %[[V55]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST_OZAKI: %[[V60:.*]] = stablehlo.reshape %[[V56]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST_OZAKI: %[[V61:.*]] = stablehlo.concatenate %[[V57]], %[[V58]], %[[V59]], %[[V60]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x4xf32>
// LAST_OZAKI: %[[Vcst_5:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// LAST_OZAKI: %[[V62:.*]] = stablehlo.reduce(%[[V61]] init: %[[Vcst_5]]) applies stablehlo.add across dimensions = [1] : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
// LAST_OZAKI: %[[V63:.*]] = stablehlo.dot_general %[[V18]], %[[V21]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2x2xf32>) -> tensor<2xf32>
// LAST_OZAKI: %[[V64:.*]] = stablehlo.dot_general %[[V19]], %[[V20]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2x2xf32>) -> tensor<2xf32>
// LAST_OZAKI: %[[V65:.*]] = stablehlo.add %[[V62]], %[[V63]] : tensor<2xf32>
// LAST_OZAKI: %[[V66:.*]] = stablehlo.subtract %[[V65]], %[[V63]] : tensor<2xf32>
// LAST_OZAKI: %[[V67:.*]] = stablehlo.subtract %[[V65]], %[[V66]] : tensor<2xf32>
// LAST_OZAKI: %[[V68:.*]] = stablehlo.subtract %[[V62]], %[[V66]] : tensor<2xf32>
// LAST_OZAKI: %[[V69:.*]] = stablehlo.subtract %[[V63]], %[[V67]] : tensor<2xf32>
// LAST_OZAKI: %[[V70:.*]] = stablehlo.add %[[V68]], %[[V69]] : tensor<2xf32>
// LAST_OZAKI: %[[V71:.*]] = stablehlo.add %[[V65]], %[[V64]] : tensor<2xf32>
// LAST_OZAKI: %[[V72:.*]] = stablehlo.subtract %[[V71]], %[[V64]] : tensor<2xf32>
// LAST_OZAKI: %[[V73:.*]] = stablehlo.subtract %[[V71]], %[[V72]] : tensor<2xf32>
// LAST_OZAKI: %[[V74:.*]] = stablehlo.subtract %[[V65]], %[[V72]] : tensor<2xf32>
// LAST_OZAKI: %[[V75:.*]] = stablehlo.subtract %[[V64]], %[[V73]] : tensor<2xf32>
// LAST_OZAKI: %[[V76:.*]] = stablehlo.add %[[V74]], %[[V75]] : tensor<2xf32>
// LAST_OZAKI: %[[V77:.*]] = stablehlo.add %[[V70]], %[[V76]] : tensor<2xf32>
// LAST_OZAKI: %[[V78:.*]] = stablehlo.add %[[V71]], %[[V77]] : tensor<2xf32>
// LAST_OZAKI: %[[V79:.*]] = stablehlo.subtract %[[V78]], %[[V77]] : tensor<2xf32>
// LAST_OZAKI: %[[V80:.*]] = stablehlo.subtract %[[V78]], %[[V79]] : tensor<2xf32>
// LAST_OZAKI: %[[V81:.*]] = stablehlo.subtract %[[V71]], %[[V79]] : tensor<2xf32>
// LAST_OZAKI: %[[V82:.*]] = stablehlo.subtract %[[V77]], %[[V80]] : tensor<2xf32>
// LAST_OZAKI: %[[V83:.*]] = stablehlo.add %[[V81]], %[[V82]] : tensor<2xf32>
// LAST_OZAKI: %[[V84:.*]] = stablehlo.reshape %[[V78]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST_OZAKI: %[[V85:.*]] = stablehlo.reshape %[[V83]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST_OZAKI: %[[V86:.*]] = stablehlo.concatenate %[[V84]], %[[V85]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// LAST_OZAKI: %[[V87:.*]] = stablehlo.convert %[[V86]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
// LAST_OZAKI: %[[Vcst_6:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// LAST_OZAKI: %[[V88:.*]] = stablehlo.reduce(%[[V87]] init: %[[Vcst_6]]) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
// LAST_OZAKI: return %[[V88]] : tensor<2xf64>

func.func @main() attributes {enzyme.no_multifloat} {
  %c = stablehlo.constant dense<[1.5, 2.0]> : tensor<2xf64>
  %m = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>
  %expected = stablehlo.constant dense<[7.5, 11.0]> : tensor<2xf64>
  
  %res_first = func.call @dot_mini_first(%c, %m) : (tensor<2xf64>, tensor<2x2xf64>) -> tensor<2xf64>
  "check.expect_close"(%res_first, %expected) {max_ulp_difference = 100 : ui64} : (tensor<2xf64>, tensor<2xf64>) -> ()
  
  %res_last = func.call @dot_mini_last(%c, %m) : (tensor<2xf64>, tensor<2x2xf64>) -> tensor<2xf64>
  "check.expect_close"(%res_last, %expected) {max_ulp_difference = 100 : ui64} : (tensor<2xf64>, tensor<2xf64>) -> ()
  
  return
}
