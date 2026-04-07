// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f32 target-type=bf16 concat-dimension=first dot-general-to-reduce=false" %s | FileCheck --check-prefix=FIRST_BF16 %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f32 target-type=bf16 concat-dimension=last dot-general-to-reduce=false" %s | FileCheck --check-prefix=LAST_BF16 %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f32 target-type=bf16 concat-dimension=first dot-general-to-reduce=false" %s | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f32 target-type=bf16 concat-dimension=last dot-general-to-reduce=false" %s | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @dot_bf16_first(%arg0: tensor<2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2x2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// FIRST_BF16-LABEL: func.func @dot_bf16_first
// FIRST_BF16: %[[V0:.*]] = stablehlo.convert %arg0 : (tensor<2xf32>) -> tensor<2xbf16>
// FIRST_BF16: %[[V1:.*]] = stablehlo.convert %[[V0]] : (tensor<2xbf16>) -> tensor<2xf32>
// FIRST_BF16: %[[V2:.*]] = stablehlo.subtract %arg0, %[[V1]] : tensor<2xf32>
// FIRST_BF16: %[[V3:.*]] = stablehlo.convert %[[V2]] : (tensor<2xf32>) -> tensor<2xbf16>
// FIRST_BF16: %[[V4:.*]] = stablehlo.reshape %[[V0]] : (tensor<2xbf16>) -> tensor<1x2xbf16>
// FIRST_BF16: %[[V5:.*]] = stablehlo.reshape %[[V3]] : (tensor<2xbf16>) -> tensor<1x2xbf16>
// FIRST_BF16: %[[V6:.*]] = stablehlo.concatenate %[[V4]], %[[V5]], dim = 0 : (tensor<1x2xbf16>, tensor<1x2xbf16>) -> tensor<2x2xbf16>
// FIRST_BF16: %[[V7:.*]] = stablehlo.convert %arg1 : (tensor<2x2xf32>) -> tensor<2x2xbf16>
// FIRST_BF16: %[[V8:.*]] = stablehlo.convert %[[V7]] : (tensor<2x2xbf16>) -> tensor<2x2xf32>
// FIRST_BF16: %[[V9:.*]] = stablehlo.subtract %arg1, %[[V8]] : tensor<2x2xf32>
// FIRST_BF16: %[[V10:.*]] = stablehlo.convert %[[V9]] : (tensor<2x2xf32>) -> tensor<2x2xbf16>
// FIRST_BF16: %[[V11:.*]] = stablehlo.reshape %[[V7]] : (tensor<2x2xbf16>) -> tensor<1x2x2xbf16>
// FIRST_BF16: %[[V12:.*]] = stablehlo.reshape %[[V10]] : (tensor<2x2xbf16>) -> tensor<1x2x2xbf16>
// FIRST_BF16: %[[V13:.*]] = stablehlo.concatenate %[[V11]], %[[V12]], dim = 0 : (tensor<1x2x2xbf16>, tensor<1x2x2xbf16>) -> tensor<2x2x2xbf16>
// FIRST_BF16: %[[V14:.*]] = stablehlo.slice %[[V6]] [0:1, 0:2] : (tensor<2x2xbf16>) -> tensor<1x2xbf16>
// FIRST_BF16: %[[V15:.*]] = stablehlo.slice %[[V6]] [1:2, 0:2] : (tensor<2x2xbf16>) -> tensor<1x2xbf16>
// FIRST_BF16: %[[V16:.*]] = stablehlo.slice %[[V13]] [0:1, 0:2, 0:2] : (tensor<2x2x2xbf16>) -> tensor<1x2x2xbf16>
// FIRST_BF16: %[[V17:.*]] = stablehlo.slice %[[V13]] [1:2, 0:2, 0:2] : (tensor<2x2x2xbf16>) -> tensor<1x2x2xbf16>
// FIRST_BF16: %[[V18:.*]] = stablehlo.reshape %[[V14]] : (tensor<1x2xbf16>) -> tensor<2xbf16>
// FIRST_BF16: %[[V19:.*]] = stablehlo.reshape %[[V15]] : (tensor<1x2xbf16>) -> tensor<2xbf16>
// FIRST_BF16: %[[V20:.*]] = stablehlo.reshape %[[V16]] : (tensor<1x2x2xbf16>) -> tensor<2x2xbf16>
// FIRST_BF16: %[[V21:.*]] = stablehlo.reshape %[[V17]] : (tensor<1x2x2xbf16>) -> tensor<2x2xbf16>
// FIRST_BF16: %[[V22:.*]] = stablehlo.abs %[[V18]] : tensor<2xbf16>
// FIRST_BF16: %[[Vcst:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
// FIRST_BF16: %[[V23:.*]] = stablehlo.reduce(%[[V22]] init: %[[Vcst]]) applies stablehlo.maximum across dimensions = [0] : (tensor<2xbf16>, tensor<bf16>) -> tensor<bf16>
// FIRST_BF16: %[[V24:.*]] = stablehlo.abs %[[V20]] : tensor<2x2xbf16>
// FIRST_BF16: %[[Vcst_0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
// FIRST_BF16: %[[V25:.*]] = stablehlo.reduce(%[[V24]] init: %[[Vcst_0]]) applies stablehlo.maximum across dimensions = [0, 1] : (tensor<2x2xbf16>, tensor<bf16>) -> tensor<bf16>
// FIRST_BF16: %[[Vcst_1:.*]] = stablehlo.constant dense<6.914060e-01> : tensor<bf16>
// FIRST_BF16: %[[V26:.*]] = stablehlo.log %[[V23]] : tensor<bf16>
// FIRST_BF16: %[[V27:.*]] = stablehlo.divide %[[V26]], %[[Vcst_1]] : tensor<bf16>
// FIRST_BF16: %[[V28:.*]] = stablehlo.ceil %[[V27]] : tensor<bf16>
// FIRST_BF16: %[[V29:.*]] = stablehlo.multiply %[[V28]], %[[Vcst_1]] : tensor<bf16>
// FIRST_BF16: %[[V30:.*]] = stablehlo.exponential %[[V29]] : tensor<bf16>
// FIRST_BF16: %[[V31:.*]] = stablehlo.broadcast_in_dim %[[V30]], dims = [] : (tensor<bf16>) -> tensor<2xbf16>
// FIRST_BF16: %[[V32:.*]] = stablehlo.log %[[V25]] : tensor<bf16>
// FIRST_BF16: %[[V33:.*]] = stablehlo.divide %[[V32]], %[[Vcst_1]] : tensor<bf16>
// FIRST_BF16: %[[V34:.*]] = stablehlo.ceil %[[V33]] : tensor<bf16>
// FIRST_BF16: %[[V35:.*]] = stablehlo.multiply %[[V34]], %[[Vcst_1]] : tensor<bf16>
// FIRST_BF16: %[[V36:.*]] = stablehlo.exponential %[[V35]] : tensor<bf16>
// FIRST_BF16: %[[V37:.*]] = stablehlo.broadcast_in_dim %[[V36]], dims = [] : (tensor<bf16>) -> tensor<2x2xbf16>
// FIRST_BF16: %[[V38:.*]] = stablehlo.divide %[[V18]], %[[V31]] : tensor<2xbf16>
// FIRST_BF16: %[[V39:.*]] = stablehlo.divide %[[V20]], %[[V37]] : tensor<2x2xbf16>
// FIRST_BF16: %[[Vcst_2:.*]] = stablehlo.constant dense<8.000000e+00> : tensor<2xbf16>
// FIRST_BF16: %[[V40:.*]] = stablehlo.multiply %[[V38]], %[[Vcst_2]] : tensor<2xbf16>
// FIRST_BF16: %[[V41:.*]] = stablehlo.floor %[[V40]] : tensor<2xbf16>
// FIRST_BF16: %[[V42:.*]] = stablehlo.subtract %[[V40]], %[[V41]] : tensor<2xbf16>
// FIRST_BF16: %[[Vcst_3:.*]] = stablehlo.constant dense<8.000000e+00> : tensor<2x2xbf16>
// FIRST_BF16: %[[V43:.*]] = stablehlo.multiply %[[V39]], %[[Vcst_3]] : tensor<2x2xbf16>
// FIRST_BF16: %[[V44:.*]] = stablehlo.floor %[[V43]] : tensor<2x2xbf16>
// FIRST_BF16: %[[V45:.*]] = stablehlo.subtract %[[V43]], %[[V44]] : tensor<2x2xbf16>
// FIRST_BF16: %[[V46:.*]] = stablehlo.dot_general %[[V41]], %[[V44]], contracting_dims = [0] x [0] : (tensor<2xbf16>, tensor<2x2xbf16>) -> tensor<2xbf16>
// FIRST_BF16: %[[V47:.*]] = stablehlo.dot_general %[[V41]], %[[V45]], contracting_dims = [0] x [0] : (tensor<2xbf16>, tensor<2x2xbf16>) -> tensor<2xbf16>
// FIRST_BF16: %[[V48:.*]] = stablehlo.dot_general %[[V42]], %[[V44]], contracting_dims = [0] x [0] : (tensor<2xbf16>, tensor<2x2xbf16>) -> tensor<2xbf16>
// FIRST_BF16: %[[V49:.*]] = stablehlo.dot_general %[[V42]], %[[V45]], contracting_dims = [0] x [0] : (tensor<2xbf16>, tensor<2x2xbf16>) -> tensor<2xbf16>
// FIRST_BF16: %[[V50:.*]] = stablehlo.multiply %[[V30]], %[[V36]] : tensor<bf16>
// FIRST_BF16: %[[Vcst_4:.*]] = stablehlo.constant dense<6.400000e+01> : tensor<bf16>
// FIRST_BF16: %[[V51:.*]] = stablehlo.divide %[[V50]], %[[Vcst_4]] : tensor<bf16>
// FIRST_BF16: %[[V52:.*]] = stablehlo.broadcast_in_dim %[[V51]], dims = [] : (tensor<bf16>) -> tensor<2xbf16>
// FIRST_BF16: %[[V53:.*]] = stablehlo.multiply %[[V46]], %[[V52]] : tensor<2xbf16>
// FIRST_BF16: %[[V54:.*]] = stablehlo.multiply %[[V47]], %[[V52]] : tensor<2xbf16>
// FIRST_BF16: %[[V55:.*]] = stablehlo.multiply %[[V48]], %[[V52]] : tensor<2xbf16>
// FIRST_BF16: %[[V56:.*]] = stablehlo.multiply %[[V49]], %[[V52]] : tensor<2xbf16>
// FIRST_BF16: %[[V57:.*]] = stablehlo.reshape %[[V53]] : (tensor<2xbf16>) -> tensor<2x1xbf16>
// FIRST_BF16: %[[V58:.*]] = stablehlo.reshape %[[V54]] : (tensor<2xbf16>) -> tensor<2x1xbf16>
// FIRST_BF16: %[[V59:.*]] = stablehlo.reshape %[[V55]] : (tensor<2xbf16>) -> tensor<2x1xbf16>
// FIRST_BF16: %[[V60:.*]] = stablehlo.reshape %[[V56]] : (tensor<2xbf16>) -> tensor<2x1xbf16>
// FIRST_BF16: %[[V61:.*]] = stablehlo.concatenate %[[V57]], %[[V58]], %[[V59]], %[[V60]], dim = 1 : (tensor<2x1xbf16>, tensor<2x1xbf16>, tensor<2x1xbf16>, tensor<2x1xbf16>) -> tensor<2x4xbf16>
// FIRST_BF16: %[[Vcst_5:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
// FIRST_BF16: %[[V62:.*]] = stablehlo.reduce(%[[V61]] init: %[[Vcst_5]]) applies stablehlo.add across dimensions = [1] : (tensor<2x4xbf16>, tensor<bf16>) -> tensor<2xbf16>
// FIRST_BF16: %[[V63:.*]] = stablehlo.dot_general %[[V18]], %[[V21]], contracting_dims = [0] x [0] : (tensor<2xbf16>, tensor<2x2xbf16>) -> tensor<2xbf16>
// FIRST_BF16: %[[V64:.*]] = stablehlo.dot_general %[[V19]], %[[V20]], contracting_dims = [0] x [0] : (tensor<2xbf16>, tensor<2x2xbf16>) -> tensor<2xbf16>
// FIRST_BF16: %[[V65:.*]] = stablehlo.add %[[V62]], %[[V63]] : tensor<2xbf16>
// FIRST_BF16: %[[V66:.*]] = stablehlo.subtract %[[V65]], %[[V63]] : tensor<2xbf16>
// FIRST_BF16: %[[V67:.*]] = stablehlo.subtract %[[V65]], %[[V66]] : tensor<2xbf16>
// FIRST_BF16: %[[V68:.*]] = stablehlo.subtract %[[V62]], %[[V66]] : tensor<2xbf16>
// FIRST_BF16: %[[V69:.*]] = stablehlo.subtract %[[V63]], %[[V67]] : tensor<2xbf16>
// FIRST_BF16: %[[V70:.*]] = stablehlo.add %[[V68]], %[[V69]] : tensor<2xbf16>
// FIRST_BF16: %[[V71:.*]] = stablehlo.add %[[V65]], %[[V64]] : tensor<2xbf16>
// FIRST_BF16: %[[V72:.*]] = stablehlo.subtract %[[V71]], %[[V64]] : tensor<2xbf16>
// FIRST_BF16: %[[V73:.*]] = stablehlo.subtract %[[V71]], %[[V72]] : tensor<2xbf16>
// FIRST_BF16: %[[V74:.*]] = stablehlo.subtract %[[V65]], %[[V72]] : tensor<2xbf16>
// FIRST_BF16: %[[V75:.*]] = stablehlo.subtract %[[V64]], %[[V73]] : tensor<2xbf16>
// FIRST_BF16: %[[V76:.*]] = stablehlo.add %[[V74]], %[[V75]] : tensor<2xbf16>
// FIRST_BF16: %[[V77:.*]] = stablehlo.add %[[V70]], %[[V76]] : tensor<2xbf16>
// FIRST_BF16: %[[V78:.*]] = stablehlo.add %[[V71]], %[[V77]] : tensor<2xbf16>
// FIRST_BF16: %[[V79:.*]] = stablehlo.subtract %[[V78]], %[[V77]] : tensor<2xbf16>
// FIRST_BF16: %[[V80:.*]] = stablehlo.subtract %[[V78]], %[[V79]] : tensor<2xbf16>
// FIRST_BF16: %[[V81:.*]] = stablehlo.subtract %[[V71]], %[[V79]] : tensor<2xbf16>
// FIRST_BF16: %[[V82:.*]] = stablehlo.subtract %[[V77]], %[[V80]] : tensor<2xbf16>
// FIRST_BF16: %[[V83:.*]] = stablehlo.add %[[V81]], %[[V82]] : tensor<2xbf16>
// FIRST_BF16: %[[V84:.*]] = stablehlo.reshape %[[V78]] : (tensor<2xbf16>) -> tensor<1x2xbf16>
// FIRST_BF16: %[[V85:.*]] = stablehlo.reshape %[[V83]] : (tensor<2xbf16>) -> tensor<1x2xbf16>
// FIRST_BF16: %[[V86:.*]] = stablehlo.concatenate %[[V84]], %[[V85]], dim = 0 : (tensor<1x2xbf16>, tensor<1x2xbf16>) -> tensor<2x2xbf16>
// FIRST_BF16: %[[V87:.*]] = stablehlo.convert %[[V86]] : (tensor<2x2xbf16>) -> tensor<2x2xf32>
// FIRST_BF16: %[[Vcst_6:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// FIRST_BF16: %[[V88:.*]] = stablehlo.reduce(%[[V87]] init: %[[Vcst_6]]) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
// FIRST_BF16: return %[[V88]] : tensor<2xf32>

func.func @dot_bf16_last(%arg0: tensor<2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2x2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// LAST_BF16-LABEL: func.func @dot_bf16_last
// LAST_BF16: %[[V0:.*]] = stablehlo.convert %arg0 : (tensor<2xf32>) -> tensor<2xbf16>
// LAST_BF16: %[[V1:.*]] = stablehlo.convert %[[V0]] : (tensor<2xbf16>) -> tensor<2xf32>
// LAST_BF16: %[[V2:.*]] = stablehlo.subtract %arg0, %[[V1]] : tensor<2xf32>
// LAST_BF16: %[[V3:.*]] = stablehlo.convert %[[V2]] : (tensor<2xf32>) -> tensor<2xbf16>
// LAST_BF16: %[[V4:.*]] = stablehlo.reshape %[[V0]] : (tensor<2xbf16>) -> tensor<2x1xbf16>
// LAST_BF16: %[[V5:.*]] = stablehlo.reshape %[[V3]] : (tensor<2xbf16>) -> tensor<2x1xbf16>
// LAST_BF16: %[[V6:.*]] = stablehlo.concatenate %[[V4]], %[[V5]], dim = 1 : (tensor<2x1xbf16>, tensor<2x1xbf16>) -> tensor<2x2xbf16>
// LAST_BF16: %[[V7:.*]] = stablehlo.convert %arg1 : (tensor<2x2xf32>) -> tensor<2x2xbf16>
// LAST_BF16: %[[V8:.*]] = stablehlo.convert %[[V7]] : (tensor<2x2xbf16>) -> tensor<2x2xf32>
// LAST_BF16: %[[V9:.*]] = stablehlo.subtract %arg1, %[[V8]] : tensor<2x2xf32>
// LAST_BF16: %[[V10:.*]] = stablehlo.convert %[[V9]] : (tensor<2x2xf32>) -> tensor<2x2xbf16>
// LAST_BF16: %[[V11:.*]] = stablehlo.reshape %[[V7]] : (tensor<2x2xbf16>) -> tensor<2x2x1xbf16>
// LAST_BF16: %[[V12:.*]] = stablehlo.reshape %[[V10]] : (tensor<2x2xbf16>) -> tensor<2x2x1xbf16>
// LAST_BF16: %[[V13:.*]] = stablehlo.concatenate %[[V11]], %[[V12]], dim = 2 : (tensor<2x2x1xbf16>, tensor<2x2x1xbf16>) -> tensor<2x2x2xbf16>
// LAST_BF16: %[[V14:.*]] = stablehlo.slice %[[V6]] [0:2, 0:1] : (tensor<2x2xbf16>) -> tensor<2x1xbf16>
// LAST_BF16: %[[V15:.*]] = stablehlo.slice %[[V6]] [0:2, 1:2] : (tensor<2x2xbf16>) -> tensor<2x1xbf16>
// LAST_BF16: %[[V16:.*]] = stablehlo.slice %[[V13]] [0:2, 0:2, 0:1] : (tensor<2x2x2xbf16>) -> tensor<2x2x1xbf16>
// LAST_BF16: %[[V17:.*]] = stablehlo.slice %[[V13]] [0:2, 0:2, 1:2] : (tensor<2x2x2xbf16>) -> tensor<2x2x1xbf16>
// LAST_BF16: %[[V18:.*]] = stablehlo.reshape %[[V14]] : (tensor<2x1xbf16>) -> tensor<2xbf16>
// LAST_BF16: %[[V19:.*]] = stablehlo.reshape %[[V15]] : (tensor<2x1xbf16>) -> tensor<2xbf16>
// LAST_BF16: %[[V20:.*]] = stablehlo.reshape %[[V16]] : (tensor<2x2x1xbf16>) -> tensor<2x2xbf16>
// LAST_BF16: %[[V21:.*]] = stablehlo.reshape %[[V17]] : (tensor<2x2x1xbf16>) -> tensor<2x2xbf16>
// LAST_BF16: %[[V22:.*]] = stablehlo.abs %[[V18]] : tensor<2xbf16>
// LAST_BF16: %[[Vcst:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
// LAST_BF16: %[[V23:.*]] = stablehlo.reduce(%[[V22]] init: %[[Vcst]]) applies stablehlo.maximum across dimensions = [0] : (tensor<2xbf16>, tensor<bf16>) -> tensor<bf16>
// LAST_BF16: %[[V24:.*]] = stablehlo.abs %[[V20]] : tensor<2x2xbf16>
// LAST_BF16: %[[Vcst_0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
// LAST_BF16: %[[V25:.*]] = stablehlo.reduce(%[[V24]] init: %[[Vcst_0]]) applies stablehlo.maximum across dimensions = [0, 1] : (tensor<2x2xbf16>, tensor<bf16>) -> tensor<bf16>
// LAST_BF16: %[[Vcst_1:.*]] = stablehlo.constant dense<6.914060e-01> : tensor<bf16>
// LAST_BF16: %[[V26:.*]] = stablehlo.log %[[V23]] : tensor<bf16>
// LAST_BF16: %[[V27:.*]] = stablehlo.divide %[[V26]], %[[Vcst_1]] : tensor<bf16>
// LAST_BF16: %[[V28:.*]] = stablehlo.ceil %[[V27]] : tensor<bf16>
// LAST_BF16: %[[V29:.*]] = stablehlo.multiply %[[V28]], %[[Vcst_1]] : tensor<bf16>
// LAST_BF16: %[[V30:.*]] = stablehlo.exponential %[[V29]] : tensor<bf16>
// LAST_BF16: %[[V31:.*]] = stablehlo.broadcast_in_dim %[[V30]], dims = [] : (tensor<bf16>) -> tensor<2xbf16>
// LAST_BF16: %[[V32:.*]] = stablehlo.log %[[V25]] : tensor<bf16>
// LAST_BF16: %[[V33:.*]] = stablehlo.divide %[[V32]], %[[Vcst_1]] : tensor<bf16>
// LAST_BF16: %[[V34:.*]] = stablehlo.ceil %[[V33]] : tensor<bf16>
// LAST_BF16: %[[V35:.*]] = stablehlo.multiply %[[V34]], %[[Vcst_1]] : tensor<bf16>
// LAST_BF16: %[[V36:.*]] = stablehlo.exponential %[[V35]] : tensor<bf16>
// LAST_BF16: %[[V37:.*]] = stablehlo.broadcast_in_dim %[[V36]], dims = [] : (tensor<bf16>) -> tensor<2x2xbf16>
// LAST_BF16: %[[V38:.*]] = stablehlo.divide %[[V18]], %[[V31]] : tensor<2xbf16>
// LAST_BF16: %[[V39:.*]] = stablehlo.divide %[[V20]], %[[V37]] : tensor<2x2xbf16>
// LAST_BF16: %[[Vcst_2:.*]] = stablehlo.constant dense<8.000000e+00> : tensor<2xbf16>
// LAST_BF16: %[[V40:.*]] = stablehlo.multiply %[[V38]], %[[Vcst_2]] : tensor<2xbf16>
// LAST_BF16: %[[V41:.*]] = stablehlo.floor %[[V40]] : tensor<2xbf16>
// LAST_BF16: %[[V42:.*]] = stablehlo.subtract %[[V40]], %[[V41]] : tensor<2xbf16>
// LAST_BF16: %[[Vcst_3:.*]] = stablehlo.constant dense<8.000000e+00> : tensor<2x2xbf16>
// LAST_BF16: %[[V43:.*]] = stablehlo.multiply %[[V39]], %[[Vcst_3]] : tensor<2x2xbf16>
// LAST_BF16: %[[V44:.*]] = stablehlo.floor %[[V43]] : tensor<2x2xbf16>
// LAST_BF16: %[[V45:.*]] = stablehlo.subtract %[[V43]], %[[V44]] : tensor<2x2xbf16>
// LAST_BF16: %[[V46:.*]] = stablehlo.dot_general %[[V41]], %[[V44]], contracting_dims = [0] x [0] : (tensor<2xbf16>, tensor<2x2xbf16>) -> tensor<2xbf16>
// LAST_BF16: %[[V47:.*]] = stablehlo.dot_general %[[V41]], %[[V45]], contracting_dims = [0] x [0] : (tensor<2xbf16>, tensor<2x2xbf16>) -> tensor<2xbf16>
// LAST_BF16: %[[V48:.*]] = stablehlo.dot_general %[[V42]], %[[V44]], contracting_dims = [0] x [0] : (tensor<2xbf16>, tensor<2x2xbf16>) -> tensor<2xbf16>
// LAST_BF16: %[[V49:.*]] = stablehlo.dot_general %[[V42]], %[[V45]], contracting_dims = [0] x [0] : (tensor<2xbf16>, tensor<2x2xbf16>) -> tensor<2xbf16>
// LAST_BF16: %[[V50:.*]] = stablehlo.multiply %[[V30]], %[[V36]] : tensor<bf16>
// LAST_BF16: %[[Vcst_4:.*]] = stablehlo.constant dense<6.400000e+01> : tensor<bf16>
// LAST_BF16: %[[V51:.*]] = stablehlo.divide %[[V50]], %[[Vcst_4]] : tensor<bf16>
// LAST_BF16: %[[V52:.*]] = stablehlo.broadcast_in_dim %[[V51]], dims = [] : (tensor<bf16>) -> tensor<2xbf16>
// LAST_BF16: %[[V53:.*]] = stablehlo.multiply %[[V46]], %[[V52]] : tensor<2xbf16>
// LAST_BF16: %[[V54:.*]] = stablehlo.multiply %[[V47]], %[[V52]] : tensor<2xbf16>
// LAST_BF16: %[[V55:.*]] = stablehlo.multiply %[[V48]], %[[V52]] : tensor<2xbf16>
// LAST_BF16: %[[V56:.*]] = stablehlo.multiply %[[V49]], %[[V52]] : tensor<2xbf16>
// LAST_BF16: %[[V57:.*]] = stablehlo.reshape %[[V53]] : (tensor<2xbf16>) -> tensor<2x1xbf16>
// LAST_BF16: %[[V58:.*]] = stablehlo.reshape %[[V54]] : (tensor<2xbf16>) -> tensor<2x1xbf16>
// LAST_BF16: %[[V59:.*]] = stablehlo.reshape %[[V55]] : (tensor<2xbf16>) -> tensor<2x1xbf16>
// LAST_BF16: %[[V60:.*]] = stablehlo.reshape %[[V56]] : (tensor<2xbf16>) -> tensor<2x1xbf16>
// LAST_BF16: %[[V61:.*]] = stablehlo.concatenate %[[V57]], %[[V58]], %[[V59]], %[[V60]], dim = 1 : (tensor<2x1xbf16>, tensor<2x1xbf16>, tensor<2x1xbf16>, tensor<2x1xbf16>) -> tensor<2x4xbf16>
// LAST_BF16: %[[Vcst_5:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
// LAST_BF16: %[[V62:.*]] = stablehlo.reduce(%[[V61]] init: %[[Vcst_5]]) applies stablehlo.add across dimensions = [1] : (tensor<2x4xbf16>, tensor<bf16>) -> tensor<2xbf16>
// LAST_BF16: %[[V63:.*]] = stablehlo.dot_general %[[V18]], %[[V21]], contracting_dims = [0] x [0] : (tensor<2xbf16>, tensor<2x2xbf16>) -> tensor<2xbf16>
// LAST_BF16: %[[V64:.*]] = stablehlo.dot_general %[[V19]], %[[V20]], contracting_dims = [0] x [0] : (tensor<2xbf16>, tensor<2x2xbf16>) -> tensor<2xbf16>
// LAST_BF16: %[[V65:.*]] = stablehlo.add %[[V62]], %[[V63]] : tensor<2xbf16>
// LAST_BF16: %[[V66:.*]] = stablehlo.subtract %[[V65]], %[[V63]] : tensor<2xbf16>
// LAST_BF16: %[[V67:.*]] = stablehlo.subtract %[[V65]], %[[V66]] : tensor<2xbf16>
// LAST_BF16: %[[V68:.*]] = stablehlo.subtract %[[V62]], %[[V66]] : tensor<2xbf16>
// LAST_BF16: %[[V69:.*]] = stablehlo.subtract %[[V63]], %[[V67]] : tensor<2xbf16>
// LAST_BF16: %[[V70:.*]] = stablehlo.add %[[V68]], %[[V69]] : tensor<2xbf16>
// LAST_BF16: %[[V71:.*]] = stablehlo.add %[[V65]], %[[V64]] : tensor<2xbf16>
// LAST_BF16: %[[V72:.*]] = stablehlo.subtract %[[V71]], %[[V64]] : tensor<2xbf16>
// LAST_BF16: %[[V73:.*]] = stablehlo.subtract %[[V71]], %[[V72]] : tensor<2xbf16>
// LAST_BF16: %[[V74:.*]] = stablehlo.subtract %[[V65]], %[[V72]] : tensor<2xbf16>
// LAST_BF16: %[[V75:.*]] = stablehlo.subtract %[[V64]], %[[V73]] : tensor<2xbf16>
// LAST_BF16: %[[V76:.*]] = stablehlo.add %[[V74]], %[[V75]] : tensor<2xbf16>
// LAST_BF16: %[[V77:.*]] = stablehlo.add %[[V70]], %[[V76]] : tensor<2xbf16>
// LAST_BF16: %[[V78:.*]] = stablehlo.add %[[V71]], %[[V77]] : tensor<2xbf16>
// LAST_BF16: %[[V79:.*]] = stablehlo.subtract %[[V78]], %[[V77]] : tensor<2xbf16>
// LAST_BF16: %[[V80:.*]] = stablehlo.subtract %[[V78]], %[[V79]] : tensor<2xbf16>
// LAST_BF16: %[[V81:.*]] = stablehlo.subtract %[[V71]], %[[V79]] : tensor<2xbf16>
// LAST_BF16: %[[V82:.*]] = stablehlo.subtract %[[V77]], %[[V80]] : tensor<2xbf16>
// LAST_BF16: %[[V83:.*]] = stablehlo.add %[[V81]], %[[V82]] : tensor<2xbf16>
// LAST_BF16: %[[V84:.*]] = stablehlo.reshape %[[V78]] : (tensor<2xbf16>) -> tensor<2x1xbf16>
// LAST_BF16: %[[V85:.*]] = stablehlo.reshape %[[V83]] : (tensor<2xbf16>) -> tensor<2x1xbf16>
// LAST_BF16: %[[V86:.*]] = stablehlo.concatenate %[[V84]], %[[V85]], dim = 1 : (tensor<2x1xbf16>, tensor<2x1xbf16>) -> tensor<2x2xbf16>
// LAST_BF16: %[[V87:.*]] = stablehlo.convert %[[V86]] : (tensor<2x2xbf16>) -> tensor<2x2xf32>
// LAST_BF16: %[[Vcst_6:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// LAST_BF16: %[[V88:.*]] = stablehlo.reduce(%[[V87]] init: %[[Vcst_6]]) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
// LAST_BF16: return %[[V88]] : tensor<2xf32>

func.func @main() attributes {enzyme.no_multifloat} {
  %c = stablehlo.constant dense<[1.5, 2.0]> : tensor<2xf32>
  %m = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %expected = stablehlo.constant dense<[7.5, 11.0]> : tensor<2xf32>
  
  %res_first = func.call @dot_bf16_first(%c, %m) : (tensor<2xf32>, tensor<2x2xf32>) -> tensor<2xf32>
  "check.expect_close"(%res_first, %expected) {max_ulp_difference = 100 : ui64} : (tensor<2xf32>, tensor<2xf32>) -> ()
  
  %res_last = func.call @dot_bf16_last(%c, %m) : (tensor<2xf32>, tensor<2x2xf32>) -> tensor<2xf32>
  "check.expect_close"(%res_last, %expected) {max_ulp_difference = 100 : ui64} : (tensor<2xf32>, tensor<2xf32>) -> ()
  
  return
}
