// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck --check-prefix=FIRST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=TUPLE %s
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @dot_general(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>) -> tensor<f64> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [0] : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
  return %0 : tensor<f64>
}

// TUPLE: module {
// TUPLE: func.func @dot_general(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>) -> tensor<f64> {
// TUPLE: %[[V0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// TUPLE: %[[V1:.*]] = stablehlo.convert %[[V0]] : (tensor<2xf32>) -> tensor<2xf64>
// TUPLE: %[[V2:.*]] = stablehlo.subtract %arg0, %[[V1]] : tensor<2xf64>
// TUPLE: %[[V3:.*]] = stablehlo.convert %[[V2]] : (tensor<2xf64>) -> tensor<2xf32>
// TUPLE: %[[V4:.*]] = stablehlo.tuple %[[V0]], %[[V3]] : tuple<tensor<2xf32>, tensor<2xf32>>
// TUPLE: %[[V5:.*]] = stablehlo.convert %arg1 : (tensor<2xf64>) -> tensor<2xf32>
// TUPLE: %[[V6:.*]] = stablehlo.convert %[[V5]] : (tensor<2xf32>) -> tensor<2xf64>
// TUPLE: %[[V7:.*]] = stablehlo.subtract %arg1, %[[V6]] : tensor<2xf64>
// TUPLE: %[[V8:.*]] = stablehlo.convert %[[V7]] : (tensor<2xf64>) -> tensor<2xf32>
// TUPLE: %[[V9:.*]] = stablehlo.tuple %[[V5]], %[[V8]] : tuple<tensor<2xf32>, tensor<2xf32>>
// TUPLE: %[[V10:.*]] = stablehlo.get_tuple_element %[[V4]][0] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE: %[[V11:.*]] = stablehlo.get_tuple_element %[[V4]][1] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE: %[[V12:.*]] = stablehlo.broadcast_in_dim %[[V10]], dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
// TUPLE: %[[V13:.*]] = stablehlo.broadcast_in_dim %[[V11]], dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
// TUPLE: %[[V14:.*]] = stablehlo.tuple %[[V12]], %[[V13]] : tuple<tensor<2xf32>, tensor<2xf32>>
// TUPLE: %[[V15:.*]] = stablehlo.get_tuple_element %[[V9]][0] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE: %[[V16:.*]] = stablehlo.get_tuple_element %[[V9]][1] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE: %[[V17:.*]] = stablehlo.multiply %[[V12]], %[[V15]] : tensor<2xf32>
// TUPLE: %[[Vcst:.*]] = stablehlo.constant dense<4.097000e+03> : tensor<2xf32>
// TUPLE: %[[V18:.*]] = stablehlo.multiply %[[V12]], %[[Vcst]] : tensor<2xf32>
// TUPLE: %[[V19:.*]] = stablehlo.subtract %[[V18]], %[[V12]] : tensor<2xf32>
// TUPLE: %[[V20:.*]] = stablehlo.subtract %[[V18]], %[[V19]] : tensor<2xf32>
// TUPLE: %[[V21:.*]] = stablehlo.subtract %[[V12]], %[[V20]] : tensor<2xf32>
// TUPLE: %[[Vcst_0:.*]] = stablehlo.constant dense<4.097000e+03> : tensor<2xf32>
// TUPLE: %[[V22:.*]] = stablehlo.multiply %[[V15]], %[[Vcst_0]] : tensor<2xf32>
// TUPLE: %[[V23:.*]] = stablehlo.subtract %[[V22]], %[[V15]] : tensor<2xf32>
// TUPLE: %[[V24:.*]] = stablehlo.subtract %[[V22]], %[[V23]] : tensor<2xf32>
// TUPLE: %[[V25:.*]] = stablehlo.subtract %[[V15]], %[[V24]] : tensor<2xf32>
// TUPLE: %[[V26:.*]] = stablehlo.multiply %[[V20]], %[[V24]] : tensor<2xf32>
// TUPLE: %[[V27:.*]] = stablehlo.multiply %[[V20]], %[[V25]] : tensor<2xf32>
// TUPLE: %[[V28:.*]] = stablehlo.multiply %[[V21]], %[[V24]] : tensor<2xf32>
// TUPLE: %[[V29:.*]] = stablehlo.multiply %[[V21]], %[[V25]] : tensor<2xf32>
// TUPLE: %[[V30:.*]] = stablehlo.subtract %[[V26]], %[[V17]] : tensor<2xf32>
// TUPLE: %[[V31:.*]] = stablehlo.add %[[V27]], %[[V28]] : tensor<2xf32>
// TUPLE: %[[V32:.*]] = stablehlo.add %[[V30]], %[[V31]] : tensor<2xf32>
// TUPLE: %[[V33:.*]] = stablehlo.add %[[V32]], %[[V29]] : tensor<2xf32>
// TUPLE: %[[V34:.*]] = stablehlo.multiply %[[V12]], %[[V16]] : tensor<2xf32>
// TUPLE: %[[V35:.*]] = stablehlo.multiply %[[V13]], %[[V15]] : tensor<2xf32>
// TUPLE: %[[V36:.*]] = stablehlo.add %[[V34]], %[[V35]] : tensor<2xf32>
// TUPLE: %[[V37:.*]] = stablehlo.add %[[V33]], %[[V36]] : tensor<2xf32>
// TUPLE: %[[V38:.*]] = stablehlo.add %[[V17]], %[[V37]] : tensor<2xf32>
// TUPLE: %[[V39:.*]] = stablehlo.subtract %[[V38]], %[[V17]] : tensor<2xf32>
// TUPLE: %[[V40:.*]] = stablehlo.subtract %[[V37]], %[[V39]] : tensor<2xf32>
// TUPLE: %[[V41:.*]] = stablehlo.tuple %[[V38]], %[[V40]] : tuple<tensor<2xf32>, tensor<2xf32>>
// TUPLE: %[[V42:.*]] = stablehlo.slice %[[V38]] [0:1] : (tensor<2xf32>) -> tensor<1xf32>
// TUPLE: %[[V43:.*]] = stablehlo.slice %[[V40]] [0:1] : (tensor<2xf32>) -> tensor<1xf32>
// TUPLE: %[[V44:.*]] = stablehlo.tuple %[[V42]], %[[V43]] : tuple<tensor<1xf32>, tensor<1xf32>>
// TUPLE: %[[V45:.*]] = stablehlo.reshape %[[V42]] : (tensor<1xf32>) -> tensor<f32>
// TUPLE: %[[V46:.*]] = stablehlo.reshape %[[V43]] : (tensor<1xf32>) -> tensor<f32>
// TUPLE: %[[V47:.*]] = stablehlo.tuple %[[V45]], %[[V46]] : tuple<tensor<f32>, tensor<f32>>
// TUPLE: %[[V48:.*]] = stablehlo.slice %[[V38]] [1:2] : (tensor<2xf32>) -> tensor<1xf32>
// TUPLE: %[[V49:.*]] = stablehlo.slice %[[V40]] [1:2] : (tensor<2xf32>) -> tensor<1xf32>
// TUPLE: %[[V50:.*]] = stablehlo.tuple %[[V48]], %[[V49]] : tuple<tensor<1xf32>, tensor<1xf32>>
// TUPLE: %[[V51:.*]] = stablehlo.reshape %[[V48]] : (tensor<1xf32>) -> tensor<f32>
// TUPLE: %[[V52:.*]] = stablehlo.reshape %[[V49]] : (tensor<1xf32>) -> tensor<f32>
// TUPLE: %[[V53:.*]] = stablehlo.tuple %[[V51]], %[[V52]] : tuple<tensor<f32>, tensor<f32>>
// TUPLE: %[[V54:.*]] = stablehlo.add %[[V45]], %[[V51]] : tensor<f32>
// TUPLE: %[[V55:.*]] = stablehlo.subtract %[[V54]], %[[V51]] : tensor<f32>
// TUPLE: %[[V56:.*]] = stablehlo.subtract %[[V54]], %[[V55]] : tensor<f32>
// TUPLE: %[[V57:.*]] = stablehlo.subtract %[[V45]], %[[V55]] : tensor<f32>
// TUPLE: %[[V58:.*]] = stablehlo.subtract %[[V51]], %[[V56]] : tensor<f32>
// TUPLE: %[[V59:.*]] = stablehlo.add %[[V57]], %[[V58]] : tensor<f32>
// TUPLE: %[[V60:.*]] = stablehlo.add %[[V46]], %[[V52]] : tensor<f32>
// TUPLE: %[[V61:.*]] = stablehlo.subtract %[[V60]], %[[V52]] : tensor<f32>
// TUPLE: %[[V62:.*]] = stablehlo.subtract %[[V60]], %[[V61]] : tensor<f32>
// TUPLE: %[[V63:.*]] = stablehlo.subtract %[[V46]], %[[V61]] : tensor<f32>
// TUPLE: %[[V64:.*]] = stablehlo.subtract %[[V52]], %[[V62]] : tensor<f32>
// TUPLE: %[[V65:.*]] = stablehlo.add %[[V63]], %[[V64]] : tensor<f32>
// TUPLE: %[[V66:.*]] = stablehlo.add %[[V54]], %[[V60]] : tensor<f32>
// TUPLE: %[[V67:.*]] = stablehlo.subtract %[[V66]], %[[V54]] : tensor<f32>
// TUPLE: %[[V68:.*]] = stablehlo.subtract %[[V60]], %[[V67]] : tensor<f32>
// TUPLE: %[[V69:.*]] = stablehlo.add %[[V59]], %[[V65]] : tensor<f32>
// TUPLE: %[[V70:.*]] = stablehlo.add %[[V69]], %[[V68]] : tensor<f32>
// TUPLE: %[[V71:.*]] = stablehlo.add %[[V66]], %[[V70]] : tensor<f32>
// TUPLE: %[[V72:.*]] = stablehlo.subtract %[[V71]], %[[V66]] : tensor<f32>
// TUPLE: %[[V73:.*]] = stablehlo.subtract %[[V70]], %[[V72]] : tensor<f32>
// TUPLE: %[[V74:.*]] = stablehlo.tuple %[[V71]], %[[V73]] : tuple<tensor<f32>, tensor<f32>>
// TUPLE: %[[V75:.*]] = stablehlo.convert %[[V71]] : (tensor<f32>) -> tensor<f64>
// TUPLE: %[[V76:.*]] = stablehlo.convert %[[V73]] : (tensor<f32>) -> tensor<f64>
// TUPLE: %[[V77:.*]] = stablehlo.add %[[V75]], %[[V76]] : tensor<f64>
// TUPLE: return %[[V77]] : tensor<f64>
// TUPLE: }

// FIRST: module {
// FIRST: func.func @dot_general(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>) -> tensor<f64> {
// FIRST: %[[V0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// FIRST: %[[V1:.*]] = stablehlo.convert %[[V0]] : (tensor<2xf32>) -> tensor<2xf64>
// FIRST: %[[V2:.*]] = stablehlo.subtract %arg0, %[[V1]] : tensor<2xf64>
// FIRST: %[[V3:.*]] = stablehlo.convert %[[V2]] : (tensor<2xf64>) -> tensor<2xf32>
// FIRST: %[[V4:.*]] = stablehlo.reshape %[[V0]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST: %[[V5:.*]] = stablehlo.reshape %[[V3]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST: %[[V6:.*]] = stablehlo.concatenate %[[V4]], %[[V5]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST: %[[V7:.*]] = stablehlo.convert %arg1 : (tensor<2xf64>) -> tensor<2xf32>
// FIRST: %[[V8:.*]] = stablehlo.convert %[[V7]] : (tensor<2xf32>) -> tensor<2xf64>
// FIRST: %[[V9:.*]] = stablehlo.subtract %arg1, %[[V8]] : tensor<2xf64>
// FIRST: %[[V10:.*]] = stablehlo.convert %[[V9]] : (tensor<2xf64>) -> tensor<2xf32>
// FIRST: %[[V11:.*]] = stablehlo.reshape %[[V7]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST: %[[V12:.*]] = stablehlo.reshape %[[V10]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST: %[[V13:.*]] = stablehlo.concatenate %[[V11]], %[[V12]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST: %[[V14:.*]] = stablehlo.broadcast_in_dim %[[V6]], dims = [0, 1] : (tensor<2x2xf32>) -> tensor<2x2xf32>
// FIRST: %[[V15:.*]] = stablehlo.slice %[[V14]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST: %[[V16:.*]] = stablehlo.slice %[[V14]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST: %[[V17:.*]] = stablehlo.slice %[[V13]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST: %[[V18:.*]] = stablehlo.slice %[[V13]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST: %[[V19:.*]] = stablehlo.multiply %[[V15]], %[[V17]] : tensor<1x2xf32>
// FIRST: %[[Vcst:.*]] = stablehlo.constant dense<4.097000e+03> : tensor<1x2xf32>
// FIRST: %[[V20:.*]] = stablehlo.multiply %[[V15]], %[[Vcst]] : tensor<1x2xf32>
// FIRST: %[[V21:.*]] = stablehlo.subtract %[[V20]], %[[V15]] : tensor<1x2xf32>
// FIRST: %[[V22:.*]] = stablehlo.subtract %[[V20]], %[[V21]] : tensor<1x2xf32>
// FIRST: %[[V23:.*]] = stablehlo.subtract %[[V15]], %[[V22]] : tensor<1x2xf32>
// FIRST: %[[Vcst_0:.*]] = stablehlo.constant dense<4.097000e+03> : tensor<1x2xf32>
// FIRST: %[[V24:.*]] = stablehlo.multiply %[[V17]], %[[Vcst_0]] : tensor<1x2xf32>
// FIRST: %[[V25:.*]] = stablehlo.subtract %[[V24]], %[[V17]] : tensor<1x2xf32>
// FIRST: %[[V26:.*]] = stablehlo.subtract %[[V24]], %[[V25]] : tensor<1x2xf32>
// FIRST: %[[V27:.*]] = stablehlo.subtract %[[V17]], %[[V26]] : tensor<1x2xf32>
// FIRST: %[[V28:.*]] = stablehlo.multiply %[[V22]], %[[V26]] : tensor<1x2xf32>
// FIRST: %[[V29:.*]] = stablehlo.multiply %[[V22]], %[[V27]] : tensor<1x2xf32>
// FIRST: %[[V30:.*]] = stablehlo.multiply %[[V23]], %[[V26]] : tensor<1x2xf32>
// FIRST: %[[V31:.*]] = stablehlo.multiply %[[V23]], %[[V27]] : tensor<1x2xf32>
// FIRST: %[[V32:.*]] = stablehlo.subtract %[[V28]], %[[V19]] : tensor<1x2xf32>
// FIRST: %[[V33:.*]] = stablehlo.add %[[V29]], %[[V30]] : tensor<1x2xf32>
// FIRST: %[[V34:.*]] = stablehlo.add %[[V32]], %[[V33]] : tensor<1x2xf32>
// FIRST: %[[V35:.*]] = stablehlo.add %[[V34]], %[[V31]] : tensor<1x2xf32>
// FIRST: %[[V36:.*]] = stablehlo.multiply %[[V15]], %[[V18]] : tensor<1x2xf32>
// FIRST: %[[V37:.*]] = stablehlo.multiply %[[V16]], %[[V17]] : tensor<1x2xf32>
// FIRST: %[[V38:.*]] = stablehlo.add %[[V36]], %[[V37]] : tensor<1x2xf32>
// FIRST: %[[V39:.*]] = stablehlo.add %[[V35]], %[[V38]] : tensor<1x2xf32>
// FIRST: %[[V40:.*]] = stablehlo.add %[[V19]], %[[V39]] : tensor<1x2xf32>
// FIRST: %[[V41:.*]] = stablehlo.subtract %[[V40]], %[[V19]] : tensor<1x2xf32>
// FIRST: %[[V42:.*]] = stablehlo.subtract %[[V39]], %[[V41]] : tensor<1x2xf32>
// FIRST: %[[V43:.*]] = stablehlo.concatenate %[[V40]], %[[V42]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST: %[[V44:.*]] = stablehlo.slice %[[V43]] [0:2, 0:1] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// FIRST: %[[V45:.*]] = stablehlo.reshape %[[V44]] : (tensor<2x1xf32>) -> tensor<2xf32>
// FIRST: %[[V46:.*]] = stablehlo.slice %[[V43]] [0:2, 1:2] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// FIRST: %[[V47:.*]] = stablehlo.reshape %[[V46]] : (tensor<2x1xf32>) -> tensor<2xf32>
// FIRST: %[[V48:.*]] = stablehlo.slice %[[V45]] [0:1] : (tensor<2xf32>) -> tensor<1xf32>
// FIRST: %[[V49:.*]] = stablehlo.slice %[[V45]] [1:2] : (tensor<2xf32>) -> tensor<1xf32>
// FIRST: %[[V50:.*]] = stablehlo.slice %[[V47]] [0:1] : (tensor<2xf32>) -> tensor<1xf32>
// FIRST: %[[V51:.*]] = stablehlo.slice %[[V47]] [1:2] : (tensor<2xf32>) -> tensor<1xf32>
// FIRST: %[[V52:.*]] = stablehlo.add %[[V48]], %[[V50]] : tensor<1xf32>
// FIRST: %[[V53:.*]] = stablehlo.subtract %[[V52]], %[[V50]] : tensor<1xf32>
// FIRST: %[[V54:.*]] = stablehlo.subtract %[[V52]], %[[V53]] : tensor<1xf32>
// FIRST: %[[V55:.*]] = stablehlo.subtract %[[V48]], %[[V53]] : tensor<1xf32>
// FIRST: %[[V56:.*]] = stablehlo.subtract %[[V50]], %[[V54]] : tensor<1xf32>
// FIRST: %[[V57:.*]] = stablehlo.add %[[V55]], %[[V56]] : tensor<1xf32>
// FIRST: %[[V58:.*]] = stablehlo.add %[[V49]], %[[V51]] : tensor<1xf32>
// FIRST: %[[V59:.*]] = stablehlo.subtract %[[V58]], %[[V51]] : tensor<1xf32>
// FIRST: %[[V60:.*]] = stablehlo.subtract %[[V58]], %[[V59]] : tensor<1xf32>
// FIRST: %[[V61:.*]] = stablehlo.subtract %[[V49]], %[[V59]] : tensor<1xf32>
// FIRST: %[[V62:.*]] = stablehlo.subtract %[[V51]], %[[V60]] : tensor<1xf32>
// FIRST: %[[V63:.*]] = stablehlo.add %[[V61]], %[[V62]] : tensor<1xf32>
// FIRST: %[[V64:.*]] = stablehlo.add %[[V52]], %[[V58]] : tensor<1xf32>
// FIRST: %[[V65:.*]] = stablehlo.subtract %[[V64]], %[[V52]] : tensor<1xf32>
// FIRST: %[[V66:.*]] = stablehlo.subtract %[[V58]], %[[V65]] : tensor<1xf32>
// FIRST: %[[V67:.*]] = stablehlo.add %[[V57]], %[[V63]] : tensor<1xf32>
// FIRST: %[[V68:.*]] = stablehlo.add %[[V67]], %[[V66]] : tensor<1xf32>
// FIRST: %[[V69:.*]] = stablehlo.add %[[V64]], %[[V68]] : tensor<1xf32>
// FIRST: %[[V70:.*]] = stablehlo.subtract %[[V69]], %[[V64]] : tensor<1xf32>
// FIRST: %[[V71:.*]] = stablehlo.subtract %[[V68]], %[[V70]] : tensor<1xf32>
// FIRST: %[[V72:.*]] = stablehlo.concatenate %[[V69]], %[[V71]], dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
// FIRST: %[[V73:.*]] = stablehlo.convert %[[V72]] : (tensor<2xf32>) -> tensor<2xf64>
// FIRST: %[[Vcst_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// FIRST: %[[V74:.*]] = stablehlo.reduce(%[[V73]] init: %[[Vcst_1]]) applies stablehlo.add across dimensions = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<f64>
// FIRST: return %[[V74]] : tensor<f64>
// FIRST: }

// LAST: module {
// LAST: func.func @dot_general(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>) -> tensor<f64> {
// LAST: %[[V0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// LAST: %[[V1:.*]] = stablehlo.convert %[[V0]] : (tensor<2xf32>) -> tensor<2xf64>
// LAST: %[[V2:.*]] = stablehlo.subtract %arg0, %[[V1]] : tensor<2xf64>
// LAST: %[[V3:.*]] = stablehlo.convert %[[V2]] : (tensor<2xf64>) -> tensor<2xf32>
// LAST: %[[V4:.*]] = stablehlo.reshape %[[V0]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST: %[[V5:.*]] = stablehlo.reshape %[[V3]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST: %[[V6:.*]] = stablehlo.concatenate %[[V4]], %[[V5]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// LAST: %[[V7:.*]] = stablehlo.convert %arg1 : (tensor<2xf64>) -> tensor<2xf32>
// LAST: %[[V8:.*]] = stablehlo.convert %[[V7]] : (tensor<2xf32>) -> tensor<2xf64>
// LAST: %[[V9:.*]] = stablehlo.subtract %arg1, %[[V8]] : tensor<2xf64>
// LAST: %[[V10:.*]] = stablehlo.convert %[[V9]] : (tensor<2xf64>) -> tensor<2xf32>
// LAST: %[[V11:.*]] = stablehlo.reshape %[[V7]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST: %[[V12:.*]] = stablehlo.reshape %[[V10]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST: %[[V13:.*]] = stablehlo.concatenate %[[V11]], %[[V12]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// LAST: %[[V14:.*]] = stablehlo.broadcast_in_dim %[[V6]], dims = [0, 1] : (tensor<2x2xf32>) -> tensor<2x2xf32>
// LAST: %[[V15:.*]] = stablehlo.slice %[[V14]] [0:2, 0:1] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// LAST: %[[V16:.*]] = stablehlo.slice %[[V14]] [0:2, 1:2] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// LAST: %[[V17:.*]] = stablehlo.slice %[[V13]] [0:2, 0:1] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// LAST: %[[V18:.*]] = stablehlo.slice %[[V13]] [0:2, 1:2] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// LAST: %[[V19:.*]] = stablehlo.multiply %[[V15]], %[[V17]] : tensor<2x1xf32>
// LAST: %[[Vcst:.*]] = stablehlo.constant dense<4.097000e+03> : tensor<2x1xf32>
// LAST: %[[V20:.*]] = stablehlo.multiply %[[V15]], %[[Vcst]] : tensor<2x1xf32>
// LAST: %[[V21:.*]] = stablehlo.subtract %[[V20]], %[[V15]] : tensor<2x1xf32>
// LAST: %[[V22:.*]] = stablehlo.subtract %[[V20]], %[[V21]] : tensor<2x1xf32>
// LAST: %[[V23:.*]] = stablehlo.subtract %[[V15]], %[[V22]] : tensor<2x1xf32>
// LAST: %[[Vcst_0:.*]] = stablehlo.constant dense<4.097000e+03> : tensor<2x1xf32>
// LAST: %[[V24:.*]] = stablehlo.multiply %[[V17]], %[[Vcst_0]] : tensor<2x1xf32>
// LAST: %[[V25:.*]] = stablehlo.subtract %[[V24]], %[[V17]] : tensor<2x1xf32>
// LAST: %[[V26:.*]] = stablehlo.subtract %[[V24]], %[[V25]] : tensor<2x1xf32>
// LAST: %[[V27:.*]] = stablehlo.subtract %[[V17]], %[[V26]] : tensor<2x1xf32>
// LAST: %[[V28:.*]] = stablehlo.multiply %[[V22]], %[[V26]] : tensor<2x1xf32>
// LAST: %[[V29:.*]] = stablehlo.multiply %[[V22]], %[[V27]] : tensor<2x1xf32>
// LAST: %[[V30:.*]] = stablehlo.multiply %[[V23]], %[[V26]] : tensor<2x1xf32>
// LAST: %[[V31:.*]] = stablehlo.multiply %[[V23]], %[[V27]] : tensor<2x1xf32>
// LAST: %[[V32:.*]] = stablehlo.subtract %[[V28]], %[[V19]] : tensor<2x1xf32>
// LAST: %[[V33:.*]] = stablehlo.add %[[V29]], %[[V30]] : tensor<2x1xf32>
// LAST: %[[V34:.*]] = stablehlo.add %[[V32]], %[[V33]] : tensor<2x1xf32>
// LAST: %[[V35:.*]] = stablehlo.add %[[V34]], %[[V31]] : tensor<2x1xf32>
// LAST: %[[V36:.*]] = stablehlo.multiply %[[V15]], %[[V18]] : tensor<2x1xf32>
// LAST: %[[V37:.*]] = stablehlo.multiply %[[V16]], %[[V17]] : tensor<2x1xf32>
// LAST: %[[V38:.*]] = stablehlo.add %[[V36]], %[[V37]] : tensor<2x1xf32>
// LAST: %[[V39:.*]] = stablehlo.add %[[V35]], %[[V38]] : tensor<2x1xf32>
// LAST: %[[V40:.*]] = stablehlo.add %[[V19]], %[[V39]] : tensor<2x1xf32>
// LAST: %[[V41:.*]] = stablehlo.subtract %[[V40]], %[[V19]] : tensor<2x1xf32>
// LAST: %[[V42:.*]] = stablehlo.subtract %[[V39]], %[[V41]] : tensor<2x1xf32>
// LAST: %[[V43:.*]] = stablehlo.concatenate %[[V40]], %[[V42]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// LAST: %[[V44:.*]] = stablehlo.slice %[[V43]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// LAST: %[[V45:.*]] = stablehlo.reshape %[[V44]] : (tensor<1x2xf32>) -> tensor<2xf32>
// LAST: %[[V46:.*]] = stablehlo.slice %[[V43]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// LAST: %[[V47:.*]] = stablehlo.reshape %[[V46]] : (tensor<1x2xf32>) -> tensor<2xf32>
// LAST: %[[V48:.*]] = stablehlo.slice %[[V45]] [0:1] : (tensor<2xf32>) -> tensor<1xf32>
// LAST: %[[V49:.*]] = stablehlo.slice %[[V45]] [1:2] : (tensor<2xf32>) -> tensor<1xf32>
// LAST: %[[V50:.*]] = stablehlo.slice %[[V47]] [0:1] : (tensor<2xf32>) -> tensor<1xf32>
// LAST: %[[V51:.*]] = stablehlo.slice %[[V47]] [1:2] : (tensor<2xf32>) -> tensor<1xf32>
// LAST: %[[V52:.*]] = stablehlo.add %[[V48]], %[[V50]] : tensor<1xf32>
// LAST: %[[V53:.*]] = stablehlo.subtract %[[V52]], %[[V50]] : tensor<1xf32>
// LAST: %[[V54:.*]] = stablehlo.subtract %[[V52]], %[[V53]] : tensor<1xf32>
// LAST: %[[V55:.*]] = stablehlo.subtract %[[V48]], %[[V53]] : tensor<1xf32>
// LAST: %[[V56:.*]] = stablehlo.subtract %[[V50]], %[[V54]] : tensor<1xf32>
// LAST: %[[V57:.*]] = stablehlo.add %[[V55]], %[[V56]] : tensor<1xf32>
// LAST: %[[V58:.*]] = stablehlo.add %[[V49]], %[[V51]] : tensor<1xf32>
// LAST: %[[V59:.*]] = stablehlo.subtract %[[V58]], %[[V51]] : tensor<1xf32>
// LAST: %[[V60:.*]] = stablehlo.subtract %[[V58]], %[[V59]] : tensor<1xf32>
// LAST: %[[V61:.*]] = stablehlo.subtract %[[V49]], %[[V59]] : tensor<1xf32>
// LAST: %[[V62:.*]] = stablehlo.subtract %[[V51]], %[[V60]] : tensor<1xf32>
// LAST: %[[V63:.*]] = stablehlo.add %[[V61]], %[[V62]] : tensor<1xf32>
// LAST: %[[V64:.*]] = stablehlo.add %[[V52]], %[[V58]] : tensor<1xf32>
// LAST: %[[V65:.*]] = stablehlo.subtract %[[V64]], %[[V52]] : tensor<1xf32>
// LAST: %[[V66:.*]] = stablehlo.subtract %[[V58]], %[[V65]] : tensor<1xf32>
// LAST: %[[V67:.*]] = stablehlo.add %[[V57]], %[[V63]] : tensor<1xf32>
// LAST: %[[V68:.*]] = stablehlo.add %[[V67]], %[[V66]] : tensor<1xf32>
// LAST: %[[V69:.*]] = stablehlo.add %[[V64]], %[[V68]] : tensor<1xf32>
// LAST: %[[V70:.*]] = stablehlo.subtract %[[V69]], %[[V64]] : tensor<1xf32>
// LAST: %[[V71:.*]] = stablehlo.subtract %[[V68]], %[[V70]] : tensor<1xf32>
// LAST: %[[V72:.*]] = stablehlo.concatenate %[[V69]], %[[V71]], dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
// LAST: %[[V73:.*]] = stablehlo.convert %[[V72]] : (tensor<2xf32>) -> tensor<2xf64>
// LAST: %[[Vcst_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// LAST: %[[V74:.*]] = stablehlo.reduce(%[[V73]] init: %[[Vcst_1]]) applies stablehlo.add across dimensions = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<f64>
// LAST: return %[[V74]] : tensor<f64>
// LAST: }

// FIRST-LABEL: func.func @main

// LAST-LABEL: func.func @main

// TUPLE-LABEL: func.func @main

func.func @main() attributes {enzyme.no_multifloat} {
  %c1 = stablehlo.constant dense<[1.10000001, 2.2]> : tensor<2xf64>
  %c2 = stablehlo.constant dense<[-1.1, 1.0]> : tensor<2xf64>
  
  %expected_mf = stablehlo.constant dense<0.989999989000001> : tensor<f64>
  
  %res = func.call @dot_general(%c1, %c2) : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
  
  // Strict test against Julia MultiFloat
  "check.expect_close"(%res, %expected_mf) {max_ulp_difference = 20 : ui64} : (tensor<f64>, tensor<f64>) -> ()
  
  // Approximate test against regular f64
  %expected_f64 = stablehlo.constant dense<0.989999989> : tensor<f64>
  "check.expect_close"(%res, %expected_f64) {max_ulp_difference = 20 : ui64} : (tensor<f64>, tensor<f64>) -> ()
  return
}
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<2xf64>
// FIRST:     %[[CST_0:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<2xf64>
// FIRST:     %[[CST_1:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<f64>
// FIRST:     %[[V_0:.*]] = call @dot_general(%[[CST]], %[[CST_0]]) : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// FIRST:     check.expect_close %[[V_0]], %[[CST_1]], max_ulp_difference = 20 : tensor<f64>, tensor<f64>
// FIRST:     %[[CST_2:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<f64>
// FIRST:     check.expect_close %[[V_0]], %[[CST_2]], max_ulp_difference = 20 : tensor<f64>, tensor<f64>
// FIRST:     return
