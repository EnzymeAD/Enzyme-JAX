// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck --check-prefix=FIRST_LIMB %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first precise-reduce=true" %s | FileCheck --check-prefix=FIRST_PRECISE %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=LAST_LIMB %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last precise-reduce=true" %s | FileCheck --check-prefix=LAST_PRECISE %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=TUPLE %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first precise-reduce=true" %s | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple precise-reduce=true" %s | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @reduce_test(%arg0: tensor<2x2xf64>) -> tensor<2xf64> {
  %cst = stablehlo.constant dense<0.0> : tensor<f64>
  %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
  return %0 : tensor<2xf64>
}

// FIRST_LIMB-LABEL: func.func @reduce_test
// FIRST_LIMB: %[[INPUT:.*]] = stablehlo.concatenate
// FIRST_LIMB: %[[RED:.*]] = stablehlo.reduce(%[[INPUT]] init: %{{.*}}) applies stablehlo.add across dimensions = [1]
// FIRST_LIMB: %[[HI:.*]] = stablehlo.slice %[[RED]] [0:1, 0:2]
// FIRST_LIMB: %[[LO:.*]] = stablehlo.slice %[[RED]] [1:2, 0:2]
// FIRST_LIMB: %[[RESHAPE_HI:.*]] = stablehlo.reshape %[[HI]] : (tensor<1x2xf32>) -> tensor<2xf32>
// FIRST_LIMB: %[[RESHAPE_LO:.*]] = stablehlo.reshape %[[LO]] : (tensor<1x2xf32>) -> tensor<2xf32>
// FIRST_LIMB: %[[BACK_HI:.*]] = stablehlo.reshape %[[RESHAPE_HI]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST_LIMB: %[[BACK_LO:.*]] = stablehlo.reshape %[[RESHAPE_LO]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST_LIMB: %[[SUM:.*]] = stablehlo.add %[[BACK_HI]], %[[BACK_LO]]
// FIRST_LIMB: %[[SUB1:.*]] = stablehlo.subtract %[[SUM]], %[[BACK_LO]]
// FIRST_LIMB: %[[SUB2:.*]] = stablehlo.subtract %[[SUM]], %[[SUB1]]
// FIRST_LIMB: %[[SUB3:.*]] = stablehlo.subtract %[[BACK_HI]], %[[SUB1]]
// FIRST_LIMB: %[[SUB4:.*]] = stablehlo.subtract %[[BACK_LO]], %[[SUB2]]
// FIRST_LIMB: %[[ERR:.*]] = stablehlo.add %[[SUB3]], %[[SUB4]]
// FIRST_LIMB: stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST_LIMB: %[[FINAL:.*]] = stablehlo.reduce(%{{.*}} init: %{{.*}}) applies stablehlo.add across dimensions = [0]
// FIRST_LIMB: return %[[FINAL]]

// FIRST_PRECISE-LABEL: func.func @reduce_test
// FIRST_PRECISE: %[[INPUT:.*]] = stablehlo.concatenate
// FIRST_PRECISE: %[[HI:.*]] = stablehlo.slice %[[INPUT]] [0:1, 0:2, 0:2]
// FIRST_PRECISE: %[[LO:.*]] = stablehlo.slice %[[INPUT]] [1:2, 0:2, 0:2]
// FIRST_PRECISE: stablehlo.reduce(%[[HI]] init: %{{.*}}), (%[[LO]] init: %{{.*}}) across dimensions = [1]

// LAST_LIMB-LABEL: func.func @reduce_test
// LAST_LIMB: %[[INPUT:.*]] = stablehlo.concatenate
// LAST_LIMB: %[[RED:.*]] = stablehlo.reduce(%[[INPUT]] init: %{{.*}}) applies stablehlo.add across dimensions = [0]
// LAST_LIMB: %[[HI:.*]] = stablehlo.slice %[[RED]] [0:2, 0:1]
// LAST_LIMB: %[[LO:.*]] = stablehlo.slice %[[RED]] [0:2, 1:2]
// LAST_LIMB: %[[RESHAPE_HI:.*]] = stablehlo.reshape %[[HI]] : (tensor<2x1xf32>) -> tensor<2xf32>
// LAST_LIMB: %[[RESHAPE_LO:.*]] = stablehlo.reshape %[[LO]] : (tensor<2x1xf32>) -> tensor<2xf32>
// LAST_LIMB: %[[BACK_HI:.*]] = stablehlo.reshape %[[RESHAPE_HI]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST_LIMB: %[[BACK_LO:.*]] = stablehlo.reshape %[[RESHAPE_LO]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST_LIMB: %[[SUM:.*]] = stablehlo.add %[[BACK_HI]], %[[BACK_LO]]
// LAST_LIMB: %[[SUB1:.*]] = stablehlo.subtract %[[SUM]], %[[BACK_LO]]
// LAST_LIMB: %[[SUB2:.*]] = stablehlo.subtract %[[SUM]], %[[SUB1]]
// LAST_LIMB: %[[SUB3:.*]] = stablehlo.subtract %[[BACK_HI]], %[[SUB1]]
// LAST_LIMB: %[[SUB4:.*]] = stablehlo.subtract %[[BACK_LO]], %[[SUB2]]
// LAST_LIMB: %[[ERR:.*]] = stablehlo.add %[[SUB3]], %[[SUB4]]
// LAST_LIMB: stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// LAST_LIMB: %[[FINAL:.*]] = stablehlo.reduce(%{{.*}} init: %{{.*}}) applies stablehlo.add across dimensions = [1]
// LAST_LIMB: return %[[FINAL]]


// LAST_PRECISE-LABEL: func.func @reduce_test
// LAST_PRECISE: %[[INPUT:.*]] = stablehlo.concatenate
// LAST_PRECISE: %[[HI:.*]] = stablehlo.slice %[[INPUT]] [0:2, 0:2, 0:1]
// LAST_PRECISE: %[[LO:.*]] = stablehlo.slice %[[INPUT]] [0:2, 0:2, 1:2]
// LAST_PRECISE: stablehlo.reduce(%[[HI]] init: %{{.*}}), (%[[LO]] init: %{{.*}}) across dimensions = [0]

// TUPLE-LABEL: func.func @reduce_test
// TUPLE: %[[V0:.*]] = stablehlo.convert %arg0 : (tensor<2x2xf64>) -> tensor<2x2xf32>
// TUPLE: %[[V1:.*]] = stablehlo.convert %[[V0]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
// TUPLE: %[[V2:.*]] = stablehlo.subtract %arg0, %[[V1]] : tensor<2x2xf64>
// TUPLE: %[[V3:.*]] = stablehlo.convert %[[V2]] : (tensor<2x2xf64>) -> tensor<2x2xf32>
// TUPLE: %[[V4:.*]] = stablehlo.tuple %[[V0]], %[[V3]] : tuple<tensor<2x2xf32>, tensor<2x2xf32>>
// TUPLE: %[[Vcst:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// TUPLE: %[[Vcst_0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// TUPLE: %[[V5:.*]] = stablehlo.tuple %[[Vcst]], %[[Vcst_0]] : tuple<tensor<f32>, tensor<f32>>
// TUPLE: %[[V6:.*]] = stablehlo.get_tuple_element %[[V4]][0] : (tuple<tensor<2x2xf32>, tensor<2x2xf32>>) -> tensor<2x2xf32>
// TUPLE: %[[V7:.*]] = stablehlo.get_tuple_element %[[V4]][1] : (tuple<tensor<2x2xf32>, tensor<2x2xf32>>) -> tensor<2x2xf32>
// TUPLE: %[[V8:.*]] = stablehlo.reduce(%[[V6]] init: %[[Vcst]]) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
// TUPLE: %[[V9:.*]] = stablehlo.reduce(%[[V7]] init: %[[Vcst_0]]) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
// TUPLE: %[[Vcst_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// TUPLE: %[[V10:.*]] = stablehlo.tuple %[[V8]], %[[Vcst_1]] : tuple<tensor<2xf32>, tensor<2xf32>>
// TUPLE: %[[Vcst_2:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// TUPLE: %[[V11:.*]] = stablehlo.tuple %[[V9]], %[[Vcst_2]] : tuple<tensor<2xf32>, tensor<2xf32>>
// TUPLE: %[[V12:.*]] = stablehlo.add %[[V8]], %[[V9]] : tensor<2xf32>
// TUPLE: %[[V13:.*]] = stablehlo.subtract %[[V12]], %[[V9]] : tensor<2xf32>
// TUPLE: %[[V14:.*]] = stablehlo.subtract %[[V12]], %[[V13]] : tensor<2xf32>
// TUPLE: %[[V15:.*]] = stablehlo.subtract %[[V8]], %[[V13]] : tensor<2xf32>
// TUPLE: %[[V16:.*]] = stablehlo.subtract %[[V9]], %[[V14]] : tensor<2xf32>
// TUPLE: %[[V17:.*]] = stablehlo.add %[[V15]], %[[V16]] : tensor<2xf32>
// TUPLE: %[[V18:.*]] = stablehlo.add %[[Vcst_1]], %[[Vcst_2]] : tensor<2xf32>
// TUPLE: %[[V19:.*]] = stablehlo.subtract %[[V18]], %[[Vcst_2]] : tensor<2xf32>
// TUPLE: %[[V20:.*]] = stablehlo.subtract %[[V18]], %[[V19]] : tensor<2xf32>
// TUPLE: %[[V21:.*]] = stablehlo.subtract %[[Vcst_1]], %[[V19]] : tensor<2xf32>
// TUPLE: %[[V22:.*]] = stablehlo.subtract %[[Vcst_2]], %[[V20]] : tensor<2xf32>
// TUPLE: %[[V23:.*]] = stablehlo.add %[[V21]], %[[V22]] : tensor<2xf32>
// TUPLE: %[[V24:.*]] = stablehlo.add %[[V12]], %[[V18]] : tensor<2xf32>
// TUPLE: %[[V25:.*]] = stablehlo.subtract %[[V24]], %[[V12]] : tensor<2xf32>
// TUPLE: %[[V26:.*]] = stablehlo.subtract %[[V18]], %[[V25]] : tensor<2xf32>
// TUPLE: %[[V27:.*]] = stablehlo.add %[[V17]], %[[V23]] : tensor<2xf32>
// TUPLE: %[[V28:.*]] = stablehlo.add %[[V27]], %[[V26]] : tensor<2xf32>
// TUPLE: %[[V29:.*]] = stablehlo.add %[[V24]], %[[V28]] : tensor<2xf32>
// TUPLE: %[[V30:.*]] = stablehlo.subtract %[[V29]], %[[V24]] : tensor<2xf32>
// TUPLE: %[[V31:.*]] = stablehlo.subtract %[[V28]], %[[V30]] : tensor<2xf32>
// TUPLE: %[[V32:.*]] = stablehlo.tuple %[[V29]], %[[V31]] : tuple<tensor<2xf32>, tensor<2xf32>>
// TUPLE: %[[V33:.*]] = stablehlo.convert %[[V29]] : (tensor<2xf32>) -> tensor<2xf64>
// TUPLE: %[[V34:.*]] = stablehlo.convert %[[V31]] : (tensor<2xf32>) -> tensor<2xf64>
// TUPLE: %[[V35:.*]] = stablehlo.add %[[V33]], %[[V34]] : tensor<2xf64>
// TUPLE: return %[[V35]] : tensor<2xf64>

func.func @reduce_2d_test(%arg0: tensor<2x2xf64>) -> tensor<f64> {
  %cst = stablehlo.constant dense<0.0> : tensor<f64>
  %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [0, 1] : (tensor<2x2xf64>, tensor<f64>) -> tensor<f64>
  return %0 : tensor<f64>
}

// FIRST_PRECISE-LABEL: func.func @reduce_2d_test
// FIRST_PRECISE: %[[INPUT:.*]] = stablehlo.concatenate
// FIRST_PRECISE: %[[HI:.*]] = stablehlo.slice %[[INPUT]] [0:1, 0:2, 0:2]
// FIRST_PRECISE: %[[LO:.*]] = stablehlo.slice %[[INPUT]] [1:2, 0:2, 0:2]
// FIRST_PRECISE: %[[RED:.*]]:2 = stablehlo.reduce(%[[HI]] init: %{{.*}}), (%[[LO]] init: %{{.*}}) across dimensions = [1, 2]
// FIRST_PRECISE:  reducer(%[[ARG1:.*]]: tensor<f32>, %[[ARG3:.*]]: tensor<f32>) (%[[ARG2:.*]]: tensor<f32>, %[[ARG4:.*]]: tensor<f32>)  {
// FIRST_PRECISE:   %[[CMP1:.*]] = stablehlo.compare GT, %[[ARG1]], %[[ARG3]]
// FIRST_PRECISE:   %[[CMP2:.*]] = stablehlo.compare EQ, %[[ARG1]], %[[ARG3]]
// FIRST_PRECISE:   %[[CMP3:.*]] = stablehlo.compare GT, %[[ARG2]], %[[ARG4]]
// FIRST_PRECISE:   %[[SEL1:.*]] = stablehlo.select %[[CMP2]], %[[CMP3]], %[[CMP1]]
// FIRST_PRECISE:   %[[SEL2:.*]] = stablehlo.select %[[SEL1]], %[[ARG1]], %[[ARG3]]
// FIRST_PRECISE:   %[[SEL3:.*]] = stablehlo.select %[[SEL1]], %[[ARG2]], %[[ARG4]]
// FIRST_PRECISE:   stablehlo.return %[[SEL2]], %[[SEL3]]
// FIRST_PRECISE: }
// FIRST_PRECISE: %[[CONCAT:.*]] = stablehlo.concatenate %[[RED]]#0, %[[RED]]#1, dim = 0
// FIRST_PRECISE: %[[CONV:.*]] = stablehlo.convert %[[CONCAT]] : (tensor<2xf32>) -> tensor<2xf64>
// FIRST_PRECISE: %[[FINAL:.*]] = stablehlo.reduce(%[[CONV]] init: %{{.*}}) applies stablehlo.add across dimensions = [0]
// FIRST_PRECISE: return %[[FINAL]]

// FIRST_LIMB-LABEL: func.func @reduce_2d_test
// FIRST_LIMB: %[[INPUT:.*]] = stablehlo.concatenate
// FIRST_LIMB: %[[HI:.*]] = stablehlo.slice %[[INPUT]] [0:1, 0:2, 0:2]
// FIRST_LIMB: %[[LO:.*]] = stablehlo.slice %[[INPUT]] [1:2, 0:2, 0:2]
// FIRST_LIMB: %[[RED:.*]]:2 = stablehlo.reduce(%[[HI]] init: %{{.*}}), (%[[LO]] init: %{{.*}}) across dimensions = [1, 2]

// LAST_PRECISE-LABEL: func.func @reduce_2d_test
// LAST_PRECISE: %[[INPUT:.*]] = stablehlo.concatenate
// LAST_PRECISE: %[[HI:.*]] = stablehlo.slice %[[INPUT]] [0:2, 0:2, 0:1]
// LAST_PRECISE: %[[LO:.*]] = stablehlo.slice %[[INPUT]] [0:2, 0:2, 1:2]
// LAST_PRECISE: %[[RED:.*]]:2 = stablehlo.reduce(%[[HI]] init: %{{.*}}), (%[[LO]] init: %{{.*}}) across dimensions = [0, 1]
// LAST_PRECISE:  reducer(%[[ARG1:.*]]: tensor<f32>, %[[ARG3:.*]]: tensor<f32>) (%[[ARG2:.*]]: tensor<f32>, %[[ARG4:.*]]: tensor<f32>)  {
// LAST_PRECISE:   %[[CMP1:.*]] = stablehlo.compare GT, %[[ARG1]], %[[ARG3]]
// LAST_PRECISE:   %[[CMP2:.*]] = stablehlo.compare EQ, %[[ARG1]], %[[ARG3]]
// LAST_PRECISE:   %[[CMP3:.*]] = stablehlo.compare GT, %[[ARG2]], %[[ARG4]]
// LAST_PRECISE:   %[[SEL1:.*]] = stablehlo.select %[[CMP2]], %[[CMP3]], %[[CMP1]]
// LAST_PRECISE:   %[[SEL2:.*]] = stablehlo.select %[[SEL1]], %[[ARG1]], %[[ARG3]]
// LAST_PRECISE:   %[[SEL3:.*]] = stablehlo.select %[[SEL1]], %[[ARG2]], %[[ARG4]]
// LAST_PRECISE:   stablehlo.return %[[SEL2]], %[[SEL3]]
// LAST_PRECISE: }
// LAST_PRECISE: %[[CONCAT:.*]] = stablehlo.concatenate %[[RED]]#0, %[[RED]]#1, dim = 0
// LAST_PRECISE: %[[CONV:.*]] = stablehlo.convert %[[CONCAT]] : (tensor<2xf32>) -> tensor<2xf64>
// LAST_PRECISE: %[[FINAL:.*]] = stablehlo.reduce(%[[CONV]] init: %{{.*}}) applies stablehlo.add across dimensions = [0]
// LAST_PRECISE: return %[[FINAL]]

// TUPLE-LABEL: func.func @reduce_2d_test
// TUPLE: %[[V0:.*]] = stablehlo.convert %arg0 : (tensor<2x2xf64>) -> tensor<2x2xf32>
// TUPLE: %[[V1:.*]] = stablehlo.convert %[[V0]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
// TUPLE: %[[V2:.*]] = stablehlo.subtract %arg0, %[[V1]] : tensor<2x2xf64>
// TUPLE: %[[V3:.*]] = stablehlo.convert %[[V2]] : (tensor<2x2xf64>) -> tensor<2x2xf32>
// TUPLE: %[[V4:.*]] = stablehlo.tuple %[[V0]], %[[V3]] : tuple<tensor<2x2xf32>, tensor<2x2xf32>>
// TUPLE: %[[Vcst:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// TUPLE: %[[Vcst_0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// TUPLE: %[[V5:.*]] = stablehlo.tuple %[[Vcst]], %[[Vcst_0]] : tuple<tensor<f32>, tensor<f32>>
// TUPLE: %[[V6:.*]] = stablehlo.get_tuple_element %[[V4]][0] : (tuple<tensor<2x2xf32>, tensor<2x2xf32>>) -> tensor<2x2xf32>
// TUPLE: %[[V7:.*]] = stablehlo.get_tuple_element %[[V4]][1] : (tuple<tensor<2x2xf32>, tensor<2x2xf32>>) -> tensor<2x2xf32>
// TUPLE: %8:2 = stablehlo.reduce(%[[V6]] init: %[[Vcst]]), (%[[V7]] init: %[[Vcst_0]]) across dimensions = [0, 1] : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
// TUPLE: reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<f32>, %arg4: tensor<f32>)  {
// TUPLE: %[[V13:.*]] = stablehlo.compare GT, %arg1, %arg3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
// TUPLE: %[[V14:.*]] = stablehlo.compare EQ, %arg1, %arg3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
// TUPLE: %[[V15:.*]] = stablehlo.compare GT, %arg2, %arg4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
// TUPLE: %[[V16:.*]] = stablehlo.select %[[V14]], %[[V15]], %[[V13]] : tensor<i1>, tensor<i1>
// TUPLE: %[[V17:.*]] = stablehlo.select %[[V16]], %arg1, %arg3 : tensor<i1>, tensor<f32>
// TUPLE: %[[V18:.*]] = stablehlo.select %[[V16]], %arg2, %arg4 : tensor<i1>, tensor<f32>
// TUPLE: stablehlo.return %[[V17]], %[[V18]] : tensor<f32>, tensor<f32>
// TUPLE: }
// TUPLE: %[[V9:.*]] = stablehlo.tuple %8#0, %8#1 : tuple<tensor<f32>, tensor<f32>>
// TUPLE: %[[V10:.*]] = stablehlo.convert %8#0 : (tensor<f32>) -> tensor<f64>
// TUPLE: %[[V11:.*]] = stablehlo.convert %8#1 : (tensor<f32>) -> tensor<f64>
// TUPLE: %[[V12:.*]] = stablehlo.add %[[V10]], %[[V11]] : tensor<f64>
// TUPLE: return %[[V12]] : tensor<f64>

func.func @main() attributes {enzyme.no_multifloat} {
  %c = stablehlo.constant dense<[[1.1, 2.2], [3.3, 4.4]]> : tensor<2x2xf64>
  %expected = stablehlo.constant dense<[4.4, 6.6]> : tensor<2xf64>
  
  %res = func.call @reduce_test(%c) : (tensor<2x2xf64>) -> tensor<2xf64>
  
  "check.expect_close"(%res, %expected) {max_ulp_difference = 100 : ui64} : (tensor<2xf64>, tensor<2xf64>) -> ()

  %expected_2d = stablehlo.constant dense<4.4> : tensor<f64>
  %res_2d = func.call @reduce_2d_test(%c) : (tensor<2x2xf64>) -> tensor<f64>
  "check.expect_close"(%res_2d, %expected_2d) {max_ulp_difference = 100 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  return
}
