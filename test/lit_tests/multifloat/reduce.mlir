// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck --check-prefix=FIRST_LIMB %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first precise-reduce=true" %s | FileCheck --check-prefix=FIRST_PRECISE %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=LAST_LIMB %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last precise-reduce=true" %s | FileCheck --check-prefix=LAST_PRECISE %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first precise-reduce=true" %s | stablehlo-translate - --interpret --allow-unregistered-dialect

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
