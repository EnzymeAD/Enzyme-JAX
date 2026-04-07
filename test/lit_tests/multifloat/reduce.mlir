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
// FIRST_LIMB: %[[HI:.*]] = stablehlo.slice %[[INPUT]] [0:1, 0:2, 0:2]
// FIRST_LIMB: %[[LO:.*]] = stablehlo.slice %[[INPUT]] [1:2, 0:2, 0:2]
// FIRST_LIMB: %[[RED_HI:.*]] = stablehlo.reduce(%[[HI]] init: %{{.*}}) applies stablehlo.add across dimensions = [0, 1]
// FIRST_LIMB: %[[RED_LO:.*]] = stablehlo.reduce(%[[LO]] init: %{{.*}}) applies stablehlo.add across dimensions = [0, 1]

// FIRST_PRECISE-LABEL: func.func @reduce_test
// FIRST_PRECISE: %[[INPUT:.*]] = stablehlo.concatenate
// FIRST_PRECISE: %[[HI:.*]] = stablehlo.slice %[[INPUT]] [0:1, 0:2, 0:2]
// FIRST_PRECISE: %[[LO:.*]] = stablehlo.slice %[[INPUT]] [1:2, 0:2, 0:2]
// FIRST_PRECISE: stablehlo.reduce(%[[HI]] init: %{{.*}}), (%[[LO]] init: %{{.*}}) across dimensions = [1]

// LAST_LIMB-LABEL: func.func @reduce_test
// LAST_LIMB: %[[INPUT:.*]] = stablehlo.concatenate
// LAST_LIMB: %[[HI:.*]] = stablehlo.slice %[[INPUT]] [0:2, 0:2, 0:1]
// LAST_LIMB: %[[LO:.*]] = stablehlo.slice %[[INPUT]] [0:2, 0:2, 1:2]
// LAST_LIMB: stablehlo.reduce(%[[HI]] init: %{{.*}}) across dimensions = [0]
// LAST_LIMB: stablehlo.reduce(%[[LO]] init: %{{.*}}) across dimensions = [0]

// LAST_PRECISE-LABEL: func.func @reduce_test
// LAST_PRECISE: %[[INPUT:.*]] = stablehlo.concatenate
// LAST_PRECISE: %[[HI:.*]] = stablehlo.slice %[[INPUT]] [0:2, 0:2, 0:1]
// LAST_PRECISE: %[[LO:.*]] = stablehlo.slice %[[INPUT]] [0:2, 0:2, 1:2]
// LAST_PRECISE: stablehlo.reduce(%[[HI]] init: %{{.*}}), (%[[LO]] init: %{{.*}}) across dimensions = [0]

func.func @main() attributes {enzyme.no_multifloat} {
  %c = stablehlo.constant dense<[[1.1, 2.2], [3.3, 4.4]]> : tensor<2x2xf64>
  %expected = stablehlo.constant dense<[4.4, 6.6]> : tensor<2xf64>
  
  %res = func.call @reduce_test(%c) : (tensor<2x2xf64>) -> tensor<2xf64>
  
  "check.expect_close"(%res, %expected) {max_ulp_difference = 100 : ui64} : (tensor<2xf64>, tensor<2xf64>) -> ()
  return
}
