// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck --check-prefix=LIMB %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first precise-reduce=true" %s | FileCheck --check-prefix=PRECISE %s

func.func @reduce_test(%arg0: tensor<2x2xf64>) -> tensor<2xf64> {
  %cst = stablehlo.constant dense<0.0> : tensor<f64>
  %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
  return %0 : tensor<2xf64>
}

// LIMB-LABEL: func.func @reduce_test
// LIMB: %[[INPUT_HI:.*]] = stablehlo.slice
// LIMB: %[[INPUT_LO:.*]] = stablehlo.slice
// LIMB: %[[REDUCE_HI:.*]] = stablehlo.reduce(%[[INPUT_HI]]
// LIMB: %[[REDUCE_LO:.*]] = stablehlo.reduce(%[[INPUT_LO]]
// LIMB: %[[PACKED:.*]] = stablehlo.concatenate %[[REDUCE_HI]], %[[REDUCE_LO]]
// LIMB: return %[[PACKED]]

// PRECISE-LABEL: func.func @reduce_test
// PRECISE: %[[INPUT_HI:.*]] = stablehlo.slice
// PRECISE: %[[INPUT_LO:.*]] = stablehlo.slice
// PRECISE: %[[REDUCE:.*]]:2 = stablehlo.reduce(%[[INPUT_HI]], %[[INPUT_LO]]
// PRECISE: %[[PACKED:.*]] = stablehlo.concatenate %[[REDUCE]]#0, %[[REDUCE]]#1
// PRECISE: return %[[PACKED]]
