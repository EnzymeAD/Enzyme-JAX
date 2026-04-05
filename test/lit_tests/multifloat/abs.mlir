// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck %s

func.func @abs(%arg0: tensor<2xf64>) -> tensor<2xf64> {
  %0 = stablehlo.abs %arg0 : tensor<2xf64>
  return %0 : tensor<2xf64>
}
// CHECK: func @abs(%arg0: tensor<2xf64>) -> tensor<2xf64> {
// CHECK: %[[C0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// CHECK: %[[C1:.*]] = stablehlo.convert %[[C0]] : (tensor<2xf32>) -> tensor<2xf64>
// CHECK: %[[SUB:.*]] = stablehlo.subtract %arg0, %[[C1]] : tensor<2xf64>
// CHECK: %[[C2:.*]] = stablehlo.convert %[[SUB]] : (tensor<2xf64>) -> tensor<2xf32>
// CHECK: %[[R1:.*]] = stablehlo.reshape %[[C0]] : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK: %[[R2:.*]] = stablehlo.reshape %[[C2]] : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK: %[[CAT:.*]] = stablehlo.concatenate %[[R1]], %[[R2]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK: %[[X_HI:.*]] = stablehlo.slice %[[CAT]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// CHECK: %[[X_LO:.*]] = stablehlo.slice %[[CAT]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// CHECK: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<1x2xf32>
// CHECK: %[[CMP:.*]] = stablehlo.compare GE, %[[X_HI]], %[[CST]] : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x2xi1>
// CHECK: %[[NEG_HI:.*]] = stablehlo.negate %[[X_HI]] : tensor<1x2xf32>
// CHECK: %[[NEG_LO:.*]] = stablehlo.negate %[[X_LO]] : tensor<1x2xf32>
// CHECK: %[[SEL_HI:.*]] = stablehlo.select %[[CMP]], %[[X_HI]], %[[NEG_HI]] : tensor<1x2xi1>, tensor<1x2xf32>
// CHECK: %[[SEL_LO:.*]] = stablehlo.select %[[CMP]], %[[X_LO]], %[[NEG_LO]] : tensor<1x2xi1>, tensor<1x2xf32>
// CHECK: %{{.*}} = stablehlo.concatenate %[[SEL_HI]], %[[SEL_LO]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK: %{{.*}} = stablehlo.convert %{{.*}} : (tensor<2x2xf32>) -> tensor<2x2xf64>
// CHECK: %[[CST_0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK: %{{.*}} = stablehlo.reduce(%{{.*}} init: %[[CST_0]]) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
// CHECK: return %{{.*}} : tensor<2xf64>
