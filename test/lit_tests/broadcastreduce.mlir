// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

// CHECK-LABEL: @one_add
// CHECK-SAME: (%[[ARG0:.+]]: tensor<1x3072xf32>, %[[ARG1:.+]]: tensor<f32>)
// CHECK: %[[V0:.+]] = stablehlo.constant dense<3.200000e+01> : tensor<f32>
// CHECK: %[[V1:.+]] = stablehlo.reduce(%[[ARG0]] init: %[[ARG1]]) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x3072xf32>, tensor<f32>) -> tensor<f32>
// CHECK: %[[V2:.+]] = stablehlo.multiply %[[V1]], %[[V0]] : tensor<f32>
func.func @one_add(%154: tensor<1x3072xf32>, %151: tensor<f32>) -> tensor<f32> {
  %211 = stablehlo.broadcast_in_dim %154, dims = [0, 1] : (tensor<1x3072xf32>) -> tensor<1x3072x32xf32>
  %212 = stablehlo.reduce(%211 init: %151) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x3072x32xf32>, tensor<f32>) -> tensor<f32>
  return %212 : tensor<f32>
}

// CHECK-LABEL: @one_mul
// CHECK-SAME: (%[[ARG0:.+]]: tensor<1x3072xf32>, %[[ARG1:.+]]: tensor<f32>)
// CHECK: %[[V0:.+]] = stablehlo.constant dense<3.200000e+01> : tensor<f32>
// CHECK: %[[V1:.+]] = stablehlo.reduce(%[[ARG0]] init: %[[ARG1]]) applies stablehlo.multiply across dimensions = [0, 1] : (tensor<1x3072xf32>, tensor<f32>) -> tensor<f32>
// CHECK: %[[V2:.+]] = stablehlo.power %[[V1]], %[[V0]] : tensor<f32>
func.func @one_mul(%154: tensor<1x3072xf32>, %151: tensor<f32>) -> tensor<f32> {
  %211 = stablehlo.broadcast_in_dim %154, dims = [0, 1] : (tensor<1x3072xf32>) -> tensor<1x3072x32xf32>
  %212 = stablehlo.reduce(%211 init: %151) applies stablehlo.multiply across dimensions = [0, 1, 2] : (tensor<1x3072x32xf32>, tensor<f32>) -> tensor<f32>
  return %212 : tensor<f32>
}

// CHECK-LABEL: @one_min
// CHECK-SAME: (%[[ARG0:.+]]: tensor<1x3072xf32>, %[[ARG1:.+]]: tensor<f32>)
// CHECK: %[[V0:.+]] = stablehlo.reduce(%[[ARG0]] init: %[[ARG1]]) applies stablehlo.minimum across dimensions = [0, 1] : (tensor<1x3072xf32>, tensor<f32>) -> tensor<f32>
func.func @one_min(%154: tensor<1x3072xf32>, %151: tensor<f32>) -> tensor<f32> {
  %211 = stablehlo.broadcast_in_dim %154, dims = [0, 1] : (tensor<1x3072xf32>) -> tensor<1x3072x32xf32>
  %212 = stablehlo.reduce(%211 init: %151) applies stablehlo.minimum across dimensions = [0, 1, 2] : (tensor<1x3072x32xf32>, tensor<f32>) -> tensor<f32>
  return %212 : tensor<f32>
}

// CHECK-LABEL: @one_max
// CHECK-SAME: (%[[ARG0:.+]]: tensor<1x3072xf32>, %[[ARG1:.+]]: tensor<f32>)
// CHECK: %[[V0:.+]] = stablehlo.reduce(%[[ARG0]] init: %[[ARG1]]) applies stablehlo.maximum across dimensions = [0, 1] : (tensor<1x3072xf32>, tensor<f32>) -> tensor<f32>
func.func @one_max(%154: tensor<1x3072xf32>, %151: tensor<f32>) -> tensor<f32> {
  %211 = stablehlo.broadcast_in_dim %154, dims = [0, 1] : (tensor<1x3072xf32>) -> tensor<1x3072x32xf32>
  %212 = stablehlo.reduce(%211 init: %151) applies stablehlo.maximum across dimensions = [0, 1, 2] : (tensor<1x3072x32xf32>, tensor<f32>) -> tensor<f32>
  return %212 : tensor<f32>
}

// CHECK-LABEL: @two_add
// CHECK-SAME: (%[[ARG0:.+]]: tensor<1x3072xf32>, %[[ARG1:.+]]: tensor<f32>)
// CHECK: %[[V0:.+]] = stablehlo.constant dense<3.200000e+01> : tensor<f32>
// CHECK: %[[V1:.+]] = stablehlo.reduce(%[[ARG0]] init: %[[ARG1]]) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x3072xf32>, tensor<f32>) -> tensor<f32>
// CHECK: %[[V2:.+]] = stablehlo.multiply %[[V1]], %[[V0]] : tensor<f32>
func.func @two_add(%154: tensor<1x3072xf32>, %151: tensor<f32>) -> tensor<f32> {
  %211 = stablehlo.broadcast_in_dim %154, dims = [0, 1] : (tensor<1x3072xf32>) -> tensor<32x3072xf32>
  %212 = stablehlo.reduce(%211 init: %151) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3072xf32>, tensor<f32>) -> tensor<f32>
  return %212 : tensor<f32>
}

// CHECK-LABEL: @two_mul
// CHECK-SAME: (%[[ARG0:.+]]: tensor<1x3072xf32>, %[[ARG1:.+]]: tensor<f32>)
// CHECK: %[[V0:.+]] = stablehlo.constant dense<3.200000e+01> : tensor<f32>
// CHECK: %[[V1:.+]] = stablehlo.reduce(%[[ARG0]] init: %[[ARG1]]) applies stablehlo.multiply across dimensions = [0, 1] : (tensor<1x3072xf32>, tensor<f32>) -> tensor<f32>
// CHECK: %[[V2:.+]] = stablehlo.power %[[V1]], %[[V0]] : tensor<f32>
func.func @two_mul(%154: tensor<1x3072xf32>, %151: tensor<f32>) -> tensor<f32> {
  %211 = stablehlo.broadcast_in_dim %154, dims = [0, 1] : (tensor<1x3072xf32>) -> tensor<32x3072xf32>
  %212 = stablehlo.reduce(%211 init: %151) applies stablehlo.multiply across dimensions = [0, 1] : (tensor<32x3072xf32>, tensor<f32>) -> tensor<f32>
  return %212 : tensor<f32>
}

// CHECK-LABEL: @three_add
// CHECK-SAME: (%[[ARG0:.+]]: tensor<1x3072xf32>, %[[ARG1:.+]]: tensor<f32>)
// CHECK: %[[V0:.+]] = stablehlo.constant dense<3.200000e+02> : tensor<f32>
// CHECK: %[[V1:.+]] = stablehlo.reduce(%[[ARG0]] init: %[[ARG1]]) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x3072xf32>, tensor<f32>) -> tensor<f32>
// CHECK: %[[V2:.+]] = stablehlo.multiply %[[V1]], %[[V0]] : tensor<f32>
func.func @three_add(%154: tensor<1x3072xf32>, %151: tensor<f32>) -> tensor<f32> {
  %211 = stablehlo.broadcast_in_dim %154, dims = [2, 1] : (tensor<1x3072xf32>) -> tensor<32x3072x10xf32>
  %212 = stablehlo.reduce(%211 init: %151) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<32x3072x10xf32>, tensor<f32>) -> tensor<f32>
  return %212 : tensor<f32>
}

// CHECK-LABEL: @three_mul
// CHECK-SAME: (%[[ARG0:.+]]: tensor<1x3072xf32>, %[[ARG1:.+]]: tensor<f32>)
// CHECK: %[[V0:.+]] = stablehlo.constant dense<3.200000e+02> : tensor<f32>
// CHECK: %[[V1:.+]] = stablehlo.reduce(%[[ARG0]] init: %[[ARG1]]) applies stablehlo.multiply across dimensions = [0, 1] : (tensor<1x3072xf32>, tensor<f32>) -> tensor<f32>
// CHECK: %[[V2:.+]] = stablehlo.power %[[V1]], %[[V0]] : tensor<f32>
func.func @three_mul(%154: tensor<1x3072xf32>, %151: tensor<f32>) -> tensor<f32> {
  %211 = stablehlo.broadcast_in_dim %154, dims = [2, 1] : (tensor<1x3072xf32>) -> tensor<32x3072x10xf32>
  %212 = stablehlo.reduce(%211 init: %151) applies stablehlo.multiply across dimensions = [0, 1, 2] : (tensor<32x3072x10xf32>, tensor<f32>) -> tensor<f32>
  return %212 : tensor<f32>
}

// CHECK-LABEL: @four_add
// CHECK-SAME: (%[[ARG0:.+]]: tensor<1x3072xf32>, %[[ARG1:.+]]: tensor<f32>)
func.func @four_add(%154: tensor<1x3072xf32>, %151: tensor<f32>) -> tensor<f32> {
// CHECK: %[[V0:.+]] = stablehlo.constant dense<3.200000e+02> : tensor<f32>
// CHECK: %[[V1:.+]] = stablehlo.reduce(%[[ARG0]] init: %[[ARG1]]) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x3072xf32>, tensor<f32>) -> tensor<f32>
// CHECK: %[[V2:.+]] = stablehlo.multiply %[[V1]], %[[V0]] : tensor<f32>
  %211 = stablehlo.broadcast_in_dim %154, dims = [0, 1] : (tensor<1x3072xf32>) -> tensor<32x3072x10xf32>
  %212 = stablehlo.reduce(%211 init: %151) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<32x3072x10xf32>, tensor<f32>) -> tensor<f32>
  return %212 : tensor<f32>
}

// CHECK-LABEL: @four_mul
// CHECK-SAME: (%[[ARG0:.+]]: tensor<1x3072xf32>, %[[ARG1:.+]]: tensor<f32>)
// CHECK: %[[V0:.+]] = stablehlo.constant dense<3.200000e+02> : tensor<f32>
// CHECK: %[[V1:.+]] = stablehlo.reduce(%[[ARG0]] init: %[[ARG1]]) applies stablehlo.multiply across dimensions = [0, 1] : (tensor<1x3072xf32>, tensor<f32>) -> tensor<f32>
// CHECK: %[[V2:.+]] = stablehlo.power %[[V1]], %[[V0]] : tensor<f32>
func.func @four_mul(%154: tensor<1x3072xf32>, %151: tensor<f32>) -> tensor<f32> {
  %211 = stablehlo.broadcast_in_dim %154, dims = [0, 1] : (tensor<1x3072xf32>) -> tensor<32x3072x10xf32>
  %212 = stablehlo.reduce(%211 init: %151) applies stablehlo.multiply across dimensions = [0, 1, 2] : (tensor<32x3072x10xf32>, tensor<f32>) -> tensor<f32>
  return %212 : tensor<f32>
}
