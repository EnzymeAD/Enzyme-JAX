// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=reduce_minor_dims;transpose_elementwise(1);transpose_reshape;transpose_broadcast_in_dim_to_broadcast_in_dim" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// A batched contraction as the raising emits it: the contracted dimension
// leads, the input is a product of broadcasts. Reduced as the trailing
// dimension instead, with the transpose this takes absorbed into the
// broadcasts; the argument x, read through its broadcast, runs along the
// contracted dimension.
func.func @row(%x: tensor<512x8x4x3xf64>, %B: tensor<3x4xf64>) -> tensor<512x8x16xf64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = stablehlo.broadcast_in_dim %x, dims = [1, 2, 3, 0] : (tensor<512x8x4x3xf64>) -> tensor<3x512x8x4x4xf64>
  %1 = stablehlo.broadcast_in_dim %B, dims = [0, 4] : (tensor<3x4xf64>) -> tensor<3x512x8x4x4xf64>
  %2 = stablehlo.multiply %0, %1 : tensor<3x512x8x4x4xf64>
  %3 = stablehlo.reshape %2 : (tensor<3x512x8x4x4xf64>) -> tensor<3x512x8x16xf64>
  %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3x512x8x16xf64>, tensor<f64>) -> tensor<512x8x16xf64>
  return %4 : tensor<512x8x16xf64>
}

// CHECK-LABEL: func.func @row(
// CHECK-SAME: %[[X:[^ ,:)]+]]: tensor<512x8x4x3xf64>, %[[B:[^ ,:)]+]]: tensor<3x4xf64>) -> tensor<512x8x16xf64> {
// CHECK-NEXT: %[[Z:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[X0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[X]], dims = [0, 1, 2, 4] : (tensor<512x8x4x3xf64>) -> tensor<512x8x4x4x3xf64>
// CHECK-NEXT: %[[B0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[B]], dims = [4, 3] : (tensor<3x4xf64>) -> tensor<512x8x4x4x3xf64>
// CHECK-NEXT: %[[M:[^ ,:)]+]] = stablehlo.multiply %[[X0]], %[[B0]] : tensor<512x8x4x4x3xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.reshape %[[M]] : (tensor<512x8x4x4x3xf64>) -> tensor<512x8x16x3xf64>
// CHECK-NEXT: %[[S:[^ ,:)]+]] = stablehlo.reduce(%[[R]] init: %[[Z]]) applies stablehlo.add across dimensions = [3] : (tensor<512x8x16x3xf64>, tensor<f64>) -> tensor<512x8x16xf64>
// CHECK-NEXT: return %[[S]] : tensor<512x8x16xf64>
// CHECK-NEXT: }

// The argument runs along a kept dimension (its minor dimension is the
// lanes): a row reduction would read it strided; left as it is.
func.func @kept(%x: tensor<512x8x3x4xf64>, %B: tensor<3x4xf64>) -> tensor<512x8x4x4xf64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = stablehlo.broadcast_in_dim %x, dims = [1, 2, 0, 3] : (tensor<512x8x3x4xf64>) -> tensor<3x512x8x4x4xf64>
  %1 = stablehlo.broadcast_in_dim %B, dims = [0, 4] : (tensor<3x4xf64>) -> tensor<3x512x8x4x4xf64>
  %2 = stablehlo.multiply %0, %1 : tensor<3x512x8x4x4xf64>
  %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3x512x8x4x4xf64>, tensor<f64>) -> tensor<512x8x4x4xf64>
  return %3 : tensor<512x8x4x4xf64>
}

// CHECK-LABEL: func.func @kept(
// CHECK: stablehlo.reduce(%{{.*}}) applies stablehlo.add across dimensions = [0] : (tensor<3x512x8x4x4xf64>, tensor<f64>) -> tensor<512x8x4x4xf64>

// A row of 6: left as it is.
func.func @long(%x: tensor<512x8x4x6xf64>, %B: tensor<6x4xf64>) -> tensor<512x8x4x4xf64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = stablehlo.broadcast_in_dim %x, dims = [1, 2, 3, 0] : (tensor<512x8x4x6xf64>) -> tensor<6x512x8x4x4xf64>
  %1 = stablehlo.broadcast_in_dim %B, dims = [0, 4] : (tensor<6x4xf64>) -> tensor<6x512x8x4x4xf64>
  %2 = stablehlo.multiply %0, %1 : tensor<6x512x8x4x4xf64>
  %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<6x512x8x4x4xf64>, tensor<f64>) -> tensor<512x8x4x4xf64>
  return %3 : tensor<512x8x4x4xf64>
}

// CHECK-LABEL: func.func @long(
// CHECK: stablehlo.reduce(%{{.*}}) applies stablehlo.add across dimensions = [0] : (tensor<6x512x8x4x4xf64>, tensor<f64>) -> tensor<512x8x4x4xf64>
