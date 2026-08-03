// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=reduce_mul_broadcast},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

// The broadcast does not vary along the reduced dimension (1), so it can be
// pulled out of the reduction and applied at the reduced shape.
func.func @pull_out_rhs(%v: tensor<256xf32>, %a: tensor<256x1024xf32>) -> tensor<256xf32> {
  %init = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %b = stablehlo.broadcast_in_dim %v, dims = [0] : (tensor<256xf32>) -> tensor<256x1024xf32>
  %0 = stablehlo.multiply %b, %a : tensor<256x1024xf32>
  %1 = stablehlo.reduce(%0 init: %init) applies stablehlo.add across dimensions = [1] : (tensor<256x1024xf32>, tensor<f32>) -> tensor<256xf32>
  return %1 : tensor<256xf32>
}

// CHECK-LABEL: func.func @pull_out_rhs
// CHECK-SAME: (%[[V:.+]]: tensor<256xf32>, %[[A:.+]]: tensor<256x1024xf32>)
// CHECK-NOT: stablehlo.multiply %{{.+}}, %{{.+}} : tensor<256x1024xf32>
// CHECK: %[[R:.+]] = stablehlo.reduce(%[[A]] init: %{{.+}}) applies stablehlo.add across dimensions = [1] : (tensor<256x1024xf32>, tensor<f32>) -> tensor<256xf32>
// CHECK: stablehlo.multiply %{{.+}}, %[[R]] : tensor<256xf32>

// The broadcast may sit on either side of the multiply.
func.func @pull_out_lhs(%v: tensor<256xf32>, %a: tensor<256x1024xf32>) -> tensor<256xf32> {
  %init = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %b = stablehlo.broadcast_in_dim %v, dims = [0] : (tensor<256xf32>) -> tensor<256x1024xf32>
  %0 = stablehlo.multiply %a, %b : tensor<256x1024xf32>
  %1 = stablehlo.reduce(%0 init: %init) applies stablehlo.add across dimensions = [1] : (tensor<256x1024xf32>, tensor<f32>) -> tensor<256xf32>
  return %1 : tensor<256xf32>
}

// CHECK-LABEL: func.func @pull_out_lhs
// CHECK-NOT: stablehlo.multiply %{{.+}}, %{{.+}} : tensor<256x1024xf32>
// CHECK: stablehlo.multiply %{{.+}}, %{{.+}} : tensor<256xf32>

// Reducing dimension 1 of a rank-3 value: the surviving broadcast dimension 2
// shifts down to 1.
func.func @shift_dims(%v: tensor<4x6xf32>, %a: tensor<4x8x6xf32>) -> tensor<4x6xf32> {
  %init = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %b = stablehlo.broadcast_in_dim %v, dims = [0, 2] : (tensor<4x6xf32>) -> tensor<4x8x6xf32>
  %0 = stablehlo.multiply %b, %a : tensor<4x8x6xf32>
  %1 = stablehlo.reduce(%0 init: %init) applies stablehlo.add across dimensions = [1] : (tensor<4x8x6xf32>, tensor<f32>) -> tensor<4x6xf32>
  return %1 : tensor<4x6xf32>
}

// CHECK-LABEL: func.func @shift_dims
// CHECK-SAME: (%[[V:.+]]: tensor<4x6xf32>, %[[A:.+]]: tensor<4x8x6xf32>)
// CHECK-NOT: stablehlo.multiply %{{.+}}, %{{.+}} : tensor<4x8x6xf32>
// CHECK: %[[R:.+]] = stablehlo.reduce(%[[A]] init: %{{.+}}) applies stablehlo.add across dimensions = [1] : (tensor<4x8x6xf32>, tensor<f32>) -> tensor<4x6xf32>
// CHECK: %[[B:.+]] = stablehlo.broadcast_in_dim %[[V]], dims = [0, 1] : (tensor<4x6xf32>) -> tensor<4x6xf32>
// CHECK: stablehlo.multiply %[[B]], %[[R]] : tensor<4x6xf32>

// The broadcast varies along the reduced dimension, so it cannot be pulled out.
func.func @no_pull_varying(%v: tensor<1024xf32>, %a: tensor<256x1024xf32>) -> tensor<256xf32> {
  %init = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %b = stablehlo.broadcast_in_dim %v, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024xf32>
  %0 = stablehlo.multiply %b, %a : tensor<256x1024xf32>
  %1 = stablehlo.reduce(%0 init: %init) applies stablehlo.add across dimensions = [1] : (tensor<256x1024xf32>, tensor<f32>) -> tensor<256xf32>
  return %1 : tensor<256xf32>
}

// CHECK-LABEL: func.func @no_pull_varying
// CHECK: %[[M:.+]] = stablehlo.multiply %{{.+}}, %{{.+}} : tensor<256x1024xf32>
// CHECK: stablehlo.reduce(%[[M]]

// A multiply feeding more than the reduction must not be duplicated.
func.func @no_pull_multi_use(%v: tensor<256xf32>, %a: tensor<256x1024xf32>) -> (tensor<256xf32>, tensor<256x1024xf32>) {
  %init = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %b = stablehlo.broadcast_in_dim %v, dims = [0] : (tensor<256xf32>) -> tensor<256x1024xf32>
  %0 = stablehlo.multiply %b, %a : tensor<256x1024xf32>
  %1 = stablehlo.reduce(%0 init: %init) applies stablehlo.add across dimensions = [1] : (tensor<256x1024xf32>, tensor<f32>) -> tensor<256xf32>
  return %1, %0 : tensor<256xf32>, tensor<256x1024xf32>
}

// CHECK-LABEL: func.func @no_pull_multi_use
// CHECK: %[[M:.+]] = stablehlo.multiply %{{.+}}, %{{.+}} : tensor<256x1024xf32>
// CHECK: stablehlo.reduce(%[[M]]

// Only sum reductions commute with scaling; max does not.
func.func @no_pull_max(%v: tensor<256xf32>, %a: tensor<256x1024xf32>) -> tensor<256xf32> {
  %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %b = stablehlo.broadcast_in_dim %v, dims = [0] : (tensor<256xf32>) -> tensor<256x1024xf32>
  %0 = stablehlo.multiply %b, %a : tensor<256x1024xf32>
  %1 = stablehlo.reduce(%0 init: %init) applies stablehlo.maximum across dimensions = [1] : (tensor<256x1024xf32>, tensor<f32>) -> tensor<256xf32>
  return %1 : tensor<256xf32>
}

// CHECK-LABEL: func.func @no_pull_max
// CHECK: %[[M:.+]] = stablehlo.multiply %{{.+}}, %{{.+}} : tensor<256x1024xf32>
// CHECK: stablehlo.reduce(%[[M]]
