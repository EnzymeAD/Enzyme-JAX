// RUN: enzymexlamlir-opt --split-input-file %s | enzymexlamlir-opt --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @matrix_multiplication(
func.func @matrix_multiplication(%a: tensor<2x3x4xf32>, %b: tensor<4x5xf32>) {
  // CHECK: %{{.+}} = mps.matrix_multiplication %{{.+}}, %{{.+}} : (tensor<2x3x4xf32>, tensor<4x5xf32>) -> tensor<2x3x5xf32>
  %0 = mps.matrix_multiplication %a, %b : (tensor<2x3x4xf32>, tensor<4x5xf32>) -> tensor<2x3x5xf32>
  return
}

// -----

// CHECK-LABEL: func.func @band_part(
func.func @band_part(%s: tensor<3x3xf32>) {
  // CHECK: %{{.+}} = mps.band_part %{{.+}} <num_lower = -1, num_upper = 0> : (tensor<3x3xf32>) -> tensor<3x3xf32>
  %0 = mps.band_part %s <num_lower = -1, num_upper = 0> : (tensor<3x3xf32>) -> tensor<3x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @inverse(
func.func @inverse(%s: tensor<3x3xf32>) {
  // CHECK: %{{.+}} = mps.inverse %{{.+}} : (tensor<3x3xf32>) -> tensor<3x3xf32>
  %0 = mps.inverse %s : (tensor<3x3xf32>) -> tensor<3x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @hamming_distance(
func.func @hamming_distance(%u: tensor<4x2xui32>) {
  // CHECK: %{{.+}} = mps.hamming_distance %{{.+}}, %{{.+}} : (tensor<4x2xui32>, tensor<4x2xui32>) -> tensor<4xui16>
  %0 = mps.hamming_distance %u, %u : (tensor<4x2xui32>, tensor<4x2xui32>) -> tensor<4xui16>
  return
}

// -----

// CHECK-LABEL: func.func @scaled_dot_product_attention(
func.func @scaled_dot_product_attention(%q: tensor<1x2x4x8xf32>, %k: tensor<1x2x4x8xf32>, %v: tensor<1x2x4x8xf32>, %mask: tensor<1x1x4x4xf32>, %sinks: tensor<2xf32>) {
  // CHECK: %{{.+}} = "mps.scaled_dot_product_attention"(%{{.+}}, %{{.+}}, %{{.+}}) <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>, scale = 5.000000e-01 : f32}> : (tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>) -> tensor<1x2x4x8xf32>
  %0 = "mps.scaled_dot_product_attention"(%q, %k, %v) <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>, scale = 5.000000e-01 : f32}> : (tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>) -> tensor<1x2x4x8xf32>
  // CHECK: %{{.+}} = "mps.scaled_dot_product_attention"(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>, scale = 5.000000e-01 : f32}> : (tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>, tensor<1x1x4x4xf32>) -> tensor<1x2x4x8xf32>
  %1 = "mps.scaled_dot_product_attention"(%q, %k, %v, %mask) <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>, scale = 5.000000e-01 : f32}> : (tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>, tensor<1x1x4x4xf32>) -> tensor<1x2x4x8xf32>
  // CHECK: %{{.+}} = "mps.scaled_dot_product_attention"(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) <{is_causal = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>, scale = 5.000000e-01 : f32}> : (tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>, tensor<1x1x4x4xf32>, tensor<2xf32>) -> tensor<1x2x4x8xf32>
  %2 = "mps.scaled_dot_product_attention"(%q, %k, %v, %mask, %sinks) <{is_causal = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>, scale = 5.000000e-01 : f32}> : (tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>, tensor<1x1x4x4xf32>, tensor<2xf32>) -> tensor<1x2x4x8xf32>
  return
}
