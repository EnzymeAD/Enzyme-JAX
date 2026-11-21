// RUN: enzymexlamlir-opt --split-input-file %s | enzymexlamlir-opt --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @quantize(
func.func @quantize(%x: tensor<4xf32>, %s: tensor<f32>, %zp: tensor<f32>, %y: tensor<2x4xf32>, %sa: tensor<4xf32>, %zpa: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.quantize %{{.+}}, %{{.+}}, %{{.+}} : (tensor<4xf32>, tensor<f32>, tensor<f32>) -> tensor<4xi8>
  %0 = mps.quantize %x, %s, %zp : (tensor<4xf32>, tensor<f32>, tensor<f32>) -> tensor<4xi8>
  // CHECK: %{{.+}} = mps.quantize %{{.+}}, %{{.+}}, %{{.+}} <axis = 1> : (tensor<2x4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<2x4xui8>
  %1 = mps.quantize %y, %sa, %zpa <axis = 1> : (tensor<2x4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<2x4xui8>
  return
}

// -----

// CHECK-LABEL: func.func @dequantize(
func.func @dequantize(%q: tensor<4xi8>, %s: tensor<f32>, %zp: tensor<f32>, %qa: tensor<2x4xui8>, %sa: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.dequantize %{{.+}}, %{{.+}}, %{{.+}} : (tensor<4xi8>, tensor<f32>, tensor<f32>) -> tensor<4xf32>
  %0 = mps.dequantize %q, %s, %zp : (tensor<4xi8>, tensor<f32>, tensor<f32>) -> tensor<4xf32>
  // CHECK: %{{.+}} = mps.dequantize %{{.+}}, %{{.+}} : (tensor<4xi8>, tensor<f32>) -> tensor<4xf32>
  %1 = mps.dequantize %q, %s : (tensor<4xi8>, tensor<f32>) -> tensor<4xf32>
  // CHECK: %{{.+}} = mps.dequantize %{{.+}}, %{{.+}} <axis = 1> : (tensor<2x4xui8>, tensor<4xf32>) -> tensor<2x4xf16>
  %2 = mps.dequantize %qa, %sa <axis = 1> : (tensor<2x4xui8>, tensor<4xf32>) -> tensor<2x4xf16>
  return
}

// -----

// CHECK-LABEL: func.func @dequantize_lut(
func.func @dequantize_lut(%q4: tensor<2x4xui4>, %lut: tensor<16xf32>, %luta: tensor<4x16xf32>) {
  // CHECK: %{{.+}} = mps.dequantize_lut %{{.+}}, %{{.+}} : (tensor<2x4xui4>, tensor<16xf32>) -> tensor<2x4xf32>
  %0 = mps.dequantize_lut %q4, %lut : (tensor<2x4xui4>, tensor<16xf32>) -> tensor<2x4xf32>
  // CHECK: %{{.+}} = mps.dequantize_lut %{{.+}}, %{{.+}} <axis = 1> : (tensor<2x4xui4>, tensor<4x16xf32>) -> tensor<2x4xf32>
  %1 = mps.dequantize_lut %q4, %luta <axis = 1> : (tensor<2x4xui4>, tensor<4x16xf32>) -> tensor<2x4xf32>
  return
}
