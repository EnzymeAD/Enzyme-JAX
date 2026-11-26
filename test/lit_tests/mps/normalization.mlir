// RUN: enzymexlamlir-opt --split-input-file %s | enzymexlamlir-opt --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @mean(
func.func @mean(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.mean %{{.+}} <axes = [0]> : (tensor<2x3xf32>) -> tensor<1x3xf32>
  %0 = mps.mean %x <axes = [0]> : (tensor<2x3xf32>) -> tensor<1x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @variance(
func.func @variance(%x: tensor<2x3xf32>, %m: tensor<1x3xf32>) {
  // CHECK: %{{.+}} = mps.variance %{{.+}} <axes = [0]> : (tensor<2x3xf32>) -> tensor<1x3xf32>
  %0 = mps.variance %x <axes = [0]> : (tensor<2x3xf32>) -> tensor<1x3xf32>
  // CHECK: %{{.+}} = mps.variance %{{.+}}, %{{.+}} <axes = [0]> : (tensor<2x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  %1 = mps.variance %x, %m <axes = [0]> : (tensor<2x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @normalize(
func.func @normalize(%x: tensor<2x3xf32>, %m: tensor<1x3xf32>, %v: tensor<1x3xf32>, %gamma: tensor<1x3xf32>, %beta: tensor<1x3xf32>) {
  // CHECK: %{{.+}} = "mps.normalize"(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) <{epsilon = 5.000000e-01 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>}> : (tensor<2x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<2x3xf32>
  %0 = "mps.normalize"(%x, %m, %v, %gamma, %beta) <{epsilon = 5.000000e-01 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>}> : (tensor<2x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<2x3xf32>
  // CHECK: %{{.+}} = "mps.normalize"(%{{.+}}, %{{.+}}, %{{.+}}) <{epsilon = 5.000000e-01 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<2x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<2x3xf32>
  %1 = "mps.normalize"(%x, %m, %v) <{epsilon = 5.000000e-01 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<2x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @normalization_gamma_gradient(
func.func @normalization_gamma_gradient(%g: tensor<2x3xf32>, %x: tensor<2x3xf32>, %m: tensor<1x3xf32>, %v: tensor<1x3xf32>) {
  // CHECK: %{{.+}} = mps.normalization_gamma_gradient %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} <reduction_axes = [0], epsilon = 5.000000e-01> : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  %0 = mps.normalization_gamma_gradient %g, %x, %m, %v <reduction_axes = [0], epsilon = 5.000000e-01> : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @normalization_beta_gradient(
func.func @normalization_beta_gradient(%g: tensor<2x3xf32>, %x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.normalization_beta_gradient %{{.+}}, %{{.+}} <reduction_axes = [0]> : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
  %0 = mps.normalization_beta_gradient %g, %x <reduction_axes = [0]> : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @normalization_gradient(
func.func @normalization_gradient(%g: tensor<2x3xf32>, %x: tensor<2x3xf32>, %m: tensor<1x3xf32>, %v: tensor<1x3xf32>, %gamma: tensor<1x3xf32>, %gamma_gradient: tensor<1x3xf32>, %beta_gradient: tensor<1x3xf32>) {
  // CHECK: %{{.+}} = "mps.normalization_gradient"(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) <{epsilon = 5.000000e-01 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1>, reduction_axes = array<i64: 0>}> : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<2x3xf32>
  %0 = "mps.normalization_gradient"(%g, %x, %m, %v, %gamma, %gamma_gradient, %beta_gradient) <{epsilon = 5.000000e-01 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1>, reduction_axes = array<i64: 0>}> : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<2x3xf32>
  // CHECK: %{{.+}} = "mps.normalization_gradient"(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) <{epsilon = 5.000000e-01 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 0>, reduction_axes = array<i64: 0>}> : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<2x3xf32>
  %1 = "mps.normalization_gradient"(%g, %x, %m, %v) <{epsilon = 5.000000e-01 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 0>, reduction_axes = array<i64: 0>}> : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<2x3xf32>
  return
}
