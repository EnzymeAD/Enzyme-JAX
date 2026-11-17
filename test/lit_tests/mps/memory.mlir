// RUN: enzymexlamlir-opt --split-input-file %s | enzymexlamlir-opt --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @constant(
func.func @constant() {
  // CHECK: %{{.+}} = mps.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  %0 = mps.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  // CHECK: %{{.+}} = mps.constant dense<(1.000000e+00,2.000000e+00)> : tensor<complex<f32>>
  %1 = mps.constant dense<(1.000000e+00,2.000000e+00)> : tensor<complex<f32>>
  return
}

// -----

// CHECK-LABEL: func.func @variable(
func.func @variable() {
  // CHECK: %{{.+}} = mps.variable dense<0.000000e+00> : tensor<4xf32>
  %0 = mps.variable dense<0.000000e+00> : tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @variable_from_tensor(
func.func @variable_from_tensor(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.variable_from_tensor %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.variable_from_tensor %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @read(
func.func @read(%var: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.read %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.read %var : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @assign(
func.func @assign(%var: tensor<4xf32>, %value: tensor<4xf32>) {
  // CHECK: mps.assign %{{.+}}, %{{.+}} : (tensor<4xf32>, tensor<4xf32>) -> ()
  mps.assign %var, %value : (tensor<4xf32>, tensor<4xf32>) -> ()
  return
}
