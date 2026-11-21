// RUN: enzymexlamlir-opt --split-input-file %s | enzymexlamlir-opt --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @relu(
func.func @relu(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.relu %{{.+}} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %0 = mps.relu %x : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @relu_gradient(
func.func @relu_gradient(%g: tensor<2x3xf32>, %x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.relu_gradient %{{.+}}, %{{.+}} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %0 = mps.relu_gradient %g, %x : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @sigmoid(
func.func @sigmoid(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.sigmoid %{{.+}} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %0 = mps.sigmoid %x : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @sigmoid_gradient(
func.func @sigmoid_gradient(%g: tensor<2x3xf32>, %x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.sigmoid_gradient %{{.+}}, %{{.+}} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %0 = mps.sigmoid_gradient %g, %x : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @softmax(
func.func @softmax(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.softmax %{{.+}} <axis = 1> : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %0 = mps.softmax %x <axis = 1> : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @softmax_gradient(
func.func @softmax_gradient(%g: tensor<2x3xf32>, %x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.softmax_gradient %{{.+}}, %{{.+}} <axis = 1> : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %0 = mps.softmax_gradient %g, %x <axis = 1> : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @leaky_relu(
func.func @leaky_relu(%x: tensor<2x3xf32>, %alpha: tensor<f32>) {
  // CHECK: %{{.+}} = mps.leaky_relu %{{.+}}, %{{.+}} : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
  %0 = mps.leaky_relu %x, %alpha : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @leaky_relu_gradient(
func.func @leaky_relu_gradient(%g: tensor<2x3xf32>, %x: tensor<2x3xf32>, %alpha: tensor<f32>) {
  // CHECK: %{{.+}} = mps.leaky_relu_gradient %{{.+}}, %{{.+}}, %{{.+}} : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
  %0 = mps.leaky_relu_gradient %g, %x, %alpha : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @softmax_cross_entropy(
func.func @softmax_cross_entropy(%x: tensor<2x3xf32>, %labels: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.softmax_cross_entropy %{{.+}}, %{{.+}} <axis = 1, reduction_type = mean> : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<f32>
  %0 = mps.softmax_cross_entropy %x, %labels <axis = 1, reduction_type = mean> : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<f32>
  return
}

// -----

// CHECK-LABEL: func.func @softmax_cross_entropy_gradient(
func.func @softmax_cross_entropy_gradient(%g: tensor<f32>, %x: tensor<2x3xf32>, %labels: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.softmax_cross_entropy_gradient %{{.+}}, %{{.+}}, %{{.+}} <axis = 1, reduction_type = sum> : (tensor<f32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %0 = mps.softmax_cross_entropy_gradient %g, %x, %labels <axis = 1, reduction_type = sum> : (tensor<f32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return
}
