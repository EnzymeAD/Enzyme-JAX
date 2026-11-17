// RUN: enzymexlamlir-opt --split-input-file %s | enzymexlamlir-opt --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @sort(
func.func @sort(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.sort %{{.+}} <axis = 1> : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %0 = mps.sort %x <axis = 1> : (tensor<2x3xf32>) -> tensor<2x3xf32>
  // CHECK: %{{.+}} = mps.sort %{{.+}} <axis = 1, descending = true> : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %1 = mps.sort %x <axis = 1, descending = true> : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @arg_sort(
func.func @arg_sort(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.arg_sort %{{.+}} <axis = 1> : (tensor<2x3xf32>) -> tensor<2x3xi32>
  %0 = mps.arg_sort %x <axis = 1> : (tensor<2x3xf32>) -> tensor<2x3xi32>
  // CHECK: %{{.+}} = mps.arg_sort %{{.+}} <axis = 0, descending = true> : (tensor<2x3xf32>) -> tensor<2x3xi32>
  %1 = mps.arg_sort %x <axis = 0, descending = true> : (tensor<2x3xf32>) -> tensor<2x3xi32>
  return
}

// -----

// CHECK-LABEL: func.func @top_k(
func.func @top_k(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}}, %{{.+}} = mps.top_k %{{.+}} <k = 2> : (tensor<2x3xf32>) -> (tensor<2x2xf32>, tensor<2x2xi32>)
  %0:2 = mps.top_k %x <k = 2> : (tensor<2x3xf32>) -> (tensor<2x2xf32>, tensor<2x2xi32>)
  // CHECK: %{{.+}}, %{{.+}} = mps.top_k %{{.+}} <axis = 0, k = 1> : (tensor<2x3xf32>) -> (tensor<1x3xf32>, tensor<1x3xi32>)
  %1:2 = mps.top_k %x <axis = 0, k = 1> : (tensor<2x3xf32>) -> (tensor<1x3xf32>, tensor<1x3xi32>)
  return
}

// -----

// CHECK-LABEL: func.func @bottom_k(
func.func @bottom_k(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}}, %{{.+}} = mps.bottom_k %{{.+}} <axis = 1, k = 2> : (tensor<2x3xf32>) -> (tensor<2x2xf32>, tensor<2x2xi32>)
  %0:2 = mps.bottom_k %x <axis = 1, k = 2> : (tensor<2x3xf32>) -> (tensor<2x2xf32>, tensor<2x2xi32>)
  return
}

// -----

// CHECK-LABEL: func.func @top_k_gradient(
func.func @top_k_gradient(%g: tensor<2x2xf32>, %x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.top_k_gradient %{{.+}}, %{{.+}} <k = 2> : (tensor<2x2xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %0 = mps.top_k_gradient %g, %x <k = 2> : (tensor<2x2xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @bottom_k_gradient(
func.func @bottom_k_gradient(%g: tensor<2x2xf32>, %x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.bottom_k_gradient %{{.+}}, %{{.+}} <axis = 1, k = 2> : (tensor<2x2xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %0 = mps.bottom_k_gradient %g, %x <axis = 1, k = 2> : (tensor<2x2xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return
}
