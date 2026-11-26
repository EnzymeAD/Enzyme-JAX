// RUN: enzymexlamlir-opt --split-input-file %s | enzymexlamlir-opt --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @reduction_sum(
func.func @reduction_sum(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.reduction_sum %{{.+}} <axes = [1]> : (tensor<2x3xf32>) -> tensor<2x1xf32>
  %0 = mps.reduction_sum %x <axes = [1]> : (tensor<2x3xf32>) -> tensor<2x1xf32>
  // CHECK: %{{.+}} = mps.reduction_sum %{{.+}} : (tensor<2x3xf32>) -> tensor<1x1xf32>
  %1 = mps.reduction_sum %x : (tensor<2x3xf32>) -> tensor<1x1xf32>
  return
}

// -----

// CHECK-LABEL: func.func @reduction_product(
func.func @reduction_product(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.reduction_product %{{.+}} <axes = [0]> : (tensor<2x3xf32>) -> tensor<1x3xf32>
  %0 = mps.reduction_product %x <axes = [0]> : (tensor<2x3xf32>) -> tensor<1x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @reduction_maximum(
func.func @reduction_maximum(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.reduction_maximum %{{.+}} <axes = [0, 1]> : (tensor<2x3xf32>) -> tensor<1x1xf32>
  %0 = mps.reduction_maximum %x <axes = [0, 1]> : (tensor<2x3xf32>) -> tensor<1x1xf32>
  return
}

// -----

// CHECK-LABEL: func.func @reduction_minimum(
func.func @reduction_minimum(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.reduction_minimum %{{.+}} <axes = [1]> : (tensor<2x3xf32>) -> tensor<2x1xf32>
  %0 = mps.reduction_minimum %x <axes = [1]> : (tensor<2x3xf32>) -> tensor<2x1xf32>
  return
}

// -----

// CHECK-LABEL: func.func @reduction_maximum_propagate_nan(
func.func @reduction_maximum_propagate_nan(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.reduction_maximum_propagate_nan %{{.+}} <axes = [1]> : (tensor<2x3xf32>) -> tensor<2x1xf32>
  %0 = mps.reduction_maximum_propagate_nan %x <axes = [1]> : (tensor<2x3xf32>) -> tensor<2x1xf32>
  return
}

// -----

// CHECK-LABEL: func.func @reduction_minimum_propagate_nan(
func.func @reduction_minimum_propagate_nan(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.reduction_minimum_propagate_nan %{{.+}} <axes = [1]> : (tensor<2x3xf32>) -> tensor<2x1xf32>
  %0 = mps.reduction_minimum_propagate_nan %x <axes = [1]> : (tensor<2x3xf32>) -> tensor<2x1xf32>
  return
}

// -----

// CHECK-LABEL: func.func @reduction_and(
func.func @reduction_and(%b: tensor<2x3xi1>) {
  // CHECK: %{{.+}} = mps.reduction_and %{{.+}} <axes = [1]> : (tensor<2x3xi1>) -> tensor<2x1xi1>
  %0 = mps.reduction_and %b <axes = [1]> : (tensor<2x3xi1>) -> tensor<2x1xi1>
  return
}

// -----

// CHECK-LABEL: func.func @reduction_or(
func.func @reduction_or(%b: tensor<2x3xi1>) {
  // CHECK: %{{.+}} = mps.reduction_or %{{.+}} <axes = [1]> : (tensor<2x3xi1>) -> tensor<2x1xi1>
  %0 = mps.reduction_or %b <axes = [1]> : (tensor<2x3xi1>) -> tensor<2x1xi1>
  return
}

// -----

// CHECK-LABEL: func.func @reduction_arg_maximum(
func.func @reduction_arg_maximum(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.reduction_arg_maximum %{{.+}} <axis = 1> : (tensor<2x3xf32>) -> tensor<2x1xi32>
  %0 = mps.reduction_arg_maximum %x <axis = 1> : (tensor<2x3xf32>) -> tensor<2x1xi32>
  return
}

// -----

// CHECK-LABEL: func.func @reduction_arg_minimum(
func.func @reduction_arg_minimum(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.reduction_arg_minimum %{{.+}} <axis = 1> : (tensor<2x3xf32>) -> tensor<2x1xi32>
  %0 = mps.reduction_arg_minimum %x <axis = 1> : (tensor<2x3xf32>) -> tensor<2x1xi32>
  return
}

// -----

// CHECK-LABEL: func.func @cumulative_sum(
func.func @cumulative_sum(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.cumulative_sum %{{.+}} <axis = 1> : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %0 = mps.cumulative_sum %x <axis = 1> : (tensor<2x3xf32>) -> tensor<2x3xf32>
  // CHECK: %{{.+}} = mps.cumulative_sum %{{.+}} <axis = 1, exclusive = true, reverse = true> : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %1 = mps.cumulative_sum %x <axis = 1, exclusive = true, reverse = true> : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @cumulative_product(
func.func @cumulative_product(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.cumulative_product %{{.+}} <axis = 0> : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %0 = mps.cumulative_product %x <axis = 0> : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @cumulative_minimum(
func.func @cumulative_minimum(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.cumulative_minimum %{{.+}} <axis = 0, exclusive = true> : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %0 = mps.cumulative_minimum %x <axis = 0, exclusive = true> : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @cumulative_maximum(
func.func @cumulative_maximum(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.cumulative_maximum %{{.+}} <axis = 0, reverse = true> : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %0 = mps.cumulative_maximum %x <axis = 0, reverse = true> : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return
}
