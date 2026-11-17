// RUN: enzymexlamlir-opt --split-input-file %s | enzymexlamlir-opt --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @non_maximum_suppression(
func.func @non_maximum_suppression(%boxes: tensor<1x8x4xf32>, %scores: tensor<1x8x1xf32>, %classes: tensor<1x8xi32>) {
  // CHECK: %{{.+}} = mps.non_maximum_suppression %{{.+}}, %{{.+}} <iou_threshold = 5.000000e-01, score_threshold = 2.500000e-01, per_class_suppression = false, coordinate_mode = cornersHeightFirst> : (tensor<1x8x4xf32>, tensor<1x8x1xf32>) -> tensor<1x8xi32>
  %0 = mps.non_maximum_suppression %boxes, %scores <iou_threshold = 5.000000e-01, score_threshold = 2.500000e-01, per_class_suppression = false, coordinate_mode = cornersHeightFirst> : (tensor<1x8x4xf32>, tensor<1x8x1xf32>) -> tensor<1x8xi32>
  // CHECK: %{{.+}} = mps.non_maximum_suppression %{{.+}}, %{{.+}}, %{{.+}} <iou_threshold = 5.000000e-01, score_threshold = 2.500000e-01, per_class_suppression = true, coordinate_mode = centersWidthFirst> : (tensor<1x8x4xf32>, tensor<1x8x1xf32>, tensor<1x8xi32>) -> tensor<1x8xi32>
  %1 = mps.non_maximum_suppression %boxes, %scores, %classes <iou_threshold = 5.000000e-01, score_threshold = 2.500000e-01, per_class_suppression = true, coordinate_mode = centersWidthFirst> : (tensor<1x8x4xf32>, tensor<1x8x1xf32>, tensor<1x8xi32>) -> tensor<1x8xi32>
  return
}

// -----

// CHECK-LABEL: func.func @non_zero_indices(
func.func @non_zero_indices(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.non_zero_indices %{{.+}} : (tensor<2x3xf32>) -> tensor<?x2xi32>
  %0 = mps.non_zero_indices %x : (tensor<2x3xf32>) -> tensor<?x2xi32>
  return
}

// -----

// CHECK-LABEL: func.func @one_hot(
func.func @one_hot(%i: tensor<3xi32>) {
  // CHECK: %{{.+}} = mps.one_hot %{{.+}} : (tensor<3xi32>) -> tensor<3x4xf32>
  %0 = mps.one_hot %i : (tensor<3xi32>) -> tensor<3x4xf32>
  // CHECK: %{{.+}} = mps.one_hot %{{.+}} <axis = 0, on_value = 2.000000e+00, off_value = -1.000000e+00> : (tensor<3xi32>) -> tensor<4x3xf32>
  %1 = mps.one_hot %i <axis = 0, on_value = 2.000000e+00, off_value = -1.000000e+00> : (tensor<3xi32>) -> tensor<4x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @sparse_tensor(
func.func @sparse_tensor(%vals: tensor<5xf32>, %rows: tensor<5xi32>, %cols: tensor<5xi32>) {
  // CHECK: %{{.+}} = mps.sparse_tensor %{{.+}}, %{{.+}}, %{{.+}} <storage_type = COO> : (tensor<5xf32>, tensor<5xi32>, tensor<5xi32>) -> tensor<4x4xf32>
  %0 = mps.sparse_tensor %vals, %rows, %cols <storage_type = COO> : (tensor<5xf32>, tensor<5xi32>, tensor<5xi32>) -> tensor<4x4xf32>
  return
}
