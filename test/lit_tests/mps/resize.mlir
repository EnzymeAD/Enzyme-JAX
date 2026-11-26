// RUN: enzymexlamlir-opt --split-input-file %s | enzymexlamlir-opt --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @resize(
func.func @resize(%img: tensor<1x4x4x3xf32>) {
  // CHECK: %{{.+}} = mps.resize %{{.+}} <mode = bilinear, center_result = true, align_corners = false, layout = NHWC> : (tensor<1x4x4x3xf32>) -> tensor<1x8x8x3xf32>
  %0 = mps.resize %img <mode = bilinear, center_result = true, align_corners = false, layout = NHWC> : (tensor<1x4x4x3xf32>) -> tensor<1x8x8x3xf32>
  // CHECK: %{{.+}} = mps.resize %{{.+}} <mode = nearest, nearest_rounding_mode = floor, center_result = false, align_corners = true> : (tensor<1x4x4x3xf32>) -> tensor<1x8x8x3xf32>
  %1 = mps.resize %img <mode = nearest, nearest_rounding_mode = floor, center_result = false, align_corners = true> : (tensor<1x4x4x3xf32>) -> tensor<1x8x8x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @resize_with_scale_offset(
func.func @resize_with_scale_offset(%img: tensor<1x4x4x3xf32>, %so: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.resize_with_scale_offset %{{.+}}, %{{.+}} <mode = bilinear, layout = NHWC> : (tensor<1x4x4x3xf32>, tensor<4xf32>) -> tensor<1x8x8x3xf32>
  %0 = mps.resize_with_scale_offset %img, %so <mode = bilinear, layout = NHWC> : (tensor<1x4x4x3xf32>, tensor<4xf32>) -> tensor<1x8x8x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @resize_with_scale_and_offset(
func.func @resize_with_scale_and_offset(%img: tensor<1x4x4x3xf32>, %s: tensor<4xf32>, %o: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.resize_with_scale_and_offset %{{.+}}, %{{.+}}, %{{.+}} <mode = nearest, nearest_rounding_mode = roundToEven> : (tensor<1x4x4x3xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<1x8x8x3xf32>
  %0 = mps.resize_with_scale_and_offset %img, %s, %o <mode = nearest, nearest_rounding_mode = roundToEven> : (tensor<1x4x4x3xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<1x8x8x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @resize_gradient(
func.func @resize_gradient(%g: tensor<1x8x8x3xf32>, %img: tensor<1x4x4x3xf32>) {
  // CHECK: %{{.+}} = mps.resize_gradient %{{.+}}, %{{.+}} <mode = bilinear, center_result = true, align_corners = false, layout = NHWC> : (tensor<1x8x8x3xf32>, tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32>
  %0 = mps.resize_gradient %g, %img <mode = bilinear, center_result = true, align_corners = false, layout = NHWC> : (tensor<1x8x8x3xf32>, tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @resize_gradient_with_scale_offset(
func.func @resize_gradient_with_scale_offset(%g: tensor<1x8x8x3xf32>, %img: tensor<1x4x4x3xf32>, %so: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.resize_gradient_with_scale_offset %{{.+}}, %{{.+}}, %{{.+}} <mode = nearest, nearest_rounding_mode = ceil, layout = NHWC> : (tensor<1x8x8x3xf32>, tensor<1x4x4x3xf32>, tensor<4xf32>) -> tensor<1x4x4x3xf32>
  %0 = mps.resize_gradient_with_scale_offset %g, %img, %so <mode = nearest, nearest_rounding_mode = ceil, layout = NHWC> : (tensor<1x8x8x3xf32>, tensor<1x4x4x3xf32>, tensor<4xf32>) -> tensor<1x4x4x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @resize_gradient_with_scale_and_offset(
func.func @resize_gradient_with_scale_and_offset(%g: tensor<1x8x8x3xf32>, %img: tensor<1x4x4x3xf32>, %s: tensor<4xf32>, %o: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.resize_gradient_with_scale_and_offset %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} <mode = bilinear> : (tensor<1x8x8x3xf32>, tensor<1x4x4x3xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<1x4x4x3xf32>
  %0 = mps.resize_gradient_with_scale_and_offset %g, %img, %s, %o <mode = bilinear> : (tensor<1x8x8x3xf32>, tensor<1x4x4x3xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<1x4x4x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @sample_grid(
func.func @sample_grid(%src: tensor<1x4x4x3xf32>, %coords: tensor<1x8x8x2xf32>) {
  // CHECK: %{{.+}} = mps.sample_grid %{{.+}}, %{{.+}} <layout = NHWC, normalize_coordinates = true, relative_coordinates = false, align_corners = false, padding_mode = zero, sampling_mode = bilinear, constant_value = 0.000000e+00> : (tensor<1x4x4x3xf32>, tensor<1x8x8x2xf32>) -> tensor<1x8x8x3xf32>
  %0 = mps.sample_grid %src, %coords <layout = NHWC, normalize_coordinates = true, relative_coordinates = false, align_corners = false, padding_mode = zero, sampling_mode = bilinear, constant_value = 0.000000e+00> : (tensor<1x4x4x3xf32>, tensor<1x8x8x2xf32>) -> tensor<1x8x8x3xf32>
  // CHECK: %{{.+}} = mps.sample_grid %{{.+}}, %{{.+}} <layout = NHWC, normalize_coordinates = false, relative_coordinates = true, align_corners = true, padding_mode = clampToEdge, sampling_mode = nearest, nearest_rounding_mode = roundPreferFloor, constant_value = 1.000000e+00> : (tensor<1x4x4x3xf32>, tensor<1x8x8x2xf32>) -> tensor<1x8x8x3xf32>
  %1 = mps.sample_grid %src, %coords <layout = NHWC, normalize_coordinates = false, relative_coordinates = true, align_corners = true, padding_mode = clampToEdge, sampling_mode = nearest, nearest_rounding_mode = roundPreferFloor, constant_value = 1.000000e+00> : (tensor<1x4x4x3xf32>, tensor<1x8x8x2xf32>) -> tensor<1x8x8x3xf32>
  return
}
