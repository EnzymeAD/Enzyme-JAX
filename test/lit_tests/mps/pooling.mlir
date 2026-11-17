// RUN: enzymexlamlir-opt --split-input-file %s | enzymexlamlir-opt --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @max_pooling_2d(
func.func @max_pooling_2d(%src: tensor<1x3x8x8xf32>) {
  // CHECK: %{{.+}} = mps.max_pooling_2d %{{.+}} <kernel_width = 2, kernel_height = 2, stride_in_x = 2, stride_in_y = 2, dilation_rate_in_x = 1, dilation_rate_in_y = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_style = explicit, data_layout = NCHW, ceil_mode = false, include_zero_pad_to_average = false> : (tensor<1x3x8x8xf32>) -> tensor<1x3x4x4xf32>
  %0 = mps.max_pooling_2d %src <kernel_width = 2, kernel_height = 2, stride_in_x = 2, stride_in_y = 2, dilation_rate_in_x = 1, dilation_rate_in_y = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_style = explicit, data_layout = NCHW, ceil_mode = false, include_zero_pad_to_average = false> : (tensor<1x3x8x8xf32>) -> tensor<1x3x4x4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @max_pooling_2d_return_indices(
func.func @max_pooling_2d_return_indices(%src: tensor<1x3x8x8xf32>) {
  // CHECK: %{{.+}}, %{{.+}} = mps.max_pooling_2d_return_indices %{{.+}} <kernel_width = 2, kernel_height = 2, stride_in_x = 2, stride_in_y = 2, dilation_rate_in_x = 1, dilation_rate_in_y = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_style = explicit, data_layout = NCHW, ceil_mode = false, include_zero_pad_to_average = false, return_indices_mode = globalFlatten2D> : (tensor<1x3x8x8xf32>) -> (tensor<1x3x4x4xf32>, tensor<1x3x4x4xi32>)
  %0:2 = mps.max_pooling_2d_return_indices %src <kernel_width = 2, kernel_height = 2, stride_in_x = 2, stride_in_y = 2, dilation_rate_in_x = 1, dilation_rate_in_y = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_style = explicit, data_layout = NCHW, ceil_mode = false, include_zero_pad_to_average = false, return_indices_mode = globalFlatten2D> : (tensor<1x3x8x8xf32>) -> (tensor<1x3x4x4xf32>, tensor<1x3x4x4xi32>)
  return
}

// -----

// CHECK-LABEL: func.func @max_pooling_2d_gradient(
func.func @max_pooling_2d_gradient(%g: tensor<1x3x4x4xf32>, %src: tensor<1x3x8x8xf32>) {
  // CHECK: %{{.+}} = mps.max_pooling_2d_gradient %{{.+}}, %{{.+}} <kernel_width = 2, kernel_height = 2, stride_in_x = 2, stride_in_y = 2, dilation_rate_in_x = 1, dilation_rate_in_y = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_style = explicit, data_layout = NCHW, ceil_mode = false, include_zero_pad_to_average = false> : (tensor<1x3x4x4xf32>, tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32>
  %0 = mps.max_pooling_2d_gradient %g, %src <kernel_width = 2, kernel_height = 2, stride_in_x = 2, stride_in_y = 2, dilation_rate_in_x = 1, dilation_rate_in_y = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_style = explicit, data_layout = NCHW, ceil_mode = false, include_zero_pad_to_average = false> : (tensor<1x3x4x4xf32>, tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32>
  return
}

// -----

// CHECK-LABEL: func.func @max_pooling_2d_gradient_with_indices(
func.func @max_pooling_2d_gradient_with_indices(%g: tensor<1x3x4x4xf32>, %idx: tensor<1x3x4x4xi32>) {
  // CHECK: %{{.+}} = mps.max_pooling_2d_gradient_with_indices %{{.+}}, %{{.+}} <kernel_width = 2, kernel_height = 2, stride_in_x = 2, stride_in_y = 2, dilation_rate_in_x = 1, dilation_rate_in_y = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_style = explicit, data_layout = NCHW, ceil_mode = false, include_zero_pad_to_average = false, return_indices_mode = globalFlatten2D> : (tensor<1x3x4x4xf32>, tensor<1x3x4x4xi32>) -> tensor<1x3x8x8xf32>
  %0 = mps.max_pooling_2d_gradient_with_indices %g, %idx <kernel_width = 2, kernel_height = 2, stride_in_x = 2, stride_in_y = 2, dilation_rate_in_x = 1, dilation_rate_in_y = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_style = explicit, data_layout = NCHW, ceil_mode = false, include_zero_pad_to_average = false, return_indices_mode = globalFlatten2D> : (tensor<1x3x4x4xf32>, tensor<1x3x4x4xi32>) -> tensor<1x3x8x8xf32>
  return
}

// -----

// CHECK-LABEL: func.func @avg_pooling_2d(
func.func @avg_pooling_2d(%src: tensor<1x3x8x8xf32>) {
  // CHECK: %{{.+}} = mps.avg_pooling_2d %{{.+}} <kernel_width = 2, kernel_height = 2, stride_in_x = 2, stride_in_y = 2, dilation_rate_in_x = 1, dilation_rate_in_y = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_style = explicit, data_layout = NCHW, ceil_mode = true, include_zero_pad_to_average = true> : (tensor<1x3x8x8xf32>) -> tensor<1x3x4x4xf32>
  %0 = mps.avg_pooling_2d %src <kernel_width = 2, kernel_height = 2, stride_in_x = 2, stride_in_y = 2, dilation_rate_in_x = 1, dilation_rate_in_y = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_style = explicit, data_layout = NCHW, ceil_mode = true, include_zero_pad_to_average = true> : (tensor<1x3x8x8xf32>) -> tensor<1x3x4x4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @avg_pooling_2d_gradient(
func.func @avg_pooling_2d_gradient(%g: tensor<1x3x4x4xf32>, %src: tensor<1x3x8x8xf32>) {
  // CHECK: %{{.+}} = mps.avg_pooling_2d_gradient %{{.+}}, %{{.+}} <kernel_width = 2, kernel_height = 2, stride_in_x = 2, stride_in_y = 2, dilation_rate_in_x = 1, dilation_rate_in_y = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_style = explicit, data_layout = NCHW, ceil_mode = true, include_zero_pad_to_average = true> : (tensor<1x3x4x4xf32>, tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32>
  %0 = mps.avg_pooling_2d_gradient %g, %src <kernel_width = 2, kernel_height = 2, stride_in_x = 2, stride_in_y = 2, dilation_rate_in_x = 1, dilation_rate_in_y = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_style = explicit, data_layout = NCHW, ceil_mode = true, include_zero_pad_to_average = true> : (tensor<1x3x4x4xf32>, tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32>
  return
}

// -----

// CHECK-LABEL: func.func @max_pooling_4d(
func.func @max_pooling_4d(%src: tensor<1x3x8x8xf32>) {
  // CHECK: %{{.+}} = mps.max_pooling_4d %{{.+}} <kernel_sizes = [1, 1, 2, 2], strides = [1, 1, 2, 2], dilation_rates = [1, 1, 1, 1], padding_values = [0, 0, 0, 0, 0, 0, 0, 0], padding_style = explicit, ceil_mode = false, include_zero_pad_to_average = false> : (tensor<1x3x8x8xf32>) -> tensor<1x3x4x4xf32>
  %0 = mps.max_pooling_4d %src <kernel_sizes = [1, 1, 2, 2], strides = [1, 1, 2, 2], dilation_rates = [1, 1, 1, 1], padding_values = [0, 0, 0, 0, 0, 0, 0, 0], padding_style = explicit, ceil_mode = false, include_zero_pad_to_average = false> : (tensor<1x3x8x8xf32>) -> tensor<1x3x4x4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @max_pooling_4d_return_indices(
func.func @max_pooling_4d_return_indices(%src: tensor<1x3x8x8xf32>) {
  // CHECK: %{{.+}}, %{{.+}} = mps.max_pooling_4d_return_indices %{{.+}} <kernel_sizes = [1, 1, 2, 2], strides = [1, 1, 2, 2], dilation_rates = [1, 1, 1, 1], padding_values = [0, 0, 0, 0, 0, 0, 0, 0], padding_style = explicit, ceil_mode = false, include_zero_pad_to_average = false, return_indices_mode = localFlatten4D> : (tensor<1x3x8x8xf32>) -> (tensor<1x3x4x4xf32>, tensor<1x3x4x4xi32>)
  %0:2 = mps.max_pooling_4d_return_indices %src <kernel_sizes = [1, 1, 2, 2], strides = [1, 1, 2, 2], dilation_rates = [1, 1, 1, 1], padding_values = [0, 0, 0, 0, 0, 0, 0, 0], padding_style = explicit, ceil_mode = false, include_zero_pad_to_average = false, return_indices_mode = localFlatten4D> : (tensor<1x3x8x8xf32>) -> (tensor<1x3x4x4xf32>, tensor<1x3x4x4xi32>)
  return
}

// -----

// CHECK-LABEL: func.func @max_pooling_4d_gradient(
func.func @max_pooling_4d_gradient(%g: tensor<1x3x4x4xf32>, %src: tensor<1x3x8x8xf32>) {
  // CHECK: %{{.+}} = mps.max_pooling_4d_gradient %{{.+}}, %{{.+}} <kernel_sizes = [1, 1, 2, 2], strides = [1, 1, 2, 2], dilation_rates = [1, 1, 1, 1], padding_values = [0, 0, 0, 0, 0, 0, 0, 0], padding_style = explicit, ceil_mode = false, include_zero_pad_to_average = false> : (tensor<1x3x4x4xf32>, tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32>
  %0 = mps.max_pooling_4d_gradient %g, %src <kernel_sizes = [1, 1, 2, 2], strides = [1, 1, 2, 2], dilation_rates = [1, 1, 1, 1], padding_values = [0, 0, 0, 0, 0, 0, 0, 0], padding_style = explicit, ceil_mode = false, include_zero_pad_to_average = false> : (tensor<1x3x4x4xf32>, tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32>
  return
}

// -----

// CHECK-LABEL: func.func @max_pooling_4d_gradient_with_indices(
func.func @max_pooling_4d_gradient_with_indices(%g: tensor<1x3x4x4xf32>, %idx: tensor<1x3x4x4xi32>) {
  // CHECK: %{{.+}} = mps.max_pooling_4d_gradient_with_indices %{{.+}}, %{{.+}} <kernel_sizes = [1, 1, 2, 2], strides = [1, 1, 2, 2], dilation_rates = [1, 1, 1, 1], padding_values = [0, 0, 0, 0, 0, 0, 0, 0], padding_style = explicit, ceil_mode = false, include_zero_pad_to_average = false, return_indices_mode = localFlatten4D> : (tensor<1x3x4x4xf32>, tensor<1x3x4x4xi32>) -> tensor<1x3x8x8xf32>
  %0 = mps.max_pooling_4d_gradient_with_indices %g, %idx <kernel_sizes = [1, 1, 2, 2], strides = [1, 1, 2, 2], dilation_rates = [1, 1, 1, 1], padding_values = [0, 0, 0, 0, 0, 0, 0, 0], padding_style = explicit, ceil_mode = false, include_zero_pad_to_average = false, return_indices_mode = localFlatten4D> : (tensor<1x3x4x4xf32>, tensor<1x3x4x4xi32>) -> tensor<1x3x8x8xf32>
  return
}

// -----

// CHECK-LABEL: func.func @avg_pooling_4d(
func.func @avg_pooling_4d(%src: tensor<1x3x8x8xf32>) {
  // CHECK: %{{.+}} = mps.avg_pooling_4d %{{.+}} <kernel_sizes = [1, 1, 2, 2], strides = [1, 1, 2, 2], dilation_rates = [1, 1, 1, 1], padding_values = [0, 0, 0, 0, 0, 0, 0, 0], padding_style = explicit, ceil_mode = false, include_zero_pad_to_average = false> : (tensor<1x3x8x8xf32>) -> tensor<1x3x4x4xf32>
  %0 = mps.avg_pooling_4d %src <kernel_sizes = [1, 1, 2, 2], strides = [1, 1, 2, 2], dilation_rates = [1, 1, 1, 1], padding_values = [0, 0, 0, 0, 0, 0, 0, 0], padding_style = explicit, ceil_mode = false, include_zero_pad_to_average = false> : (tensor<1x3x8x8xf32>) -> tensor<1x3x4x4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @avg_pooling_4d_gradient(
func.func @avg_pooling_4d_gradient(%g: tensor<1x3x4x4xf32>, %src: tensor<1x3x8x8xf32>) {
  // CHECK: %{{.+}} = mps.avg_pooling_4d_gradient %{{.+}}, %{{.+}} <kernel_sizes = [1, 1, 2, 2], strides = [1, 1, 2, 2], dilation_rates = [1, 1, 1, 1], padding_values = [0, 0, 0, 0, 0, 0, 0, 0], padding_style = explicit, ceil_mode = false, include_zero_pad_to_average = false> : (tensor<1x3x4x4xf32>, tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32>
  %0 = mps.avg_pooling_4d_gradient %g, %src <kernel_sizes = [1, 1, 2, 2], strides = [1, 1, 2, 2], dilation_rates = [1, 1, 1, 1], padding_values = [0, 0, 0, 0, 0, 0, 0, 0], padding_style = explicit, ceil_mode = false, include_zero_pad_to_average = false> : (tensor<1x3x4x4xf32>, tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32>
  return
}

// -----

// CHECK-LABEL: func.func @l2_norm_pooling_4d(
func.func @l2_norm_pooling_4d(%src: tensor<1x3x8x8xf32>) {
  // CHECK: %{{.+}} = mps.l2_norm_pooling_4d %{{.+}} <kernel_sizes = [1, 1, 2, 2], strides = [1, 1, 2, 2], dilation_rates = [1, 1, 1, 1], padding_values = [0, 0, 0, 0, 0, 0, 0, 0], padding_style = explicit, ceil_mode = false, include_zero_pad_to_average = false> : (tensor<1x3x8x8xf32>) -> tensor<1x3x4x4xf32>
  %0 = mps.l2_norm_pooling_4d %src <kernel_sizes = [1, 1, 2, 2], strides = [1, 1, 2, 2], dilation_rates = [1, 1, 1, 1], padding_values = [0, 0, 0, 0, 0, 0, 0, 0], padding_style = explicit, ceil_mode = false, include_zero_pad_to_average = false> : (tensor<1x3x8x8xf32>) -> tensor<1x3x4x4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @l2_norm_pooling_4d_gradient(
func.func @l2_norm_pooling_4d_gradient(%g: tensor<1x3x4x4xf32>, %src: tensor<1x3x8x8xf32>) {
  // CHECK: %{{.+}} = mps.l2_norm_pooling_4d_gradient %{{.+}}, %{{.+}} <kernel_sizes = [1, 1, 2, 2], strides = [1, 1, 2, 2], dilation_rates = [1, 1, 1, 1], padding_values = [0, 0, 0, 0, 0, 0, 0, 0], padding_style = explicit, ceil_mode = false, include_zero_pad_to_average = false> : (tensor<1x3x4x4xf32>, tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32>
  %0 = mps.l2_norm_pooling_4d_gradient %g, %src <kernel_sizes = [1, 1, 2, 2], strides = [1, 1, 2, 2], dilation_rates = [1, 1, 1, 1], padding_values = [0, 0, 0, 0, 0, 0, 0, 0], padding_style = explicit, ceil_mode = false, include_zero_pad_to_average = false> : (tensor<1x3x4x4xf32>, tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32>
  return
}
