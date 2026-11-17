// RUN: enzymexlamlir-opt --split-input-file %s | enzymexlamlir-opt --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @convolution_2d(
func.func @convolution_2d(%src: tensor<1x3x8x8xf32>, %w: tensor<4x3x3x3xf32>) {
  // CHECK: %{{.+}} = mps.convolution_2d %{{.+}}, %{{.+}} <stride_in_x = 1, stride_in_y = 1, dilation_rate_in_x = 1, dilation_rate_in_y = 1, groups = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_style = explicit, data_layout = NCHW, weights_layout = OIHW> : (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>) -> tensor<1x4x6x6xf32>
  %0 = mps.convolution_2d %src, %w <stride_in_x = 1, stride_in_y = 1, dilation_rate_in_x = 1, dilation_rate_in_y = 1, groups = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_style = explicit, data_layout = NCHW, weights_layout = OIHW> : (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>) -> tensor<1x4x6x6xf32>
  return
}

// -----

// CHECK-LABEL: func.func @convolution_2d_data_gradient(
func.func @convolution_2d_data_gradient(%dst: tensor<1x4x6x6xf32>, %w: tensor<4x3x3x3xf32>) {
  // CHECK: %{{.+}} = mps.convolution_2d_data_gradient %{{.+}}, %{{.+}} <stride_in_x = 1, stride_in_y = 1, dilation_rate_in_x = 1, dilation_rate_in_y = 1, groups = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_style = TF_VALID, data_layout = NCHW, weights_layout = OIHW> : (tensor<1x4x6x6xf32>, tensor<4x3x3x3xf32>) -> tensor<1x3x8x8xf32>
  %0 = mps.convolution_2d_data_gradient %dst, %w <stride_in_x = 1, stride_in_y = 1, dilation_rate_in_x = 1, dilation_rate_in_y = 1, groups = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_style = TF_VALID, data_layout = NCHW, weights_layout = OIHW> : (tensor<1x4x6x6xf32>, tensor<4x3x3x3xf32>) -> tensor<1x3x8x8xf32>
  return
}

// -----

// CHECK-LABEL: func.func @convolution_2d_weights_gradient(
func.func @convolution_2d_weights_gradient(%dst: tensor<1x4x6x6xf32>, %src: tensor<1x3x8x8xf32>) {
  // CHECK: %{{.+}} = mps.convolution_2d_weights_gradient %{{.+}}, %{{.+}} <stride_in_x = 1, stride_in_y = 1, dilation_rate_in_x = 1, dilation_rate_in_y = 1, groups = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_style = TF_SAME, data_layout = NCHW, weights_layout = OIHW> : (tensor<1x4x6x6xf32>, tensor<1x3x8x8xf32>) -> tensor<4x3x3x3xf32>
  %0 = mps.convolution_2d_weights_gradient %dst, %src <stride_in_x = 1, stride_in_y = 1, dilation_rate_in_x = 1, dilation_rate_in_y = 1, groups = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_style = TF_SAME, data_layout = NCHW, weights_layout = OIHW> : (tensor<1x4x6x6xf32>, tensor<1x3x8x8xf32>) -> tensor<4x3x3x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @convolution_3d(
func.func @convolution_3d(%src: tensor<1x2x4x4x4xf32>, %w: tensor<3x2x3x3x3xf32>) {
  // CHECK: %{{.+}} = mps.convolution_3d %{{.+}}, %{{.+}} <stride_in_x = 1, stride_in_y = 1, stride_in_z = 1, dilation_rate_in_x = 1, dilation_rate_in_y = 1, dilation_rate_in_z = 1, groups = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_front = 0, padding_back = 0, padding_style = explicit, data_layout = NCDHW, weights_layout = OIDHW> : (tensor<1x2x4x4x4xf32>, tensor<3x2x3x3x3xf32>) -> tensor<1x3x2x2x2xf32>
  %0 = mps.convolution_3d %src, %w <stride_in_x = 1, stride_in_y = 1, stride_in_z = 1, dilation_rate_in_x = 1, dilation_rate_in_y = 1, dilation_rate_in_z = 1, groups = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_front = 0, padding_back = 0, padding_style = explicit, data_layout = NCDHW, weights_layout = OIDHW> : (tensor<1x2x4x4x4xf32>, tensor<3x2x3x3x3xf32>) -> tensor<1x3x2x2x2xf32>
  return
}

// -----

// CHECK-LABEL: func.func @convolution_3d_data_gradient(
func.func @convolution_3d_data_gradient(%dst: tensor<1x3x2x2x2xf32>, %w: tensor<3x2x3x3x3xf32>) {
  // CHECK: %{{.+}} = mps.convolution_3d_data_gradient %{{.+}}, %{{.+}} <stride_in_x = 1, stride_in_y = 1, stride_in_z = 1, dilation_rate_in_x = 1, dilation_rate_in_y = 1, dilation_rate_in_z = 1, groups = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_front = 0, padding_back = 0, padding_style = explicitOffset, data_layout = NCDHW, weights_layout = OIDHW> : (tensor<1x3x2x2x2xf32>, tensor<3x2x3x3x3xf32>) -> tensor<1x2x4x4x4xf32>
  %0 = mps.convolution_3d_data_gradient %dst, %w <stride_in_x = 1, stride_in_y = 1, stride_in_z = 1, dilation_rate_in_x = 1, dilation_rate_in_y = 1, dilation_rate_in_z = 1, groups = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_front = 0, padding_back = 0, padding_style = explicitOffset, data_layout = NCDHW, weights_layout = OIDHW> : (tensor<1x3x2x2x2xf32>, tensor<3x2x3x3x3xf32>) -> tensor<1x2x4x4x4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @convolution_3d_weights_gradient(
func.func @convolution_3d_weights_gradient(%dst: tensor<1x3x2x2x2xf32>, %src: tensor<1x2x4x4x4xf32>) {
  // CHECK: %{{.+}} = mps.convolution_3d_weights_gradient %{{.+}}, %{{.+}} <stride_in_x = 1, stride_in_y = 1, stride_in_z = 1, dilation_rate_in_x = 1, dilation_rate_in_y = 1, dilation_rate_in_z = 1, groups = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_front = 0, padding_back = 0, padding_style = ONNX_SAME_LOWER, data_layout = NDHWC, weights_layout = DHWIO> : (tensor<1x3x2x2x2xf32>, tensor<1x2x4x4x4xf32>) -> tensor<3x2x3x3x3xf32>
  %0 = mps.convolution_3d_weights_gradient %dst, %src <stride_in_x = 1, stride_in_y = 1, stride_in_z = 1, dilation_rate_in_x = 1, dilation_rate_in_y = 1, dilation_rate_in_z = 1, groups = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_front = 0, padding_back = 0, padding_style = ONNX_SAME_LOWER, data_layout = NDHWC, weights_layout = DHWIO> : (tensor<1x3x2x2x2xf32>, tensor<1x2x4x4x4xf32>) -> tensor<3x2x3x3x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @convolution_transpose_2d(
func.func @convolution_transpose_2d(%src: tensor<1x4x6x6xf32>, %w: tensor<4x3x3x3xf32>) {
  // CHECK: %{{.+}} = mps.convolution_transpose_2d %{{.+}}, %{{.+}} <stride_in_x = 1, stride_in_y = 1, dilation_rate_in_x = 1, dilation_rate_in_y = 1, groups = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_style = explicit, data_layout = NCHW, weights_layout = OIHW> : (tensor<1x4x6x6xf32>, tensor<4x3x3x3xf32>) -> tensor<1x3x8x8xf32>
  %0 = mps.convolution_transpose_2d %src, %w <stride_in_x = 1, stride_in_y = 1, dilation_rate_in_x = 1, dilation_rate_in_y = 1, groups = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_style = explicit, data_layout = NCHW, weights_layout = OIHW> : (tensor<1x4x6x6xf32>, tensor<4x3x3x3xf32>) -> tensor<1x3x8x8xf32>
  return
}

// -----

// CHECK-LABEL: func.func @convolution_transpose_2d_data_gradient(
func.func @convolution_transpose_2d_data_gradient(%dst: tensor<1x3x8x8xf32>, %w: tensor<4x3x3x3xf32>) {
  // CHECK: %{{.+}} = mps.convolution_transpose_2d_data_gradient %{{.+}}, %{{.+}} <stride_in_x = 1, stride_in_y = 1, dilation_rate_in_x = 1, dilation_rate_in_y = 1, groups = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_style = explicit, data_layout = NCHW, weights_layout = OIHW> : (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>) -> tensor<1x4x6x6xf32>
  %0 = mps.convolution_transpose_2d_data_gradient %dst, %w <stride_in_x = 1, stride_in_y = 1, dilation_rate_in_x = 1, dilation_rate_in_y = 1, groups = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_style = explicit, data_layout = NCHW, weights_layout = OIHW> : (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>) -> tensor<1x4x6x6xf32>
  return
}

// -----

// CHECK-LABEL: func.func @convolution_transpose_2d_weights_gradient(
func.func @convolution_transpose_2d_weights_gradient(%dst: tensor<1x3x8x8xf32>, %src: tensor<1x4x6x6xf32>) {
  // CHECK: %{{.+}} = mps.convolution_transpose_2d_weights_gradient %{{.+}}, %{{.+}} <stride_in_x = 1, stride_in_y = 1, dilation_rate_in_x = 1, dilation_rate_in_y = 1, groups = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_style = explicit, data_layout = NCHW, weights_layout = OIHW> : (tensor<1x3x8x8xf32>, tensor<1x4x6x6xf32>) -> tensor<4x3x3x3xf32>
  %0 = mps.convolution_transpose_2d_weights_gradient %dst, %src <stride_in_x = 1, stride_in_y = 1, dilation_rate_in_x = 1, dilation_rate_in_y = 1, groups = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_style = explicit, data_layout = NCHW, weights_layout = OIHW> : (tensor<1x3x8x8xf32>, tensor<1x4x6x6xf32>) -> tensor<4x3x3x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @depthwise_convolution_2d(
func.func @depthwise_convolution_2d(%src: tensor<1x8x8x3xf32>, %w: tensor<3x3x3x1xf32>) {
  // CHECK: %{{.+}} = mps.depthwise_convolution_2d %{{.+}}, %{{.+}} <stride_in_x = 1, stride_in_y = 1, dilation_rate_in_x = 1, dilation_rate_in_y = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_style = explicit, data_layout = NHWC, weights_layout = HWIO> : (tensor<1x8x8x3xf32>, tensor<3x3x3x1xf32>) -> tensor<1x6x6x3xf32>
  %0 = mps.depthwise_convolution_2d %src, %w <stride_in_x = 1, stride_in_y = 1, dilation_rate_in_x = 1, dilation_rate_in_y = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_style = explicit, data_layout = NHWC, weights_layout = HWIO> : (tensor<1x8x8x3xf32>, tensor<3x3x3x1xf32>) -> tensor<1x6x6x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @depthwise_convolution_2d_data_gradient(
func.func @depthwise_convolution_2d_data_gradient(%dst: tensor<1x6x6x3xf32>, %w: tensor<3x3x3x1xf32>) {
  // CHECK: %{{.+}} = mps.depthwise_convolution_2d_data_gradient %{{.+}}, %{{.+}} <stride_in_x = 1, stride_in_y = 1, dilation_rate_in_x = 1, dilation_rate_in_y = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_style = explicit, data_layout = NHWC, weights_layout = HWIO> : (tensor<1x6x6x3xf32>, tensor<3x3x3x1xf32>) -> tensor<1x8x8x3xf32>
  %0 = mps.depthwise_convolution_2d_data_gradient %dst, %w <stride_in_x = 1, stride_in_y = 1, dilation_rate_in_x = 1, dilation_rate_in_y = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_style = explicit, data_layout = NHWC, weights_layout = HWIO> : (tensor<1x6x6x3xf32>, tensor<3x3x3x1xf32>) -> tensor<1x8x8x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @depthwise_convolution_2d_weights_gradient(
func.func @depthwise_convolution_2d_weights_gradient(%dst: tensor<1x6x6x3xf32>, %src: tensor<1x8x8x3xf32>) {
  // CHECK: %{{.+}} = mps.depthwise_convolution_2d_weights_gradient %{{.+}}, %{{.+}} <stride_in_x = 1, stride_in_y = 1, dilation_rate_in_x = 1, dilation_rate_in_y = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_style = explicit, data_layout = NHWC, weights_layout = HWIO> : (tensor<1x6x6x3xf32>, tensor<1x8x8x3xf32>) -> tensor<3x3x3x1xf32>
  %0 = mps.depthwise_convolution_2d_weights_gradient %dst, %src <stride_in_x = 1, stride_in_y = 1, dilation_rate_in_x = 1, dilation_rate_in_y = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, padding_style = explicit, data_layout = NHWC, weights_layout = HWIO> : (tensor<1x6x6x3xf32>, tensor<1x8x8x3xf32>) -> tensor<3x3x3x1xf32>
  return
}

// -----

// CHECK-LABEL: func.func @depthwise_convolution_3d(
func.func @depthwise_convolution_3d(%src: tensor<1x3x4x4x4xf32>, %w: tensor<1x3x3x3x3xf32>) {
  // CHECK: %{{.+}} = mps.depthwise_convolution_3d %{{.+}}, %{{.+}} <strides = [1, 1, 1], dilation_rates = [1, 1, 1], padding_values = [0, 0, 0, 0, 0, 0], padding_style = explicit, channel_dimension_index = -4> : (tensor<1x3x4x4x4xf32>, tensor<1x3x3x3x3xf32>) -> tensor<1x3x2x2x2xf32>
  %0 = mps.depthwise_convolution_3d %src, %w <strides = [1, 1, 1], dilation_rates = [1, 1, 1], padding_values = [0, 0, 0, 0, 0, 0], padding_style = explicit, channel_dimension_index = -4> : (tensor<1x3x4x4x4xf32>, tensor<1x3x3x3x3xf32>) -> tensor<1x3x2x2x2xf32>
  return
}

// -----

// CHECK-LABEL: func.func @depthwise_convolution_3d_data_gradient(
func.func @depthwise_convolution_3d_data_gradient(%dst: tensor<1x3x2x2x2xf32>, %w: tensor<1x3x3x3x3xf32>) {
  // CHECK: %{{.+}} = mps.depthwise_convolution_3d_data_gradient %{{.+}}, %{{.+}} <strides = [1, 1, 1], dilation_rates = [1, 1, 1], padding_values = [0, 0, 0, 0, 0, 0], padding_style = explicit, channel_dimension_index = -4> : (tensor<1x3x2x2x2xf32>, tensor<1x3x3x3x3xf32>) -> tensor<1x3x4x4x4xf32>
  %0 = mps.depthwise_convolution_3d_data_gradient %dst, %w <strides = [1, 1, 1], dilation_rates = [1, 1, 1], padding_values = [0, 0, 0, 0, 0, 0], padding_style = explicit, channel_dimension_index = -4> : (tensor<1x3x2x2x2xf32>, tensor<1x3x3x3x3xf32>) -> tensor<1x3x4x4x4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @depthwise_convolution_3d_weights_gradient(
func.func @depthwise_convolution_3d_weights_gradient(%dst: tensor<1x3x2x2x2xf32>, %src: tensor<1x3x4x4x4xf32>) {
  // CHECK: %{{.+}} = mps.depthwise_convolution_3d_weights_gradient %{{.+}}, %{{.+}} <strides = [1, 1, 1], dilation_rates = [1, 1, 1], padding_values = [0, 0, 0, 0, 0, 0], padding_style = explicit, channel_dimension_index = -4> : (tensor<1x3x2x2x2xf32>, tensor<1x3x4x4x4xf32>) -> tensor<1x3x3x3x3xf32>
  %0 = mps.depthwise_convolution_3d_weights_gradient %dst, %src <strides = [1, 1, 1], dilation_rates = [1, 1, 1], padding_values = [0, 0, 0, 0, 0, 0], padding_style = explicit, channel_dimension_index = -4> : (tensor<1x3x2x2x2xf32>, tensor<1x3x4x4x4xf32>) -> tensor<1x3x3x3x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @im_to_col(
func.func @im_to_col(%img: tensor<1x3x8x8xf32>) {
  // CHECK: %{{.+}} = mps.im_to_col %{{.+}} <kernel_width = 3, kernel_height = 3, stride_in_x = 1, stride_in_y = 1, dilation_rate_in_x = 1, dilation_rate_in_y = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, data_layout = NCHW> : (tensor<1x3x8x8xf32>) -> tensor<1x27x36xf32>
  %0 = mps.im_to_col %img <kernel_width = 3, kernel_height = 3, stride_in_x = 1, stride_in_y = 1, dilation_rate_in_x = 1, dilation_rate_in_y = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, data_layout = NCHW> : (tensor<1x3x8x8xf32>) -> tensor<1x27x36xf32>
  return
}

// -----

// CHECK-LABEL: func.func @col_to_im(
func.func @col_to_im(%col: tensor<1x27x36xf32>) {
  // CHECK: %{{.+}} = mps.col_to_im %{{.+}} <kernel_width = 3, kernel_height = 3, stride_in_x = 1, stride_in_y = 1, dilation_rate_in_x = 1, dilation_rate_in_y = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, data_layout = NCHW> : (tensor<1x27x36xf32>) -> tensor<1x3x8x8xf32>
  %0 = mps.col_to_im %col <kernel_width = 3, kernel_height = 3, stride_in_x = 1, stride_in_y = 1, dilation_rate_in_x = 1, dilation_rate_in_y = 1, padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0, data_layout = NCHW> : (tensor<1x27x36xf32>) -> tensor<1x3x8x8xf32>
  return
}

// -----

// CHECK-LABEL: func.func @stencil(
func.func @stencil(%src: tensor<1x1x8x8xf32>, %w: tensor<1x1x3x3xf32>) {
  // CHECK: %{{.+}} = mps.stencil %{{.+}}, %{{.+}} <reduction_mode = sum, offsets = [0, 0, 0, 0], strides = [1, 1, 1, 1], dilation_rates = [1, 1, 1, 1], explicit_padding = [0, 0, 0, 0, 1, 1, 1, 1], boundary_mode = zero, padding_style = explicit, padding_constant = 0.000000e+00> : (tensor<1x1x8x8xf32>, tensor<1x1x3x3xf32>) -> tensor<1x1x8x8xf32>
  %0 = mps.stencil %src, %w <reduction_mode = sum, offsets = [0, 0, 0, 0], strides = [1, 1, 1, 1], dilation_rates = [1, 1, 1, 1], explicit_padding = [0, 0, 0, 0, 1, 1, 1, 1], boundary_mode = zero, padding_style = explicit, padding_constant = 0.000000e+00> : (tensor<1x1x8x8xf32>, tensor<1x1x3x3xf32>) -> tensor<1x1x8x8xf32>
  return
}
