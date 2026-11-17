// RUN: enzymexlamlir-opt --split-input-file %s | enzymexlamlir-opt --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @reshape(
func.func @reshape(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.reshape %{{.+}} : (tensor<2x3xf32>) -> tensor<6xf32>
  %0 = mps.reshape %x : (tensor<2x3xf32>) -> tensor<6xf32>
  return
}

// -----

// CHECK-LABEL: func.func @transpose(
func.func @transpose(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.transpose %{{.+}} <permutation = [1, 0]> : (tensor<2x3xf32>) -> tensor<3x2xf32>
  %0 = mps.transpose %x <permutation = [1, 0]> : (tensor<2x3xf32>) -> tensor<3x2xf32>
  return
}

// -----

// CHECK-LABEL: func.func @flatten_2d(
func.func @flatten_2d(%y: tensor<2x3x4xf32>) {
  // CHECK: %{{.+}} = mps.flatten_2d %{{.+}} <axis = 1> : (tensor<2x3x4xf32>) -> tensor<2x12xf32>
  %0 = mps.flatten_2d %y <axis = 1> : (tensor<2x3x4xf32>) -> tensor<2x12xf32>
  return
}

// -----

// CHECK-LABEL: func.func @broadcast(
func.func @broadcast(%v: tensor<3xf32>) {
  // CHECK: %{{.+}} = mps.broadcast %{{.+}} : (tensor<3xf32>) -> tensor<2x3xf32>
  %0 = mps.broadcast %v : (tensor<3xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @squeeze(
func.func @squeeze(%u: tensor<1x3xf32>) {
  // CHECK: %{{.+}} = mps.squeeze %{{.+}} <axes = [0]> : (tensor<1x3xf32>) -> tensor<3xf32>
  %0 = mps.squeeze %u <axes = [0]> : (tensor<1x3xf32>) -> tensor<3xf32>
  // CHECK: %{{.+}} = mps.squeeze %{{.+}} : (tensor<1x3xf32>) -> tensor<3xf32>
  %1 = mps.squeeze %u : (tensor<1x3xf32>) -> tensor<3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @expand_dims(
func.func @expand_dims(%v: tensor<3xf32>) {
  // CHECK: %{{.+}} = mps.expand_dims %{{.+}} <axes = [0]> : (tensor<3xf32>) -> tensor<1x3xf32>
  %0 = mps.expand_dims %v <axes = [0]> : (tensor<3xf32>) -> tensor<1x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @reverse(
func.func @reverse(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.reverse %{{.+}} <axes = [1]> : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %0 = mps.reverse %x <axes = [1]> : (tensor<2x3xf32>) -> tensor<2x3xf32>
  // CHECK: %{{.+}} = mps.reverse %{{.+}} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %1 = mps.reverse %x : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @shape_of(
func.func @shape_of(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.shape_of %{{.+}} : (tensor<2x3xf32>) -> tensor<2xi32>
  %0 = mps.shape_of %x : (tensor<2x3xf32>) -> tensor<2xi32>
  return
}

// -----

// CHECK-LABEL: func.func @cast(
func.func @cast(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.cast %{{.+}} : (tensor<2x3xf32>) -> tensor<2x3xf16>
  %0 = mps.cast %x : (tensor<2x3xf32>) -> tensor<2x3xf16>
  return
}

// -----

// CHECK-LABEL: func.func @reinterpret_cast(
func.func @reinterpret_cast(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.reinterpret_cast %{{.+}} : (tensor<2x3xf32>) -> tensor<2x3xi32>
  %0 = mps.reinterpret_cast %x : (tensor<2x3xf32>) -> tensor<2x3xi32>
  return
}

// -----

// CHECK-LABEL: func.func @coordinate(
func.func @coordinate() {
  // CHECK: %{{.+}} = mps.coordinate  <axis = 1> : () -> tensor<2x3xi32>
  %0 = mps.coordinate <axis = 1> : () -> tensor<2x3xi32>
  return
}

// -----

// CHECK-LABEL: func.func @slice(
func.func @slice(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.slice %{{.+}} <starts = [0, 1], ends = [2, 3], strides = [1, 1]> : (tensor<2x3xf32>) -> tensor<2x2xf32>
  %0 = mps.slice %x <starts = [0, 1], ends = [2, 3], strides = [1, 1]> : (tensor<2x3xf32>) -> tensor<2x2xf32>
  // CHECK: %{{.+}} = mps.slice %{{.+}} <starts = [0, 0], ends = [1, 3], strides = [1, 1], start_mask = 2, end_mask = 2, squeeze_mask = 1> : (tensor<2x3xf32>) -> tensor<3xf32>
  %1 = mps.slice %x <starts = [0, 0], ends = [1, 3], strides = [1, 1], start_mask = 2, end_mask = 2, squeeze_mask = 1> : (tensor<2x3xf32>) -> tensor<3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @dynamic_slice(
func.func @dynamic_slice(%x: tensor<2x3xf32>, %s: tensor<2xi32>, %e: tensor<2xi32>, %st: tensor<2xi32>) {
  // CHECK: %{{.+}} = mps.dynamic_slice %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : (tensor<2x3xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x2xf32>
  %0 = mps.dynamic_slice %x, %s, %e, %st : (tensor<2x3xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x2xf32>
  // CHECK: %{{.+}} = mps.dynamic_slice %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} <start_mask = 1> : (tensor<2x3xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x2xf32>
  %1 = mps.dynamic_slice %x, %s, %e, %st <start_mask = 1> : (tensor<2x3xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func.func @slice_gradient(
func.func @slice_gradient(%u: tensor<2x2xf32>) {
  // CHECK: %{{.+}} = mps.slice_gradient %{{.+}} <starts = [0, 1], ends = [2, 3], strides = [1, 1]> : (tensor<2x2xf32>) -> tensor<2x3xf32>
  %0 = mps.slice_gradient %u <starts = [0, 1], ends = [2, 3], strides = [1, 1]> : (tensor<2x2xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @dynamic_slice_gradient(
func.func @dynamic_slice_gradient(%u: tensor<2x2xf32>, %s: tensor<2xi32>, %e: tensor<2xi32>, %st: tensor<2xi32>) {
  // CHECK: %{{.+}} = mps.dynamic_slice_gradient %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : (tensor<2x2xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x3xf32>
  %0 = mps.dynamic_slice_gradient %u, %s, %e, %st : (tensor<2x2xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @slice_update(
func.func @slice_update(%x: tensor<2x3xf32>, %u: tensor<2x2xf32>) {
  // CHECK: %{{.+}} = mps.slice_update %{{.+}}, %{{.+}} <starts = [0, 1], ends = [2, 3], strides = [1, 1]> : (tensor<2x3xf32>, tensor<2x2xf32>) -> tensor<2x3xf32>
  %0 = mps.slice_update %x, %u <starts = [0, 1], ends = [2, 3], strides = [1, 1]> : (tensor<2x3xf32>, tensor<2x2xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @dynamic_slice_update(
func.func @dynamic_slice_update(%x: tensor<2x3xf32>, %u: tensor<2x2xf32>, %s: tensor<2xi32>, %e: tensor<2xi32>, %st: tensor<2xi32>) {
  // CHECK: %{{.+}} = mps.dynamic_slice_update %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} <end_mask = 1> : (tensor<2x3xf32>, tensor<2x2xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x3xf32>
  %0 = mps.dynamic_slice_update %x, %u, %s, %e, %st <end_mask = 1> : (tensor<2x3xf32>, tensor<2x2xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @concat(
func.func @concat(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.concat %{{.+}}, %{{.+}} <dimension = 0> : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<4x3xf32>
  %0 = mps.concat %x, %x <dimension = 0> : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<4x3xf32>
  // CHECK: %{{.+}} = mps.concat %{{.+}}, %{{.+}} <dimension = 1, interleave = true> : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x6xf32>
  %1 = mps.concat %x, %x <dimension = 1, interleave = true> : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x6xf32>
  return
}

// -----

// CHECK-LABEL: func.func @split(
func.func @split(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.split %{{.+}} <axis = 1> : (tensor<2x3xf32>) -> (tensor<2x1xf32>, tensor<2x2xf32>)
  %0:2 = mps.split %x <axis = 1> : (tensor<2x3xf32>) -> (tensor<2x1xf32>, tensor<2x2xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @stack(
func.func @stack(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.stack %{{.+}}, %{{.+}} <axis = 0> : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x2x3xf32>
  %0 = mps.stack %x, %x <axis = 0> : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @tile(
func.func @tile(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.tile %{{.+}} <multiplier = [2, 1]> : (tensor<2x3xf32>) -> tensor<4x3xf32>
  %0 = mps.tile %x <multiplier = [2, 1]> : (tensor<2x3xf32>) -> tensor<4x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @tile_gradient(
func.func @tile_gradient(%gt: tensor<4x3xf32>, %x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.tile_gradient %{{.+}}, %{{.+}} <multiplier = [2, 1]> : (tensor<4x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %0 = mps.tile_gradient %gt, %x <multiplier = [2, 1]> : (tensor<4x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @pad(
func.func @pad(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.pad %{{.+}} <padding_mode = constant, left_padding = [1, 0], right_padding = [1, 2], constant_value = 0.000000e+00> : (tensor<2x3xf32>) -> tensor<4x5xf32>
  %0 = mps.pad %x <padding_mode = constant, left_padding = [1, 0], right_padding = [1, 2], constant_value = 0.000000e+00> : (tensor<2x3xf32>) -> tensor<4x5xf32>
  return
}

// -----

// CHECK-LABEL: func.func @pad_gradient(
func.func @pad_gradient(%gp: tensor<4x5xf32>, %x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.pad_gradient %{{.+}}, %{{.+}} <padding_mode = reflect, left_padding = [1, 0], right_padding = [1, 2]> : (tensor<4x5xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %0 = mps.pad_gradient %gp, %x <padding_mode = reflect, left_padding = [1, 0], right_padding = [1, 2]> : (tensor<4x5xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @space_to_depth_2d(
func.func @space_to_depth_2d(%x: tensor<1x4x4x3xf32>) {
  // CHECK: %{{.+}} = mps.space_to_depth_2d %{{.+}} <width_axis = 2, height_axis = 1, depth_axis = 3, block_size = 2, use_pixel_shuffle_order = false> : (tensor<1x4x4x3xf32>) -> tensor<1x2x2x12xf32>
  %0 = mps.space_to_depth_2d %x <width_axis = 2, height_axis = 1, depth_axis = 3, block_size = 2, use_pixel_shuffle_order = false> : (tensor<1x4x4x3xf32>) -> tensor<1x2x2x12xf32>
  return
}

// -----

// CHECK-LABEL: func.func @depth_to_space_2d(
func.func @depth_to_space_2d(%d: tensor<1x2x2x12xf32>) {
  // CHECK: %{{.+}} = mps.depth_to_space_2d %{{.+}} <width_axis = 2, height_axis = 1, depth_axis = 3, block_size = 2, use_pixel_shuffle_order = true> : (tensor<1x2x2x12xf32>) -> tensor<1x4x4x3xf32>
  %0 = mps.depth_to_space_2d %d <width_axis = 2, height_axis = 1, depth_axis = 3, block_size = 2, use_pixel_shuffle_order = true> : (tensor<1x2x2x12xf32>) -> tensor<1x4x4x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @space_to_batch(
func.func @space_to_batch(%x: tensor<1x4x4x3xf32>) {
  // CHECK: %{{.+}} = mps.space_to_batch %{{.+}} <spatial_axes = [1, 2], batch_axis = 0, block_dimensions = [2, 2], use_pixel_shuffle_order = false> : (tensor<1x4x4x3xf32>) -> tensor<4x2x2x3xf32>
  %0 = mps.space_to_batch %x <spatial_axes = [1, 2], batch_axis = 0, block_dimensions = [2, 2], use_pixel_shuffle_order = false> : (tensor<1x4x4x3xf32>) -> tensor<4x2x2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @batch_to_space(
func.func @batch_to_space(%b: tensor<4x2x2x3xf32>) {
  // CHECK: %{{.+}} = mps.batch_to_space %{{.+}} <spatial_axes = [1, 2], batch_axis = 0, block_dimensions = [2, 2], use_pixel_shuffle_order = false> : (tensor<4x2x2x3xf32>) -> tensor<1x4x4x3xf32>
  %0 = mps.batch_to_space %b <spatial_axes = [1, 2], batch_axis = 0, block_dimensions = [2, 2], use_pixel_shuffle_order = false> : (tensor<4x2x2x3xf32>) -> tensor<1x4x4x3xf32>
  return
}
