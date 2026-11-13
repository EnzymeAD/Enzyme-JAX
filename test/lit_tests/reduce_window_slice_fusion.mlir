// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// Exact #1859 reproducer: reduce_window(slice[0:511, 1:511], win=[2, 1]) + slice[2:512, 1:511]
// -> reduce_window(slice[0:512, 1:511], win=[3, 1])
// CHECK-LABEL: func.func @fuse_rw_and_trailing_slice(
// CHECK: %[[SL:.*]] = stablehlo.slice %arg0 [0:512, 1:511] : (tensor<512x512xf32>) -> tensor<512x510xf32>
// CHECK: %[[RW:.*]] = "stablehlo.reduce_window"(%[[SL]], %{{.*}}) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}>
// CHECK-NOT: stablehlo.slice
// CHECK-NOT: stablehlo.reduce_window
// CHECK: return %[[RW]] : tensor<510x510xf32>
func.func @fuse_rw_and_trailing_slice(%arg0: tensor<512x512xf32>) -> tensor<510x510xf32> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %s0 = stablehlo.slice %arg0 [0:511, 1:511] : (tensor<512x512xf32>) -> tensor<511x510xf32>
  %rw = "stablehlo.reduce_window"(%s0, %cst) <{window_dimensions = array<i64: 2, 1>}> ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    %sum = stablehlo.add %a, %b : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }) : (tensor<511x510xf32>, tensor<f32>) -> tensor<510x510xf32>
  %s1 = stablehlo.slice %arg0 [2:512, 1:511] : (tensor<512x512xf32>) -> tensor<510x510xf32>
  %res = stablehlo.add %rw, %s1 : tensor<510x510xf32>
  return %res : tensor<510x510xf32>
}

// Prepended slice + reduce_window with an extra non-window term in the chain,
// covering the entire base tensor so no slice op remains!
// CHECK-LABEL: func.func @fuse_leading_slice_and_rw_in_chain(
// CHECK-NOT: stablehlo.slice
// CHECK: %[[RW:.*]] = "stablehlo.reduce_window"(%arg0, %{{.*}}) <{base_dilations = array<i64: 1>, padding = dense<0> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 3>, window_strides = array<i64: 1>}>
// CHECK: %[[SUM:.*]] = stablehlo.add %[[RW]], %arg1 : tensor<10xf32>
// CHECK: return %[[SUM]] : tensor<10xf32>
func.func @fuse_leading_slice_and_rw_in_chain(%arg0: tensor<12xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %lead = stablehlo.slice %arg0 [0:10] : (tensor<12xf32>) -> tensor<10xf32>
  %tail_in = stablehlo.slice %arg0 [1:12] : (tensor<12xf32>) -> tensor<11xf32>
  %rw = "stablehlo.reduce_window"(%tail_in, %cst) <{window_dimensions = array<i64: 2>}> ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    %sum = stablehlo.add %a, %b : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }) : (tensor<11xf32>, tensor<f32>) -> tensor<10xf32>
  %t0 = stablehlo.add %lead, %arg1 : tensor<10xf32>
  %res = stablehlo.add %t0, %rw : tensor<10xf32>
  return %res : tensor<10xf32>
}

// Two 2D max reduce_windows adjacent along dim 0: win=[2, 2] + win=[2, 2] -> win=[4, 2]
// CHECK-LABEL: func.func @fuse_two_max_reduce_windows(
// CHECK-NOT: stablehlo.slice
// CHECK: %[[RW:.*]] = "stablehlo.reduce_window"(%arg0, %{{.*}}) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 4, 2>, window_strides = array<i64: 1, 1>}>
// CHECK-NOT: stablehlo.reduce_window
// CHECK: return %[[RW]] : tensor<5x7xf32>
func.func @fuse_two_max_reduce_windows(%arg0: tensor<8x8xf32>) -> tensor<5x7xf32> {
  %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %s0 = stablehlo.slice %arg0 [0:6, 0:8] : (tensor<8x8xf32>) -> tensor<6x8xf32>
  %rw0 = "stablehlo.reduce_window"(%s0, %init) <{window_dimensions = array<i64: 2, 2>}> ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    %m = stablehlo.maximum %a, %b : tensor<f32>
    stablehlo.return %m : tensor<f32>
  }) : (tensor<6x8xf32>, tensor<f32>) -> tensor<5x7xf32>
  %s1 = stablehlo.slice %arg0 [2:8, 0:8] : (tensor<8x8xf32>) -> tensor<6x8xf32>
  %rw1 = "stablehlo.reduce_window"(%s1, %init) <{window_dimensions = array<i64: 2, 2>}> ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    %m = stablehlo.maximum %a, %b : tensor<f32>
    stablehlo.return %m : tensor<f32>
  }) : (tensor<6x8xf32>, tensor<f32>) -> tensor<5x7xf32>
  %res = stablehlo.maximum %rw0, %rw1 : tensor<5x7xf32>
  return %res : tensor<5x7xf32>
}
