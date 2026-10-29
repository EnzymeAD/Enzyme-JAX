// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt --split-input-file | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=reduce_or_and_pad_window" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s --check-prefix=TD

// The guard is false past column 2, so the or over the columns only sees
// columns 0 and 1: every factor is cut to them.
func.func @or_window(%m: tensor<4xi1>, %g: tensor<4x2xi1>) -> tensor<4xi1> {
  %f = stablehlo.constant dense<false> : tensor<i1>
  %p = stablehlo.pad %g, %f, low = [0, 0], high = [0, 6], interior = [0, 0] : (tensor<4x2xi1>, tensor<i1>) -> tensor<4x8xi1>
  %b = stablehlo.broadcast_in_dim %m, dims = [0] : (tensor<4xi1>) -> tensor<4x8xi1>
  %a = stablehlo.and %b, %p : tensor<4x8xi1>
  %r = stablehlo.reduce(%a init: %f) applies stablehlo.or across dimensions = [1] : (tensor<4x8xi1>, tensor<i1>) -> tensor<4xi1>
  return %r : tensor<4xi1>
}

// CHECK:      func.func @or_window(%arg0: tensor<4xi1>, %arg1: tensor<4x2xi1>) -> tensor<4xi1> {
// CHECK-NEXT:   %c = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:   %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<4xi1>) -> tensor<4x2xi1>
// CHECK-NEXT:   %1 = stablehlo.and %0, %arg1 : tensor<4x2xi1>
// CHECK-NEXT:   %2 = stablehlo.reduce(%1 init: %c) applies stablehlo.or across dimensions = [1] : (tensor<4x2xi1>, tensor<i1>) -> tensor<4xi1>
// CHECK-NEXT:   return %2 : tensor<4xi1>
// CHECK-NEXT: }

// TD:      func.func @or_window(%arg0: tensor<4xi1>, %arg1: tensor<4x2xi1>) -> tensor<4xi1> {
// TD-NEXT:   %c = stablehlo.constant dense<false> : tensor<i1>
// TD-NEXT:   %0 = stablehlo.pad %arg1, %c, low = [0, 0], high = [0, 6], interior = [0, 0] : (tensor<4x2xi1>, tensor<i1>) -> tensor<4x8xi1>
// TD-NEXT:   %1 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<4xi1>) -> tensor<4x8xi1>
// TD-NEXT:   %2 = stablehlo.slice %1 [0:4, 0:2] : (tensor<4x8xi1>) -> tensor<4x2xi1>
// TD-NEXT:   %3 = stablehlo.slice %0 [0:4, 0:2] : (tensor<4x8xi1>) -> tensor<4x2xi1>
// TD-NEXT:   %4 = stablehlo.and %2, %3 : tensor<4x2xi1>
// TD-NEXT:   %5 = stablehlo.reduce(%4 init: %c) applies stablehlo.or across dimensions = [1] : (tensor<4x2xi1>, tensor<i1>) -> tensor<4xi1>
// TD-NEXT:   return %5 : tensor<4xi1>
// TD-NEXT: }

// -----

// The dual: an and over an or with a true pad.
func.func @and_window(%m: tensor<4xi1>, %g: tensor<4x2xi1>) -> tensor<4xi1> {
  %t = stablehlo.constant dense<true> : tensor<i1>
  %p = stablehlo.pad %g, %t, low = [0, 3], high = [0, 3], interior = [0, 0] : (tensor<4x2xi1>, tensor<i1>) -> tensor<4x8xi1>
  %b = stablehlo.broadcast_in_dim %m, dims = [0] : (tensor<4xi1>) -> tensor<4x8xi1>
  %o = stablehlo.or %b, %p : tensor<4x8xi1>
  %r = stablehlo.reduce(%o init: %t) applies stablehlo.and across dimensions = [1] : (tensor<4x8xi1>, tensor<i1>) -> tensor<4xi1>
  return %r : tensor<4xi1>
}

// CHECK:      func.func @and_window(%arg0: tensor<4xi1>, %arg1: tensor<4x2xi1>) -> tensor<4xi1> {
// CHECK-NEXT:   %c = stablehlo.constant dense<true> : tensor<i1>
// CHECK-NEXT:   %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<4xi1>) -> tensor<4x2xi1>
// CHECK-NEXT:   %1 = stablehlo.or %0, %arg1 : tensor<4x2xi1>
// CHECK-NEXT:   %2 = stablehlo.reduce(%1 init: %c) applies stablehlo.and across dimensions = [1] : (tensor<4x2xi1>, tensor<i1>) -> tensor<4xi1>
// CHECK-NEXT:   return %2 : tensor<4xi1>
// CHECK-NEXT: }

// TD:      func.func @and_window(%arg0: tensor<4xi1>, %arg1: tensor<4x2xi1>) -> tensor<4xi1> {
// TD-NEXT:   %c = stablehlo.constant dense<true> : tensor<i1>
// TD-NEXT:   %0 = stablehlo.pad %arg1, %c, low = [0, 3], high = [0, 3], interior = [0, 0] : (tensor<4x2xi1>, tensor<i1>) -> tensor<4x8xi1>
// TD-NEXT:   %1 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<4xi1>) -> tensor<4x8xi1>
// TD-NEXT:   %2 = stablehlo.slice %1 [0:4, 3:5] : (tensor<4x8xi1>) -> tensor<4x2xi1>
// TD-NEXT:   %3 = stablehlo.slice %0 [0:4, 3:5] : (tensor<4x8xi1>) -> tensor<4x2xi1>
// TD-NEXT:   %4 = stablehlo.or %2, %3 : tensor<4x2xi1>
// TD-NEXT:   %5 = stablehlo.reduce(%4 init: %c) applies stablehlo.and across dimensions = [1] : (tensor<4x2xi1>, tensor<i1>) -> tensor<4xi1>
// TD-NEXT:   return %5 : tensor<4xi1>
// TD-NEXT: }

// -----

// Padded only along a kept dimension: nothing to cut.
func.func @kept_dim(%m: tensor<8xi1>, %g: tensor<2x8xi1>) -> tensor<4xi1> {
  %f = stablehlo.constant dense<false> : tensor<i1>
  %p = stablehlo.pad %g, %f, low = [0, 0], high = [2, 0], interior = [0, 0] : (tensor<2x8xi1>, tensor<i1>) -> tensor<4x8xi1>
  %b = stablehlo.broadcast_in_dim %m, dims = [1] : (tensor<8xi1>) -> tensor<4x8xi1>
  %a = stablehlo.and %b, %p : tensor<4x8xi1>
  %r = stablehlo.reduce(%a init: %f) applies stablehlo.or across dimensions = [1] : (tensor<4x8xi1>, tensor<i1>) -> tensor<4xi1>
  return %r : tensor<4xi1>
}

// CHECK:      func.func @kept_dim(%arg0: tensor<8xi1>, %arg1: tensor<2x8xi1>) -> tensor<4xi1> {
// CHECK-NEXT:   %c = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:   %0 = stablehlo.pad %arg1, %c, low = [0, 0], high = [2, 0], interior = [0, 0] : (tensor<2x8xi1>, tensor<i1>) -> tensor<4x8xi1>
// CHECK-NEXT:   %1 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<8xi1>) -> tensor<4x8xi1>
// CHECK-NEXT:   %2 = stablehlo.and %1, %0 : tensor<4x8xi1>
// CHECK-NEXT:   %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.or across dimensions = [1] : (tensor<4x8xi1>, tensor<i1>) -> tensor<4xi1>
// CHECK-NEXT:   return %3 : tensor<4xi1>
// CHECK-NEXT: }

// TD:      func.func @kept_dim(%arg0: tensor<8xi1>, %arg1: tensor<2x8xi1>) -> tensor<4xi1> {
// TD-NEXT:   %c = stablehlo.constant dense<false> : tensor<i1>
// TD-NEXT:   %0 = stablehlo.pad %arg1, %c, low = [0, 0], high = [2, 0], interior = [0, 0] : (tensor<2x8xi1>, tensor<i1>) -> tensor<4x8xi1>
// TD-NEXT:   %1 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<8xi1>) -> tensor<4x8xi1>
// TD-NEXT:   %2 = stablehlo.and %1, %0 : tensor<4x8xi1>
// TD-NEXT:   %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.or across dimensions = [1] : (tensor<4x8xi1>, tensor<i1>) -> tensor<4xi1>
// TD-NEXT:   return %3 : tensor<4xi1>
// TD-NEXT: }

// -----

// The raiser's lane guard: a shared-memory condition over one z lane padded
// to the lane bound, and-ed with the x/y lane masks and or-reduced over z.
// The reduce over the single lane left is a reshape.
func.func @lane_guard(%x: tensor<1024xi1>, %y: tensor<1024xi1>, %g: tensor<1024x1024x1xi1>) -> tensor<1024x1024xi1> {
  %f = stablehlo.constant dense<false> : tensor<i1>
  %p = stablehlo.pad %g, %f, low = [0, 0, 0], high = [0, 0, 1023], interior = [0, 0, 0] : (tensor<1024x1024x1xi1>, tensor<i1>) -> tensor<1024x1024x1024xi1>
  %bx = stablehlo.broadcast_in_dim %x, dims = [0] : (tensor<1024xi1>) -> tensor<1024x1024x1024xi1>
  %by = stablehlo.broadcast_in_dim %y, dims = [1] : (tensor<1024xi1>) -> tensor<1024x1024x1024xi1>
  %a0 = stablehlo.and %bx, %by : tensor<1024x1024x1024xi1>
  %a = stablehlo.and %a0, %p : tensor<1024x1024x1024xi1>
  %r = stablehlo.reduce(%a init: %f) applies stablehlo.or across dimensions = [2] : (tensor<1024x1024x1024xi1>, tensor<i1>) -> tensor<1024x1024xi1>
  return %r : tensor<1024x1024xi1>
}

// CHECK:      func.func @lane_guard(%arg0: tensor<1024xi1>, %arg1: tensor<1024xi1>, %arg2: tensor<1024x1024x1xi1>) -> tensor<1024x1024xi1> {
// CHECK-NEXT:   %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1024xi1>) -> tensor<1024x1024x1xi1>
// CHECK-NEXT:   %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xi1>) -> tensor<1024x1024x1xi1>
// CHECK-NEXT:   %2 = stablehlo.and %0, %1 : tensor<1024x1024x1xi1>
// CHECK-NEXT:   %3 = stablehlo.and %2, %arg2 : tensor<1024x1024x1xi1>
// CHECK-NEXT:   %4 = stablehlo.reshape %3 : (tensor<1024x1024x1xi1>) -> tensor<1024x1024xi1>
// CHECK-NEXT:   return %4 : tensor<1024x1024xi1>
// CHECK-NEXT: }

// TD:      func.func @lane_guard(%arg0: tensor<1024xi1>, %arg1: tensor<1024xi1>, %arg2: tensor<1024x1024x1xi1>) -> tensor<1024x1024xi1> {
// TD-NEXT:   %c = stablehlo.constant dense<false> : tensor<i1>
// TD-NEXT:   %0 = stablehlo.pad %arg2, %c, low = [0, 0, 0], high = [0, 0, 1023], interior = [0, 0, 0] : (tensor<1024x1024x1xi1>, tensor<i1>) -> tensor<1024x1024x1024xi1>
// TD-NEXT:   %1 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1024xi1>) -> tensor<1024x1024x1024xi1>
// TD-NEXT:   %2 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xi1>) -> tensor<1024x1024x1024xi1>
// TD-NEXT:   %3 = stablehlo.slice %1 [0:1024, 0:1024, 0:1] : (tensor<1024x1024x1024xi1>) -> tensor<1024x1024x1xi1>
// TD-NEXT:   %4 = stablehlo.slice %2 [0:1024, 0:1024, 0:1] : (tensor<1024x1024x1024xi1>) -> tensor<1024x1024x1xi1>
// TD-NEXT:   %5 = stablehlo.and %3, %4 : tensor<1024x1024x1xi1>
// TD-NEXT:   %6 = stablehlo.slice %0 [0:1024, 0:1024, 0:1] : (tensor<1024x1024x1024xi1>) -> tensor<1024x1024x1xi1>
// TD-NEXT:   %7 = stablehlo.and %5, %6 : tensor<1024x1024x1xi1>
// TD-NEXT:   %8 = stablehlo.reduce(%7 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<1024x1024x1xi1>, tensor<i1>) -> tensor<1024x1024xi1>
// TD-NEXT:   return %8 : tensor<1024x1024xi1>
// TD-NEXT: }
