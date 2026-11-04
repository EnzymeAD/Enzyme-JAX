// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=and_or_pad_intersect" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// The x guard padded along rows and the y guard padded along columns: their
// and is false outside the [0:4, 0:4] corner, so it is the and over that
// corner padded back out.
func.func @and_corner(%a: tensor<4x1024xi1>, %b: tensor<1024x4xi1>) -> tensor<1024x1024xi1> {
  %f = stablehlo.constant dense<false> : tensor<i1>
  %pa = stablehlo.pad %a, %f, low = [0, 0], high = [1020, 0], interior = [0, 0] : (tensor<4x1024xi1>, tensor<i1>) -> tensor<1024x1024xi1>
  %pb = stablehlo.pad %b, %f, low = [0, 0], high = [0, 1020], interior = [0, 0] : (tensor<1024x4xi1>, tensor<i1>) -> tensor<1024x1024xi1>
  %r = stablehlo.and %pa, %pb : tensor<1024x1024xi1>
  return %r : tensor<1024x1024xi1>
}

// CHECK:      func.func @and_corner(%arg0: tensor<4x1024xi1>, %arg1: tensor<1024x4xi1>) -> tensor<1024x1024xi1> {
// CHECK-NEXT:   %c = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [0:4, 0:4] : (tensor<4x1024xi1>) -> tensor<4x4xi1>
// CHECK-NEXT:   %1 = stablehlo.slice %arg1 [0:4, 0:4] : (tensor<1024x4xi1>) -> tensor<4x4xi1>
// CHECK-NEXT:   %2 = stablehlo.and %0, %1 : tensor<4x4xi1>
// CHECK-NEXT:   %3 = stablehlo.pad %2, %c, low = [0, 0], high = [1020, 1020], interior = [0, 0] : (tensor<4x4xi1>, tensor<i1>) -> tensor<1024x1024xi1>
// CHECK-NEXT:   return %3 : tensor<1024x1024xi1>
// CHECK-NEXT: }

// -----


// The dual: or of pads with true, over overlapping windows that start at
// different offsets.
func.func @or_overlap(%a: tensor<6x8xi1>, %b: tensor<5x8xi1>) -> tensor<10x8xi1> {
  %t = stablehlo.constant dense<true> : tensor<i1>
  %pa = stablehlo.pad %a, %t, low = [0, 0], high = [4, 0], interior = [0, 0] : (tensor<6x8xi1>, tensor<i1>) -> tensor<10x8xi1>
  %pb = stablehlo.pad %b, %t, low = [3, 0], high = [2, 0], interior = [0, 0] : (tensor<5x8xi1>, tensor<i1>) -> tensor<10x8xi1>
  %r = stablehlo.or %pa, %pb : tensor<10x8xi1>
  return %r : tensor<10x8xi1>
}

// CHECK:      func.func @or_overlap(%arg0: tensor<6x8xi1>, %arg1: tensor<5x8xi1>) -> tensor<10x8xi1> {
// CHECK-NEXT:   %c = stablehlo.constant dense<true> : tensor<i1>
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [3:6, 0:8] : (tensor<6x8xi1>) -> tensor<3x8xi1>
// CHECK-NEXT:   %1 = stablehlo.slice %arg1 [0:3, 0:8] : (tensor<5x8xi1>) -> tensor<3x8xi1>
// CHECK-NEXT:   %2 = stablehlo.or %0, %1 : tensor<3x8xi1>
// CHECK-NEXT:   %3 = stablehlo.pad %2, %c, low = [3, 0], high = [4, 0], interior = [0, 0] : (tensor<3x8xi1>, tensor<i1>) -> tensor<10x8xi1>
// CHECK-NEXT:   return %3 : tensor<10x8xi1>
// CHECK-NEXT: }

// -----


// Windows that do not meet: the and is false everywhere.
func.func @and_disjoint(%a: tensor<3x8xi1>, %b: tensor<3x8xi1>) -> tensor<10x8xi1> {
  %f = stablehlo.constant dense<false> : tensor<i1>
  %pa = stablehlo.pad %a, %f, low = [0, 0], high = [7, 0], interior = [0, 0] : (tensor<3x8xi1>, tensor<i1>) -> tensor<10x8xi1>
  %pb = stablehlo.pad %b, %f, low = [5, 0], high = [2, 0], interior = [0, 0] : (tensor<3x8xi1>, tensor<i1>) -> tensor<10x8xi1>
  %r = stablehlo.and %pa, %pb : tensor<10x8xi1>
  return %r : tensor<10x8xi1>
}

// CHECK:      func.func @and_disjoint(%arg0: tensor<3x8xi1>, %arg1: tensor<3x8xi1>) -> tensor<10x8xi1> {
// CHECK-NEXT:   %c = stablehlo.constant dense<false> : tensor<10x8xi1>
// CHECK-NEXT:   return %c : tensor<10x8xi1>
// CHECK-NEXT: }

// -----


// A pad with true is not absorbing for and: left alone.
func.func @and_true_pad(%a: tensor<4x1024xi1>, %b: tensor<1024x4xi1>) -> tensor<1024x1024xi1> {
  %f = stablehlo.constant dense<false> : tensor<i1>
  %t = stablehlo.constant dense<true> : tensor<i1>
  %pa = stablehlo.pad %a, %f, low = [0, 0], high = [1020, 0], interior = [0, 0] : (tensor<4x1024xi1>, tensor<i1>) -> tensor<1024x1024xi1>
  %pb = stablehlo.pad %b, %t, low = [0, 0], high = [0, 1020], interior = [0, 0] : (tensor<1024x4xi1>, tensor<i1>) -> tensor<1024x1024xi1>
  %r = stablehlo.and %pa, %pb : tensor<1024x1024xi1>
  return %r : tensor<1024x1024xi1>
}

// CHECK:      func.func @and_true_pad(%arg0: tensor<4x1024xi1>, %arg1: tensor<1024x4xi1>) -> tensor<1024x1024xi1> {
// CHECK-NEXT:   %c = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<true> : tensor<i1>
// CHECK-NEXT:   %0 = stablehlo.pad %arg0, %c, low = [0, 0], high = [1020, 0], interior = [0, 0] : (tensor<4x1024xi1>, tensor<i1>) -> tensor<1024x1024xi1>
// CHECK-NEXT:   %1 = stablehlo.pad %arg1, %c_0, low = [0, 0], high = [0, 1020], interior = [0, 0] : (tensor<1024x4xi1>, tensor<i1>) -> tensor<1024x1024xi1>
// CHECK-NEXT:   %2 = stablehlo.and %0, %1 : tensor<1024x1024xi1>
// CHECK-NEXT:   return %2 : tensor<1024x1024xi1>
// CHECK-NEXT: }
