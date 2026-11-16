// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt --split-input-file | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=reduce_or_and_pad" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s --check-prefix=TD
// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=reduce_or_and_pad" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file %s | FileCheck %s --check-prefix=TD

// The raiser's lane guard at scale: an all-true row padded with false over
// 256 lanes, or-reduced over the lanes. Too large to constant-fold, it is
// still true everywhere: the operand row is true and the columns are
// unpadded.
func.func @lane0_large(%a: tensor<4096xf64>, %b: tensor<4096xf64>) -> tensor<4096xf64> {
  %c = stablehlo.constant dense<true> : tensor<1x4096xi1>
  %f = stablehlo.constant dense<false> : tensor<i1>
  %m = stablehlo.pad %c, %f, low = [0, 0], high = [255, 0], interior = [0, 0] : (tensor<1x4096xi1>, tensor<i1>) -> tensor<256x4096xi1>
  %r = stablehlo.reduce(%m init: %f) applies stablehlo.or across dimensions = [0] : (tensor<256x4096xi1>, tensor<i1>) -> tensor<4096xi1>
  %s = stablehlo.select %r, %a, %b : tensor<4096xi1>, tensor<4096xf64>
  return %s : tensor<4096xf64>
}

// CHECK:    func.func @lane0_large(%arg0: tensor<4096xf64>, %arg1: tensor<4096xf64>) -> tensor<4096xf64> {
// CHECK-NEXT:    return %arg0 : tensor<4096xf64>
// CHECK-NEXT:  }

// TD:  module {
// TD-NEXT:  func.func @lane0_large(%arg0: tensor<4096xf64>, %arg1: tensor<4096xf64>) -> tensor<4096xf64> {
// TD-NEXT:    %c = stablehlo.constant dense<true> : tensor<4096xi1>
// TD-NEXT:    %0 = stablehlo.select %c, %arg0, %arg1 : tensor<4096xi1>, tensor<4096xf64>
// TD-NEXT:    return %0 : tensor<4096xf64>
// TD-NEXT:  }

// -----

// Padding with true along the reduced axis makes the or true whatever the
// operand holds; an and over a pad with false padding is false likewise.
func.func @padvalue(%m0: tensor<5x4096xi1>, %m1: tensor<5x4096xi1>) -> (tensor<4096xi1>, tensor<4096xi1>) {
  %t = stablehlo.constant dense<true> : tensor<i1>
  %f = stablehlo.constant dense<false> : tensor<i1>
  %mo = stablehlo.pad %m0, %t, low = [2, 0], high = [249, 0], interior = [0, 0] : (tensor<5x4096xi1>, tensor<i1>) -> tensor<256x4096xi1>
  %r0 = stablehlo.reduce(%mo init: %f) applies stablehlo.or across dimensions = [0] : (tensor<256x4096xi1>, tensor<i1>) -> tensor<4096xi1>
  %ma = stablehlo.pad %m1, %f, low = [0, 0], high = [251, 0], interior = [0, 0] : (tensor<5x4096xi1>, tensor<i1>) -> tensor<256x4096xi1>
  %r1 = stablehlo.reduce(%ma init: %t) applies stablehlo.and across dimensions = [0] : (tensor<256x4096xi1>, tensor<i1>) -> tensor<4096xi1>
  return %r0, %r1 : tensor<4096xi1>, tensor<4096xi1>
}

// CHECK:    func.func @padvalue(%arg0: tensor<5x4096xi1>, %arg1: tensor<5x4096xi1>) -> (tensor<4096xi1>, tensor<4096xi1>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<4096xi1>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<false> : tensor<4096xi1>
// CHECK-NEXT:    return %c, %c_0 : tensor<4096xi1>, tensor<4096xi1>
// CHECK-NEXT:  }

// TD:  module {
// TD-NEXT:  func.func @padvalue(%arg0: tensor<5x4096xi1>, %arg1: tensor<5x4096xi1>) -> (tensor<4096xi1>, tensor<4096xi1>) {
// TD-NEXT:    %c = stablehlo.constant dense<true> : tensor<4096xi1>
// TD-NEXT:    %c_0 = stablehlo.constant dense<false> : tensor<4096xi1>
// TD-NEXT:    return %c, %c_0 : tensor<4096xi1>, tensor<4096xi1>
// TD-NEXT:  }

// -----

// An all-true operand padded with false along a kept axis: the padded
// columns hold no true, so the or stays.
func.func @falsecols(%a: tensor<4096xf64>, %b: tensor<4096xf64>) -> tensor<4096xf64> {
  %c = stablehlo.constant dense<true> : tensor<1x4000xi1>
  %f = stablehlo.constant dense<false> : tensor<i1>
  %m = stablehlo.pad %c, %f, low = [0, 48], high = [255, 48], interior = [0, 0] : (tensor<1x4000xi1>, tensor<i1>) -> tensor<256x4096xi1>
  %r = stablehlo.reduce(%m init: %f) applies stablehlo.or across dimensions = [0] : (tensor<256x4096xi1>, tensor<i1>) -> tensor<4096xi1>
  %s = stablehlo.select %r, %a, %b : tensor<4096xi1>, tensor<4096xf64>
  return %s : tensor<4096xf64>
}

// CHECK:    func.func @falsecols(%arg0: tensor<4096xf64>, %arg1: tensor<4096xf64>) -> tensor<4096xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<1x4000xi1>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:    %0 = stablehlo.pad %c, %c_0, low = [0, 48], high = [255, 48], interior = [0, 0] : (tensor<1x4000xi1>, tensor<i1>) -> tensor<256x4096xi1>
// CHECK-NEXT:    %1 = stablehlo.reduce(%0 init: %c_0) applies stablehlo.or across dimensions = [0] : (tensor<256x4096xi1>, tensor<i1>) -> tensor<4096xi1>
// CHECK-NEXT:    %2 = stablehlo.select %1, %arg0, %arg1 : tensor<4096xi1>, tensor<4096xf64>
// CHECK-NEXT:    return %2 : tensor<4096xf64>
// CHECK-NEXT:  }

// TD:  module {
// TD-NEXT:  func.func @falsecols(%arg0: tensor<4096xf64>, %arg1: tensor<4096xf64>) -> tensor<4096xf64> {
// TD-NEXT:    %c = stablehlo.constant dense<true> : tensor<1x4000xi1>
// TD-NEXT:    %c_0 = stablehlo.constant dense<false> : tensor<i1>
// TD-NEXT:    %0 = stablehlo.pad %c, %c_0, low = [0, 48], high = [255, 48], interior = [0, 0] : (tensor<1x4000xi1>, tensor<i1>) -> tensor<256x4096xi1>
// TD-NEXT:    %1 = stablehlo.reduce(%0 init: %c_0) applies stablehlo.or across dimensions = [0] : (tensor<256x4096xi1>, tensor<i1>) -> tensor<4096xi1>
// TD-NEXT:    %2 = stablehlo.select %1, %arg0, %arg1 : tensor<4096xi1>, tensor<4096xf64>
// TD-NEXT:    return %2 : tensor<4096xf64>
// TD-NEXT:  }
