// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt --split-input-file | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=one_hot_masked_reduce" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s --check-prefix=TD
// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=one_hot_masked_reduce" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file %s | FileCheck %s --check-prefix=TD

// The raiser's lane-0 pick at scale: a (value, mask) reduce over the lane
// axis whose mask is a pad of an all-true row, i.e. one-hot at lane 0. The
// pick is the lane-0 slice of the values, folded through the transpose.
func.func @lane0_large(%arg1: tensor<1048576xf64>) -> tensor<4096xf64> {
  %c = stablehlo.constant dense<true> : tensor<1x4096xi1>
  %c_0 = stablehlo.constant dense<false> : tensor<i1>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = stablehlo.reshape %arg1 : (tensor<1048576xf64>) -> tensor<4096x256xf64>
  %1 = stablehlo.pad %c, %c_0, low = [0, 0], high = [255, 0], interior = [0, 0] : (tensor<1x4096xi1>, tensor<i1>) -> tensor<256x4096xi1>
  %2 = stablehlo.transpose %0, dims = [1, 0] : (tensor<4096x256xf64>) -> tensor<256x4096xf64>
  %3:2 = stablehlo.reduce(%2 init: %cst), (%1 init: %c_0) across dimensions = [0] : (tensor<256x4096xf64>, tensor<256x4096xi1>, tensor<f64>, tensor<i1>) -> (tensor<4096xf64>, tensor<4096xi1>)
   reducer(%arg2: tensor<f64>, %arg4: tensor<f64>) (%arg3: tensor<i1>, %arg5: tensor<i1>)  {
    %4 = stablehlo.select %arg3, %arg2, %arg4 : tensor<i1>, tensor<f64>
    %5 = stablehlo.or %arg3, %arg5 : tensor<i1>
    stablehlo.return %4, %5 : tensor<f64>, tensor<i1>
  }
  return %3#0 : tensor<4096xf64>
}

// CHECK:    func.func @lane0_large(%arg0: tensor<1048576xf64>) -> tensor<4096xf64> {
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<1048576xf64>) -> tensor<4096x256xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %0 [0:4096, 0:1] : (tensor<4096x256xf64>) -> tensor<4096x1xf64>
// CHECK-NEXT:    %2 = stablehlo.reshape %1 : (tensor<4096x1xf64>) -> tensor<4096xf64>
// CHECK-NEXT:    return %2 : tensor<4096xf64>
// CHECK-NEXT:  }

// TD:  module {
// TD-NEXT:  func.func @lane0_large(%arg0: tensor<1048576xf64>) -> tensor<4096xf64> {
// TD-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<1048576xf64>) -> tensor<4096x256xf64>
// TD-NEXT:    %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<4096x256xf64>) -> tensor<256x4096xf64>
// TD-NEXT:    %2 = stablehlo.slice %1 [0:1, 0:4096] : (tensor<256x4096xf64>) -> tensor<1x4096xf64>
// TD-NEXT:    %3 = stablehlo.reshape %2 : (tensor<1x4096xf64>) -> tensor<4096xf64>
// TD-NEXT:    return %3 : tensor<4096xf64>
// TD-NEXT:  }

// -----

// A lane other than 0 (pad low = 3), and the mask result feeding a select.
func.func @lane3(%v: tensor<8x16xf64>, %a: tensor<16xf64>, %b: tensor<16xf64>) -> (tensor<16xf64>, tensor<16xf64>) {
  %c = stablehlo.constant dense<true> : tensor<1x16xi1>
  %f = stablehlo.constant dense<false> : tensor<i1>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %m = stablehlo.pad %c, %f, low = [3, 0], high = [4, 0], interior = [0, 0] : (tensor<1x16xi1>, tensor<i1>) -> tensor<8x16xi1>
  %r:2 = stablehlo.reduce(%v init: %cst), (%m init: %f) across dimensions = [0] : (tensor<8x16xf64>, tensor<8x16xi1>, tensor<f64>, tensor<i1>) -> (tensor<16xf64>, tensor<16xi1>)
   reducer(%arg2: tensor<f64>, %arg4: tensor<f64>) (%arg3: tensor<i1>, %arg5: tensor<i1>)  {
    %4 = stablehlo.select %arg3, %arg2, %arg4 : tensor<i1>, tensor<f64>
    %5 = stablehlo.or %arg3, %arg5 : tensor<i1>
    stablehlo.return %4, %5 : tensor<f64>, tensor<i1>
  }
  %s = stablehlo.select %r#1, %a, %b : tensor<16xi1>, tensor<16xf64>
  return %r#0, %s : tensor<16xf64>, tensor<16xf64>
}

// CHECK:    func.func @lane3(%arg0: tensor<8x16xf64>, %arg1: tensor<16xf64>, %arg2: tensor<16xf64>) -> (tensor<16xf64>, tensor<16xf64>) {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [3:4, 0:16] : (tensor<8x16xf64>) -> tensor<1x16xf64>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<1x16xf64>) -> tensor<16xf64>
// CHECK-NEXT:    return %1, %arg1 : tensor<16xf64>, tensor<16xf64>
// CHECK-NEXT:  }

// TD:  module {
// TD-NEXT:  func.func @lane3(%arg0: tensor<8x16xf64>, %arg1: tensor<16xf64>, %arg2: tensor<16xf64>) -> (tensor<16xf64>, tensor<16xf64>) {
// TD-NEXT:    %c = stablehlo.constant dense<true> : tensor<16xi1>
// TD-NEXT:    %0 = stablehlo.slice %arg0 [3:4, 0:16] : (tensor<8x16xf64>) -> tensor<1x16xf64>
// TD-NEXT:    %1 = stablehlo.reshape %0 : (tensor<1x16xf64>) -> tensor<16xf64>
// TD-NEXT:    %2 = stablehlo.select %c, %arg1, %arg2 : tensor<16xi1>, tensor<16xf64>
// TD-NEXT:    return %1, %2 : tensor<16xf64>, tensor<16xf64>
// TD-NEXT:  }

// -----

// Padding with true: whatever the operand mask holds, lane 0 is padded true,
// so the pick is lane 0 and the reduced mask is true.
func.func @padtrue(%v: tensor<8x16xf64>, %m0: tensor<5x16xi1>) -> (tensor<16xf64>, tensor<16xi1>) {
  %t = stablehlo.constant dense<true> : tensor<i1>
  %f = stablehlo.constant dense<false> : tensor<i1>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %m = stablehlo.pad %m0, %t, low = [2, 0], high = [1, 0], interior = [0, 0] : (tensor<5x16xi1>, tensor<i1>) -> tensor<8x16xi1>
  %r:2 = stablehlo.reduce(%v init: %cst), (%m init: %f) across dimensions = [0] : (tensor<8x16xf64>, tensor<8x16xi1>, tensor<f64>, tensor<i1>) -> (tensor<16xf64>, tensor<16xi1>)
   reducer(%arg2: tensor<f64>, %arg4: tensor<f64>) (%arg3: tensor<i1>, %arg5: tensor<i1>)  {
    %4 = stablehlo.select %arg3, %arg2, %arg4 : tensor<i1>, tensor<f64>
    %5 = stablehlo.or %arg3, %arg5 : tensor<i1>
    stablehlo.return %4, %5 : tensor<f64>, tensor<i1>
  }
  return %r#0, %r#1 : tensor<16xf64>, tensor<16xi1>
}

// CHECK:    func.func @padtrue(%arg0: tensor<8x16xf64>, %arg1: tensor<5x16xi1>) -> (tensor<16xf64>, tensor<16xi1>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<16xi1>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 0:16] : (tensor<8x16xf64>) -> tensor<1x16xf64>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<1x16xf64>) -> tensor<16xf64>
// CHECK-NEXT:    return %1, %c : tensor<16xf64>, tensor<16xi1>
// CHECK-NEXT:  }

// TD:  module {
// TD-NEXT:  func.func @padtrue(%arg0: tensor<8x16xf64>, %arg1: tensor<5x16xi1>) -> (tensor<16xf64>, tensor<16xi1>) {
// TD-NEXT:    %c = stablehlo.constant dense<true> : tensor<16xi1>
// TD-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 0:16] : (tensor<8x16xf64>) -> tensor<1x16xf64>
// TD-NEXT:    %1 = stablehlo.reshape %0 : (tensor<1x16xf64>) -> tensor<16xf64>
// TD-NEXT:    return %1, %c : tensor<16xf64>, tensor<16xi1>
// TD-NEXT:  }

// -----

// An all-true operand padded with false along a kept axis: the padded
// columns see no true lane, so the pick stays.
func.func @falsecols(%v: tensor<8x16xf64>) -> tensor<16xf64> {
  %c = stablehlo.constant dense<true> : tensor<1x12xi1>
  %f = stablehlo.constant dense<false> : tensor<i1>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %m = stablehlo.pad %c, %f, low = [3, 2], high = [4, 2], interior = [0, 0] : (tensor<1x12xi1>, tensor<i1>) -> tensor<8x16xi1>
  %r:2 = stablehlo.reduce(%v init: %cst), (%m init: %f) across dimensions = [0] : (tensor<8x16xf64>, tensor<8x16xi1>, tensor<f64>, tensor<i1>) -> (tensor<16xf64>, tensor<16xi1>)
   reducer(%arg2: tensor<f64>, %arg4: tensor<f64>) (%arg3: tensor<i1>, %arg5: tensor<i1>)  {
    %4 = stablehlo.select %arg3, %arg2, %arg4 : tensor<i1>, tensor<f64>
    %5 = stablehlo.or %arg3, %arg5 : tensor<i1>
    stablehlo.return %4, %5 : tensor<f64>, tensor<i1>
  }
  return %r#0 : tensor<16xf64>
}

// CHECK:    func.func @falsecols(%arg0: tensor<8x16xf64>) -> tensor<16xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<"0x0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010101010101010101010101000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"> : tensor<8x16xi1>
// CHECK-NEXT:    %0:2 = stablehlo.reduce(%arg0 init: %cst), (%c_0 init: %c) across dimensions = [0] : (tensor<8x16xf64>, tensor<8x16xi1>, tensor<f64>, tensor<i1>) -> (tensor<16xf64>, tensor<16xi1>)
// CHECK-NEXT:     reducer(%arg1: tensor<f64>, %arg3: tensor<f64>) (%arg2: tensor<i1>, %arg4: tensor<i1>)  {
// CHECK-NEXT:      %1 = stablehlo.select %arg2, %arg1, %arg3 : tensor<i1>, tensor<f64>
// CHECK-NEXT:      %2 = stablehlo.or %arg2, %arg4 : tensor<i1>
// CHECK-NEXT:      stablehlo.return %1, %2 : tensor<f64>, tensor<i1>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %0#0 : tensor<16xf64>
// CHECK-NEXT:  }

// TD:  module {
// TD-NEXT:  func.func @falsecols(%arg0: tensor<8x16xf64>) -> tensor<16xf64> {
// TD-NEXT:    %c = stablehlo.constant dense<true> : tensor<1x12xi1>
// TD-NEXT:    %c_0 = stablehlo.constant dense<false> : tensor<i1>
// TD-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// TD-NEXT:    %0 = stablehlo.pad %c, %c_0, low = [3, 2], high = [4, 2], interior = [0, 0] : (tensor<1x12xi1>, tensor<i1>) -> tensor<8x16xi1>
// TD-NEXT:    %1:2 = stablehlo.reduce(%arg0 init: %cst), (%0 init: %c_0) across dimensions = [0] : (tensor<8x16xf64>, tensor<8x16xi1>, tensor<f64>, tensor<i1>) -> (tensor<16xf64>, tensor<16xi1>)
// TD-NEXT:     reducer(%arg1: tensor<f64>, %arg3: tensor<f64>) (%arg2: tensor<i1>, %arg4: tensor<i1>)  {
// TD-NEXT:      %2 = stablehlo.select %arg2, %arg1, %arg3 : tensor<i1>, tensor<f64>
// TD-NEXT:      %3 = stablehlo.or %arg2, %arg4 : tensor<i1>
// TD-NEXT:      stablehlo.return %2, %3 : tensor<f64>, tensor<i1>
// TD-NEXT:    }
// TD-NEXT:    return %1#0 : tensor<16xf64>
// TD-NEXT:  }
