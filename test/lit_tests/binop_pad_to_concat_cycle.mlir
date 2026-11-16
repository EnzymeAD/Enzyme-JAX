// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=binop_pad_to_concat_mul<1>;concat_push_binop_mul<1>;broadcast_in_dim_op_canon<16>;concat_to_pad<1>" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// A multiply over an integer zero pad: the padded rows are zero, not the
// multiply over them (which concat_push_binop would take back through the
// concat, and concat_to_pad turn back into the pad, without end).
func.func @int_zero(%x: tensor<8x1xi64>, %y: tensor<1x1xi64>) -> tensor<8x1xi64> {
  %z = stablehlo.constant dense<0> : tensor<i64>
  %p = stablehlo.pad %y, %z, low = [0, 0], high = [7, 0], interior = [0, 0] : (tensor<1x1xi64>, tensor<i64>) -> tensor<8x1xi64>
  %m = stablehlo.multiply %x, %p : tensor<8x1xi64>
  return %m : tensor<8x1xi64>
}

// CHECK:  func.func @int_zero(%arg0: tensor<8x1xi64>, %arg1: tensor<1x1xi64>) -> tensor<8x1xi64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 0:1] : (tensor<8x1xi64>) -> tensor<1x1xi64>
// CHECK-NEXT:    %1 = stablehlo.multiply %0, %arg1 : tensor<1x1xi64>
// CHECK-NEXT:    %2 = stablehlo.concatenate %1, dim = 0 : (tensor<1x1xi64>) -> tensor<1x1xi64>
// CHECK-NEXT:    %3 = stablehlo.pad %2, %c, low = [0, 0], high = [7, 0], interior = [0, 0] : (tensor<1x1xi64>, tensor<i64>) -> tensor<8x1xi64>
// CHECK-NEXT:    return %3 : tensor<8x1xi64>
// CHECK-NEXT:  }

// -----

// A multiply over a pad of 2, only the padded rows read: the row read by none
// is left the padding value, so that the concat is not one of two multiplies.
func.func @read_part(%x: tensor<8x1xi64>, %y: tensor<1x1xi64>, %c: tensor<8x1xi64>) -> tensor<7x1xi64> {
  %z = stablehlo.constant dense<2> : tensor<i64>
  %p = stablehlo.pad %y, %z, low = [0, 0], high = [7, 0], interior = [0, 0] : (tensor<1x1xi64>, tensor<i64>) -> tensor<8x1xi64>
  %m = stablehlo.multiply %x, %p : tensor<8x1xi64>
  %a = stablehlo.add %m, %c : tensor<8x1xi64>
  %s = stablehlo.slice %a [1:8, 0:1] : (tensor<8x1xi64>) -> tensor<7x1xi64>
  return %s : tensor<7x1xi64>
}

// CHECK:  func.func @read_part(%arg0: tensor<8x1xi64>, %arg1: tensor<1x1xi64>, %arg2: tensor<8x1xi64>) -> tensor<7x1xi64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<2> : tensor<7x1xi64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [1:8, 0:1] : (tensor<8x1xi64>) -> tensor<7x1xi64>
// CHECK-NEXT:    %1 = stablehlo.multiply %0, %c_0 : tensor<7x1xi64>
// CHECK-NEXT:    %2 = stablehlo.concatenate %1, dim = 0 : (tensor<7x1xi64>) -> tensor<7x1xi64>
// CHECK-NEXT:    %3 = stablehlo.pad %2, %c, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<7x1xi64>, tensor<i64>) -> tensor<8x1xi64>
// CHECK-NEXT:    %4 = stablehlo.add %3, %arg2 : tensor<8x1xi64>
// CHECK-NEXT:    %5 = stablehlo.slice %4 [1:8, 0:1] : (tensor<8x1xi64>) -> tensor<7x1xi64>
// CHECK-NEXT:    return %5 : tensor<7x1xi64>
// CHECK-NEXT:  }
