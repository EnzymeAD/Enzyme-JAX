// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=binop_pad_to_concat_mul<1>;concat_push_binop_mul<1>;concat_to_pad<1>;broadcast_in_dim_op_canon<16>" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// An integer zero pad is a zero pad: the padded rows of the multiply are zero,
// and the multiply is left over the row it reads.
func.func @int_zero(%x: tensor<1x1xi64>, %y: tensor<8x1xi64>) -> tensor<1x1xi64> {
  %c = stablehlo.constant dense<0> : tensor<i64>
  %p = stablehlo.pad %x, %c, low = [0, 0], high = [7, 0], interior = [0, 0] : (tensor<1x1xi64>, tensor<i64>) -> tensor<8x1xi64>
  %m = stablehlo.multiply %y, %p : tensor<8x1xi64>
  %s = stablehlo.slice %m [0:1, 0:1] : (tensor<8x1xi64>) -> tensor<1x1xi64>
  return %s : tensor<1x1xi64>
}

// CHECK:  func.func @int_zero(%arg0: tensor<1x1xi64>, %arg1: tensor<8x1xi64>) -> tensor<1x1xi64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [0:1, 0:1] : (tensor<8x1xi64>) -> tensor<1x1xi64>
// CHECK-NEXT:    %1 = stablehlo.multiply %0, %arg0 : tensor<1x1xi64>
// CHECK-NEXT:    %2 = stablehlo.concatenate %1, dim = 0 : (tensor<1x1xi64>) -> tensor<1x1xi64>
// CHECK-NEXT:    %3 = stablehlo.pad %2, %c, low = [0, 0], high = [7, 0], interior = [0, 0] : (tensor<1x1xi64>, tensor<i64>) -> tensor<8x1xi64>
// CHECK-NEXT:    %4 = stablehlo.slice %3 [0:1, 0:1] : (tensor<8x1xi64>) -> tensor<1x1xi64>
// CHECK-NEXT:    return %4 : tensor<1x1xi64>
// CHECK-NEXT:  }

// -----

// Split for the slice that reads one row of it: the rows read by none are the
// padding value, which concat_to_pad takes back into a pad over the one
// multiply that is read (the multiply is not pushed back out of the
// concatenation either, which would rebuild the pad, to be split again,
// without end).
func.func @int_two(%x: tensor<1x1xi64>, %y: tensor<8x1xi64>) -> tensor<1x1xi64> {
  %c = stablehlo.constant dense<2> : tensor<i64>
  %p = stablehlo.pad %x, %c, low = [0, 0], high = [7, 0], interior = [0, 0] : (tensor<1x1xi64>, tensor<i64>) -> tensor<8x1xi64>
  %m = stablehlo.multiply %y, %p : tensor<8x1xi64>
  %s = stablehlo.slice %m [0:1, 0:1] : (tensor<8x1xi64>) -> tensor<1x1xi64>
  return %s : tensor<1x1xi64>
}

// CHECK:  func.func @int_two(%arg0: tensor<1x1xi64>, %arg1: tensor<8x1xi64>) -> tensor<1x1xi64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [0:1, 0:1] : (tensor<8x1xi64>) -> tensor<1x1xi64>
// CHECK-NEXT:    %1 = stablehlo.multiply %0, %arg0 : tensor<1x1xi64>
// CHECK-NEXT:    %2 = stablehlo.concatenate %1, dim = 0 : (tensor<1x1xi64>) -> tensor<1x1xi64>
// CHECK-NEXT:    %3 = stablehlo.pad %2, %c, low = [0, 0], high = [7, 0], interior = [0, 0] : (tensor<1x1xi64>, tensor<i64>) -> tensor<8x1xi64>
// CHECK-NEXT:    %4 = stablehlo.slice %3 [0:1, 0:1] : (tensor<8x1xi64>) -> tensor<1x1xi64>
// CHECK-NEXT:    return %4 : tensor<1x1xi64>
// CHECK-NEXT:  }
