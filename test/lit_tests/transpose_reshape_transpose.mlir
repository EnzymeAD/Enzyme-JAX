// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=transpose_reshape_transpose" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// The reshape splits the transposed D's last dimension (81) into (3, 27);
// the split is of D's own dimension 1, and the outer transpose, which moves
// the 3 to the front, composes with the inner one into a single transpose
// of the split D.
func.func @split(%D: tensor<8x81x3x3xf64>) -> tensor<3x8x3x3x27xf64> {
  %0 = stablehlo.transpose %D, dims = [0, 3, 2, 1] : (tensor<8x81x3x3xf64>) -> tensor<8x3x3x81xf64>
  %1 = stablehlo.reshape %0 : (tensor<8x3x3x81xf64>) -> tensor<8x3x3x3x27xf64>
  %2 = stablehlo.transpose %1, dims = [3, 0, 1, 2, 4] : (tensor<8x3x3x3x27xf64>) -> tensor<3x8x3x3x27xf64>
  return %2 : tensor<3x8x3x3x27xf64>
}

// CHECK: func.func @split(%arg0: tensor<8x81x3x3xf64>) -> tensor<3x8x3x3x27xf64> {
// CHECK-NEXT:   %0 = stablehlo.reshape %arg0 : (tensor<8x81x3x3xf64>) -> tensor<8x3x27x3x3xf64>
// CHECK-NEXT:   %1 = stablehlo.transpose %0, dims = [1, 0, 4, 3, 2] : (tensor<8x3x27x3x3xf64>) -> tensor<3x8x3x3x27xf64>
// CHECK-NEXT:   return %1 : tensor<3x8x3x3x27xf64>
// CHECK-NEXT: }

// The outer transpose keeps the split (2, 3) as a block: transpose_reshape's
// form (it would move the transpose the other way), left alone.
func.func @blocks(%D: tensor<8x6x4xf64>) -> tensor<4x8x2x3xf64> {
  %0 = stablehlo.transpose %D, dims = [2, 0, 1] : (tensor<8x6x4xf64>) -> tensor<4x8x6xf64>
  %1 = stablehlo.reshape %0 : (tensor<4x8x6xf64>) -> tensor<4x8x2x3xf64>
  %2 = stablehlo.transpose %1, dims = [0, 1, 2, 3] : (tensor<4x8x2x3xf64>) -> tensor<4x8x2x3xf64>
  return %2 : tensor<4x8x2x3xf64>
}

// CHECK: func.func @blocks(%arg0: tensor<8x6x4xf64>) -> tensor<4x8x2x3xf64> {
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [2, 0, 1] : (tensor<8x6x4xf64>) -> tensor<4x8x6xf64>
// CHECK-NEXT:   %1 = stablehlo.reshape %0 : (tensor<4x8x6xf64>) -> tensor<4x8x2x3xf64>
// CHECK-NEXT:   %2 = stablehlo.transpose %1, dims = [0, 1, 2, 3] : (tensor<4x8x2x3xf64>) -> tensor<4x8x2x3xf64>
// CHECK-NEXT:   return %2 : tensor<4x8x2x3xf64>
// CHECK-NEXT: }

// A reshape that merges dimensions is not a split: left alone.
func.func @merge(%D: tensor<8x3x27x3xf64>) -> tensor<3x8x81xf64> {
  %0 = stablehlo.transpose %D, dims = [0, 3, 1, 2] : (tensor<8x3x27x3xf64>) -> tensor<8x3x3x27xf64>
  %1 = stablehlo.reshape %0 : (tensor<8x3x3x27xf64>) -> tensor<8x3x81xf64>
  %2 = stablehlo.transpose %1, dims = [1, 0, 2] : (tensor<8x3x81xf64>) -> tensor<3x8x81xf64>
  return %2 : tensor<3x8x81xf64>
}

// CHECK: func.func @merge(%arg0: tensor<8x3x27x3xf64>) -> tensor<3x8x81xf64> {
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [0, 3, 1, 2] : (tensor<8x3x27x3xf64>) -> tensor<8x3x3x27xf64>
// CHECK-NEXT:   %1 = stablehlo.reshape %0 : (tensor<8x3x3x27xf64>) -> tensor<8x3x81xf64>
// CHECK-NEXT:   %2 = stablehlo.transpose %1, dims = [1, 0, 2] : (tensor<8x3x81xf64>) -> tensor<3x8x81xf64>
// CHECK-NEXT:   return %2 : tensor<3x8x81xf64>
// CHECK-NEXT: }
