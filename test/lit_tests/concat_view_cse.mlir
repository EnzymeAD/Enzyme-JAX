// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=concat_view_cse" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// The same values stacked twice, through reshapes and through transposing
// broadcasts: the second stack is a transpose of the first.
func.func @stacks(%a: tensor<8x3x4xf64>, %b: tensor<8x3x4xf64>) -> (tensor<2x8x3x4xf64>, tensor<2x8x4x3xf64>) {
  %0 = stablehlo.reshape %a : (tensor<8x3x4xf64>) -> tensor<1x8x3x4xf64>
  %1 = stablehlo.reshape %b : (tensor<8x3x4xf64>) -> tensor<1x8x3x4xf64>
  %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<1x8x3x4xf64>, tensor<1x8x3x4xf64>) -> tensor<2x8x3x4xf64>
  %3 = stablehlo.broadcast_in_dim %a, dims = [1, 3, 2] : (tensor<8x3x4xf64>) -> tensor<1x8x4x3xf64>
  %4 = stablehlo.broadcast_in_dim %b, dims = [1, 3, 2] : (tensor<8x3x4xf64>) -> tensor<1x8x4x3xf64>
  %5 = stablehlo.concatenate %3, %4, dim = 0 : (tensor<1x8x4x3xf64>, tensor<1x8x4x3xf64>) -> tensor<2x8x4x3xf64>
  return %2, %5 : tensor<2x8x3x4xf64>, tensor<2x8x4x3xf64>
}

// CHECK-LABEL: func.func @stacks
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<8x3x4xf64>) -> tensor<1x8x3x4xf64>
// CHECK-NEXT:    %1 = stablehlo.reshape %arg1 : (tensor<8x3x4xf64>) -> tensor<1x8x3x4xf64>
// CHECK-NEXT:    %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<1x8x3x4xf64>, tensor<1x8x3x4xf64>) -> tensor<2x8x3x4xf64>
// CHECK-NEXT:    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 3, 2] : (tensor<2x8x3x4xf64>) -> tensor<2x8x4x3xf64>
// CHECK-NEXT:    return %2, %3 : tensor<2x8x3x4xf64>, tensor<2x8x4x3xf64>
// CHECK-NEXT:  }

// The same values concatenated along their dimension 1, once as they are and
// once transposed.
func.func @along(%a: tensor<8x3x4xf64>, %b: tensor<8x2x4xf64>) -> (tensor<8x5x4xf64>, tensor<4x8x5xf64>) {
  %0 = stablehlo.concatenate %a, %b, dim = 1 : (tensor<8x3x4xf64>, tensor<8x2x4xf64>) -> tensor<8x5x4xf64>
  %1 = stablehlo.broadcast_in_dim %a, dims = [1, 2, 0] : (tensor<8x3x4xf64>) -> tensor<4x8x3xf64>
  %2 = stablehlo.broadcast_in_dim %b, dims = [1, 2, 0] : (tensor<8x2x4xf64>) -> tensor<4x8x2xf64>
  %3 = stablehlo.concatenate %1, %2, dim = 2 : (tensor<4x8x3xf64>, tensor<4x8x2xf64>) -> tensor<4x8x5xf64>
  return %0, %3 : tensor<8x5x4xf64>, tensor<4x8x5xf64>
}

// CHECK-LABEL: func.func @along
// CHECK-NEXT:    %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<8x3x4xf64>, tensor<8x2x4xf64>) -> tensor<8x5x4xf64>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2, 0] : (tensor<8x5x4xf64>) -> tensor<4x8x5xf64>
// CHECK-NEXT:    return %0, %1 : tensor<8x5x4xf64>, tensor<4x8x5xf64>
// CHECK-NEXT:  }

// The values in another order: left alone.
func.func @order(%a: tensor<8x3x4xf64>, %b: tensor<8x3x4xf64>) -> (tensor<2x8x3x4xf64>, tensor<2x8x4x3xf64>) {
  %0 = stablehlo.reshape %a : (tensor<8x3x4xf64>) -> tensor<1x8x3x4xf64>
  %1 = stablehlo.reshape %b : (tensor<8x3x4xf64>) -> tensor<1x8x3x4xf64>
  %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<1x8x3x4xf64>, tensor<1x8x3x4xf64>) -> tensor<2x8x3x4xf64>
  %3 = stablehlo.broadcast_in_dim %b, dims = [1, 3, 2] : (tensor<8x3x4xf64>) -> tensor<1x8x4x3xf64>
  %4 = stablehlo.broadcast_in_dim %a, dims = [1, 3, 2] : (tensor<8x3x4xf64>) -> tensor<1x8x4x3xf64>
  %5 = stablehlo.concatenate %3, %4, dim = 0 : (tensor<1x8x4x3xf64>, tensor<1x8x4x3xf64>) -> tensor<2x8x4x3xf64>
  return %2, %5 : tensor<2x8x3x4xf64>, tensor<2x8x4x3xf64>
}


// CHECK-LABEL: func.func @order
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<8x3x4xf64>) -> tensor<1x8x3x4xf64>
// CHECK-NEXT:    %1 = stablehlo.reshape %arg1 : (tensor<8x3x4xf64>) -> tensor<1x8x3x4xf64>
// CHECK-NEXT:    %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<1x8x3x4xf64>, tensor<1x8x3x4xf64>) -> tensor<2x8x3x4xf64>
// CHECK-NEXT:    %3 = stablehlo.broadcast_in_dim %arg1, dims = [1, 3, 2] : (tensor<8x3x4xf64>) -> tensor<1x8x4x3xf64>
// CHECK-NEXT:    %4 = stablehlo.broadcast_in_dim %arg0, dims = [1, 3, 2] : (tensor<8x3x4xf64>) -> tensor<1x8x4x3xf64>
// CHECK-NEXT:    %5 = stablehlo.concatenate %3, %4, dim = 0 : (tensor<1x8x4x3xf64>, tensor<1x8x4x3xf64>) -> tensor<2x8x4x3xf64>
// CHECK-NEXT:    return %2, %5 : tensor<2x8x3x4xf64>, tensor<2x8x4x3xf64>
// CHECK-NEXT:  }
