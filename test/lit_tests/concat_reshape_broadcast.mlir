// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=concat_reshape_broadcast" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Broadcasts that are also used elsewhere, stacked through reshapes that
// insert the concatenated unit dimension.
func.func @stack(%a: tensor<1x1xf64>, %b: tensor<1x1xf64>) -> (tensor<2x8x5xf64>, tensor<8x5xf64>, tensor<8x5xf64>) {
  %0 = stablehlo.broadcast_in_dim %a, dims = [0, 1] : (tensor<1x1xf64>) -> tensor<8x5xf64>
  %1 = stablehlo.broadcast_in_dim %b, dims = [0, 1] : (tensor<1x1xf64>) -> tensor<8x5xf64>
  %2 = stablehlo.reshape %0 : (tensor<8x5xf64>) -> tensor<1x8x5xf64>
  %3 = stablehlo.reshape %1 : (tensor<8x5xf64>) -> tensor<1x8x5xf64>
  %4 = stablehlo.concatenate %2, %3, dim = 0 : (tensor<1x8x5xf64>, tensor<1x8x5xf64>) -> tensor<2x8x5xf64>
  return %4, %0, %1 : tensor<2x8x5xf64>, tensor<8x5xf64>, tensor<8x5xf64>
}

// CHECK-LABEL: func.func @stack
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x1xf64>) -> tensor<8x5xf64>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<1x1xf64>) -> tensor<8x5xf64>
// CHECK-NEXT:    %2 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] : (tensor<1x1xf64>) -> tensor<1x8x5xf64>
// CHECK-NEXT:    %3 = stablehlo.broadcast_in_dim %arg1, dims = [1, 2] : (tensor<1x1xf64>) -> tensor<1x8x5xf64>
// CHECK-NEXT:    %4 = stablehlo.concatenate %2, %3, dim = 0 : (tensor<1x8x5xf64>, tensor<1x8x5xf64>) -> tensor<2x8x5xf64>
// CHECK-NEXT:    return %4, %0, %1 : tensor<2x8x5xf64>, tensor<8x5xf64>, tensor<8x5xf64>
// CHECK-NEXT:  }

// A reshape that merges dimensions is left alone.
func.func @merge(%a: tensor<4xf64>, %b: tensor<4xf64>) -> (tensor<2x12xf64>, tensor<4x3xf64>, tensor<4x3xf64>) {
  %0 = stablehlo.broadcast_in_dim %a, dims = [0] : (tensor<4xf64>) -> tensor<4x3xf64>
  %1 = stablehlo.reshape %0 : (tensor<4x3xf64>) -> tensor<1x12xf64>
  %2 = stablehlo.broadcast_in_dim %b, dims = [0] : (tensor<4xf64>) -> tensor<4x3xf64>
  %3 = stablehlo.reshape %2 : (tensor<4x3xf64>) -> tensor<1x12xf64>
  %4 = stablehlo.concatenate %1, %3, dim = 0 : (tensor<1x12xf64>, tensor<1x12xf64>) -> tensor<2x12xf64>
  return %4, %0, %2 : tensor<2x12xf64>, tensor<4x3xf64>, tensor<4x3xf64>
}

// CHECK-LABEL: func.func @merge
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<4xf64>) -> tensor<4x3xf64>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<4x3xf64>) -> tensor<1x12xf64>
// CHECK-NEXT:    %2 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<4xf64>) -> tensor<4x3xf64>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<4x3xf64>) -> tensor<1x12xf64>
// CHECK-NEXT:    %4 = stablehlo.concatenate %1, %3, dim = 0 : (tensor<1x12xf64>, tensor<1x12xf64>) -> tensor<2x12xf64>
// CHECK-NEXT:    return %4, %0, %2 : tensor<2x12xf64>, tensor<4x3xf64>, tensor<4x3xf64>
// CHECK-NEXT:  }

// A broadcast used only by its reshape is left to
// reshape_insertions_broadcast_in_dim_simplify.
func.func @single(%a: tensor<4xf64>, %b: tensor<4xf64>) -> tensor<2x4x3xf64> {
  %0 = stablehlo.broadcast_in_dim %a, dims = [0] : (tensor<4xf64>) -> tensor<4x3xf64>
  %1 = stablehlo.reshape %0 : (tensor<4x3xf64>) -> tensor<1x4x3xf64>
  %2 = stablehlo.broadcast_in_dim %b, dims = [0] : (tensor<4xf64>) -> tensor<4x3xf64>
  %3 = stablehlo.reshape %2 : (tensor<4x3xf64>) -> tensor<1x4x3xf64>
  %4 = stablehlo.concatenate %1, %3, dim = 0 : (tensor<1x4x3xf64>, tensor<1x4x3xf64>) -> tensor<2x4x3xf64>
  return %4 : tensor<2x4x3xf64>
}

// CHECK-LABEL: func.func @single
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<4xf64>) -> tensor<4x3xf64>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<4x3xf64>) -> tensor<1x4x3xf64>
// CHECK-NEXT:    %2 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<4xf64>) -> tensor<4x3xf64>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<4x3xf64>) -> tensor<1x4x3xf64>
// CHECK-NEXT:    %4 = stablehlo.concatenate %1, %3, dim = 0 : (tensor<1x4x3xf64>, tensor<1x4x3xf64>) -> tensor<2x4x3xf64>
// CHECK-NEXT:    return %4 : tensor<2x4x3xf64>
// CHECK-NEXT:  }
