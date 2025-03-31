
func.func @test_reshape_transpose(%arg0: tensor<5x4xf64>) -> tensor<4x5x1xf64> {
    %0 = stablehlo.reshape %arg0 : (tensor<5x4xf64>) -> tensor<1x5x4xf64>
    %1 = stablehlo.transpose %0, dims = [2, 1, 0] : (tensor<1x5x4xf64>) -> tensor<4x5x1xf64>
    return %1 : tensor<4x5x1xf64>
}

// CHECK-LABEL: func.func @test_reshape_transpose(%arg0: tensor<5x4xf64>) -> tensor<4x5x1xf64> {
// CHECK-NEXT:     %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 0] : (tensor<5x4xf64>) -> tensor<4x5x1xf64>
// CHECK-NEXT:     return %0 : tensor<4x5x1xf64>
// CHECK-NEXT: }

// func.func @test_transpose_reshape(%arg0: tensor<1x5x4xf64>) -> tensor<4x1x5xf64> {
//     %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<1x5x4xf64>) -> tensor<4x5x1xf64>
//     %1 = stablehlo.reshape %0: (tensor<4x5x1xf64>) -> tensor<4x1x5xf64>
//     return %1 : tensor<4x1x5xf64>
// }
