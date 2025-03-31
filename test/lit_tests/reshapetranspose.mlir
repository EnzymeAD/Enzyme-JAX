// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @test_reshape_transpose(%arg0: tensor<5x4xf64>) -> tensor<4x5x1xf64> {
    %0 = stablehlo.reshape %arg0 : (tensor<5x4xf64>) -> tensor<1x5x4xf64>
    %1 = stablehlo.transpose %0, dims = [2, 1, 0] : (tensor<1x5x4xf64>) -> tensor<4x5x1xf64>
    return %1 : tensor<4x5x1xf64>
}

// CHECK-LABEL: func.func @test_reshape_transpose(%arg0: tensor<5x4xf64>) -> tensor<4x5x1xf64> {
// CHECK-NEXT:     %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 0] : (tensor<5x4xf64>) -> tensor<4x5x1xf64>
// CHECK-NEXT:     return %0 : tensor<4x5x1xf64>
// CHECK-NEXT: }

func.func @test_reshape_transpose2(%arg0: tensor<8x65x66x512xf32>) -> tensor<1x8x66x65x512xf32> {
    %0 = stablehlo.reshape %arg0 : (tensor<8x65x66x512xf32>) -> tensor<8x65x66x1x512xf32>
    %1 = stablehlo.transpose %0, dims = [3, 0, 2, 1, 4] : (tensor<8x65x66x1x512xf32>) -> tensor<1x8x66x65x512xf32>
    return %1 : tensor<1x8x66x65x512xf32>
}

// CHECK-LABEL: func.func @test_reshape_transpose2(%arg0: tensor<8x65x66x512xf32>) -> tensor<1x8x66x65x512xf32> {
// CHECK-NEXT:     %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 3, 2, 4] : (tensor<8x65x66x512xf32>) -> tensor<1x8x66x65x512xf32>
// CHECK-NEXT:     return %0 : tensor<1x8x66x65x512xf32>
// CHECK-NEXT: }

func.func @test_reshape_transpose3(%arg0: tensor<8x65x66x512xf32>) -> tensor<8x1x66x65x512xf32> {
    %0 = stablehlo.reshape %arg0 : (tensor<8x65x66x512xf32>) -> tensor<8x65x66x1x512xf32>
    %1 = stablehlo.transpose %0, dims = [0, 3, 2, 1, 4] : (tensor<8x65x66x1x512xf32>) -> tensor<8x1x66x65x512xf32>
    return %1 : tensor<8x1x66x65x512xf32>
}

// CHECK-LABEL: func.func @test_reshape_transpose3(%arg0: tensor<8x65x66x512xf32>) -> tensor<8x1x66x65x512xf32> {
// CHECK-NEXT:     %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 3, 2, 4] : (tensor<8x65x66x512xf32>) -> tensor<8x1x66x65x512xf32>
// CHECK-NEXT:     return %0 : tensor<8x1x66x65x512xf32>
// CHECK-NEXT: }
