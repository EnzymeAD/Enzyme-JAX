// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main1(%arg0: tensor<8x4xf32>) -> tensor<8x4xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<8x4xf32>) -> tensor<4x8xf32>
    %1 = stablehlo.sign %0 : tensor<4x8xf32>
    %2 = stablehlo.abs %0 : tensor<4x8xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<4x8xf32>
    %4 = stablehlo.transpose %3, dims = [1, 0] : (tensor<4x8xf32>) -> tensor<8x4xf32>
    return %4 : tensor<8x4xf32>
}

// CHECK:  func.func @main1(%arg0: tensor<8x4xf32>) -> tensor<8x4xf32> {
// CHECK-NEXT:     return %arg0 : tensor<8x4xf32>
// CHECK-NEXT: }

func.func @main2(%arg0: tensor<8x4xf32>) -> tensor<8x4xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<8x4xf32>) -> tensor<4x8xf32>
    %1 = stablehlo.sign %0 : tensor<4x8xf32>
    %2 = stablehlo.abs %0 : tensor<4x8xf32>
    %3 = stablehlo.multiply %2, %1 : tensor<4x8xf32>
    %4 = stablehlo.transpose %3, dims = [1, 0] : (tensor<4x8xf32>) -> tensor<8x4xf32>
    return %4 : tensor<8x4xf32>
}

// CHECK:  func.func @main2(%arg0: tensor<8x4xf32>) -> tensor<8x4xf32> {
// CHECK-NEXT:     return %arg0 : tensor<8x4xf32>
// CHECK-NEXT: }
