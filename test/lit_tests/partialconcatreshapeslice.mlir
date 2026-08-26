// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt="passses=65536" | FileCheck %s

func.func @test(%arg0: tensor<5x10x10xf32>, %arg1: tensor<10x3x10xf32>) -> tensor<10x8x10xf32> {
    %0 = stablehlo.slice %arg0 [0:1, 0:10, 0:10] : (tensor<5x10x10xf32>) -> tensor<1x10x10xf32>
    %1 = stablehlo.slice %arg0 [1:2, 0:10, 0:10] : (tensor<5x10x10xf32>) -> tensor<1x10x10xf32>
    %2 = stablehlo.slice %arg0 [2:3, 0:10, 0:10] : (tensor<5x10x10xf32>) -> tensor<1x10x10xf32>
    %3 = stablehlo.slice %arg0 [3:4, 0:10, 0:10] : (tensor<5x10x10xf32>) -> tensor<1x10x10xf32>
    %4 = stablehlo.slice %arg0 [4:5, 0:10, 0:10] : (tensor<5x10x10xf32>) -> tensor<1x10x10xf32>

    %5 = stablehlo.reshape %0 : (tensor<1x10x10xf32>) -> tensor<10x1x10xf32>
    %6 = stablehlo.reshape %1 : (tensor<1x10x10xf32>) -> tensor<10x1x10xf32>
    %7 = stablehlo.reshape %2 : (tensor<1x10x10xf32>) -> tensor<10x1x10xf32>
    %8 = stablehlo.reshape %3 : (tensor<1x10x10xf32>) -> tensor<10x1x10xf32>
    %9 = stablehlo.reshape %4 : (tensor<1x10x10xf32>) -> tensor<10x1x10xf32>

    %10 = stablehlo.concatenate %5, %6, %7, %arg1, %8, %9, dim = 1 : (tensor<10x1x10xf32>, tensor<10x1x10xf32>, tensor<10x1x10xf32>, tensor<10x3x10xf32>, tensor<10x1x10xf32>, tensor<10x1x10xf32>) -> tensor<10x8x10xf32>
    return %10 : tensor<10x8x10xf32>
}

// CHECK: func.func @test(%arg0: tensor<5x10x10xf32>, %arg1: tensor<10x3x10xf32>) -> tensor<10x8x10xf32> {
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [1, 0, 2] : (tensor<5x10x10xf32>) -> tensor<10x5x10xf32>
// CHECK-NEXT:     %1 = stablehlo.slice %0 [0:10, 0:3, 0:10] : (tensor<10x5x10xf32>) -> tensor<10x3x10xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %0 [0:10, 3:5, 0:10] : (tensor<10x5x10xf32>) -> tensor<10x2x10xf32>
// CHECK-NEXT:     %3 = stablehlo.concatenate %1, %arg1, %2, dim = 1 : (tensor<10x3x10xf32>, tensor<10x3x10xf32>, tensor<10x2x10xf32>) -> tensor<10x8x10xf32>
// CHECK-NEXT:     return %3 : tensor<10x8x10xf32>
// CHECK-NEXT: }
