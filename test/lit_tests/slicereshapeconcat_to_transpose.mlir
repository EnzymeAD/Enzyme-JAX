// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

func.func @test(%arg0: tensor<4x10x10xf32>) -> tensor<10x4x10xf32> {
    %69 = stablehlo.slice %arg0 [0:1, 0:10, 0:10] : (tensor<4x10x10xf32>) -> tensor<1x10x10xf32>
    %70 = stablehlo.slice %arg0 [1:2, 0:10, 0:10] : (tensor<4x10x10xf32>) -> tensor<1x10x10xf32>
    %71 = stablehlo.slice %arg0 [2:3, 0:10, 0:10] : (tensor<4x10x10xf32>) -> tensor<1x10x10xf32>
    %72 = stablehlo.slice %arg0 [3:4, 0:10, 0:10] : (tensor<4x10x10xf32>) -> tensor<1x10x10xf32>
    %73 = stablehlo.reshape %69 : (tensor<1x10x10xf32>) -> tensor<10x1x10xf32>
    %74 = stablehlo.reshape %70 : (tensor<1x10x10xf32>) -> tensor<10x1x10xf32>
    %75 = stablehlo.reshape %71 : (tensor<1x10x10xf32>) -> tensor<10x1x10xf32>
    %76 = stablehlo.reshape %72 : (tensor<1x10x10xf32>) -> tensor<10x1x10xf32>
    %77 = stablehlo.concatenate %73, %74, %75, %76, dim = 1 : (tensor<10x1x10xf32>, tensor<10x1x10xf32>, tensor<10x1x10xf32>, tensor<10x1x10xf32>) -> tensor<10x4x10xf32>
    return %77 : tensor<10x4x10xf32>
}

// CHECK: func.func @test(%arg0: tensor<4x10x10xf32>) -> tensor<10x4x10xf32> {
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [1, 0, 2] : (tensor<4x10x10xf32>) -> tensor<10x4x10xf32>
// CHECK-NEXT:   return %0 : tensor<10x4x10xf32>
// CHECK-NEXT: }

func.func @test2(%arg0: tensor<4x10x10xf32>) -> tensor<10x10x2xf32> {
    %70 = stablehlo.slice %arg0 [1:2, 0:10, 0:10] : (tensor<4x10x10xf32>) -> tensor<1x10x10xf32>
    %71 = stablehlo.slice %arg0 [2:3, 0:10, 0:10] : (tensor<4x10x10xf32>) -> tensor<1x10x10xf32>
    %74 = stablehlo.reshape %70 : (tensor<1x10x10xf32>) -> tensor<10x10x1xf32>
    %75 = stablehlo.reshape %71 : (tensor<1x10x10xf32>) -> tensor<10x10x1xf32>
    %76 = stablehlo.concatenate %74, %75, dim = 2 : (tensor<10x10x1xf32>, tensor<10x10x1xf32>) -> tensor<10x10x2xf32>
    return %76 : tensor<10x10x2xf32>
}

// CHECK: func.func @test2(%arg0: tensor<4x10x10xf32>) -> tensor<10x10x2xf32> {
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [1:3, 0:10, 0:10] : (tensor<4x10x10xf32>) -> tensor<2x10x10xf32>
// CHECK-NEXT:   %1 = stablehlo.transpose %0, dims = [2, 1, 0] : (tensor<2x10x10xf32>) -> tensor<10x10x2xf32>
// CHECK-NEXT:   return %1 : tensor<10x10x2xf32>
// CHECK-NEXT: }

// CHECK: func.func @test_nomatch(%arg0: tensor<4x10x10xf32>) -> tensor<10x10x2xf32> {
func.func @test_nomatch(%arg0: tensor<4x10x10xf32>) -> tensor<10x10x2xf32> {
    // CHECK: stablehlo.slice
    %70 = stablehlo.slice %arg0 [1:2, 0:10, 0:10] : (tensor<4x10x10xf32>) -> tensor<1x10x10xf32>
    // CHECK: stablehlo.slice
    %71 = stablehlo.slice %arg0 [3:4, 0:10, 0:10] : (tensor<4x10x10xf32>) -> tensor<1x10x10xf32>
    // CHECK: stablehlo.reshape
    %74 = stablehlo.reshape %70 : (tensor<1x10x10xf32>) -> tensor<10x10x1xf32>
    // CHECK: stablehlo.reshape
    %75 = stablehlo.reshape %71 : (tensor<1x10x10xf32>) -> tensor<10x10x1xf32>
    // CHECK: stablehlo.concatenate
    %76 = stablehlo.concatenate %74, %75, dim = 2 : (tensor<10x10x1xf32>, tensor<10x10x1xf32>) -> tensor<10x10x2xf32>
    return %76 : tensor<10x10x2xf32>
}

func.func @test_nomatch2(%arg0: tensor<4x10x10xf32>) -> tensor<10x10x3xf32> {
    // CHECK: stablehlo.slice
    %70 = stablehlo.slice %arg0 [1:3, 0:10, 0:10] : (tensor<4x10x10xf32>) -> tensor<2x10x10xf32>
    %71 = stablehlo.slice %arg0 [2:3, 0:10, 0:10] : (tensor<4x10x10xf32>) -> tensor<1x10x10xf32>
    // CHECK: stablehlo.reshape
    %74 = stablehlo.reshape %70 : (tensor<2x10x10xf32>) -> tensor<10x10x2xf32>
    // CHECK: stablehlo.reshape
    %75 = stablehlo.reshape %71 : (tensor<1x10x10xf32>) -> tensor<10x10x1xf32>
    // CHECK: stablehlo.concatenate
    %76 = stablehlo.concatenate %74, %75, dim = 2 : (tensor<10x10x2xf32>, tensor<10x10x1xf32>) -> tensor<10x10x3xf32>
    return %76 : tensor<10x10x3xf32>
}
