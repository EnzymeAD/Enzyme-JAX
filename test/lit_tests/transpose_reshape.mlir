// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=transpose_reshape},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

func.func @main1(%arg0: tensor<1x2x3x20xf32>) -> tensor<2x1x4x5x3xf32> {
    %0 = stablehlo.reshape %arg0 : (tensor<1x2x3x20xf32>) -> tensor<1x2x3x4x5xf32>
    %1 = stablehlo.transpose %0, dims = [1, 0, 3, 4, 2] : (tensor<1x2x3x4x5xf32>) -> tensor<2x1x4x5x3xf32>
    return %1 : tensor<2x1x4x5x3xf32>
}

// CHECK: func.func @main1(%arg0: tensor<1x2x3x20xf32>) -> tensor<2x1x4x5x3xf32> {
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [1, 0, 3, 2] : (tensor<1x2x3x20xf32>) -> tensor<2x1x20x3xf32>
// CHECK-NEXT:     %1 = stablehlo.reshape %0 : (tensor<2x1x20x3xf32>) -> tensor<2x1x4x5x3xf32>
// CHECK-NEXT:     return %1 : tensor<2x1x4x5x3xf32>
// CHECK-NEXT: }

func.func @main2(%arg0: tensor<1x2x20x3xf32>) -> tensor<3x1x2x5x4xf32> {
    %0 = stablehlo.reshape %arg0 : (tensor<1x2x20x3xf32>) -> tensor<1x2x5x4x3xf32>
    %1 = stablehlo.transpose %0, dims = [4, 0, 1, 2, 3] : (tensor<1x2x5x4x3xf32>) -> tensor<3x1x2x5x4xf32>
    return %1 : tensor<3x1x2x5x4xf32>
}

// CHECK: func.func @main2(%arg0: tensor<1x2x20x3xf32>) -> tensor<3x1x2x5x4xf32> {
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [3, 0, 1, 2] : (tensor<1x2x20x3xf32>) -> tensor<3x1x2x20xf32>
// CHECK-NEXT:     %1 = stablehlo.reshape %0 : (tensor<3x1x2x20xf32>) -> tensor<3x1x2x5x4xf32>
// CHECK-NEXT:     return %1 : tensor<3x1x2x5x4xf32>
// CHECK-NEXT: }

func.func @main3(%arg0: tensor<42x2x20x3xf32>) -> tensor<1x2x7x3x2x3x4x5xf32> {
    %0 = stablehlo.reshape %arg0 : (tensor<42x2x20x3xf32>) -> tensor<7x3x2x1x2x4x5x3xf32>
    %1 = stablehlo.transpose %0, dims = [3, 4, 0, 1, 2, 7, 5, 6] : (tensor<7x3x2x1x2x4x5x3xf32>) -> tensor<1x2x7x3x2x3x4x5xf32>
    return %1 : tensor<1x2x7x3x2x3x4x5xf32>
}

// CHECK: func.func @main3(%arg0: tensor<42x2x20x3xf32>) -> tensor<1x2x7x3x2x3x4x5xf32> {
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [1, 0, 3, 2] : (tensor<42x2x20x3xf32>) -> tensor<2x42x3x20xf32>
// CHECK-NEXT:     %1 = stablehlo.reshape %0 : (tensor<2x42x3x20xf32>) -> tensor<1x2x7x3x2x3x4x5xf32>
// CHECK-NEXT:     return %1 : tensor<1x2x7x3x2x3x4x5xf32>
// CHECK-NEXT: }

func.func @fail1(%arg0: tensor<1x2x20x3xf32>) -> tensor<2x1x4x3x5xf32> {
    %0 = stablehlo.reshape %arg0 : (tensor<1x2x20x3xf32>) -> tensor<1x2x4x5x3xf32>
    %1 = stablehlo.transpose %0, dims = [1, 0, 2, 4, 3] : (tensor<1x2x4x5x3xf32>) -> tensor<2x1x4x3x5xf32>
    return %1 : tensor<2x1x4x3x5xf32>
}

// CHECK: func.func @fail1(%arg0: tensor<1x2x20x3xf32>) -> tensor<2x1x4x3x5xf32> {
// CHECK-NEXT:     %0 = stablehlo.reshape %arg0 : (tensor<1x2x20x3xf32>) -> tensor<1x2x4x5x3xf32>
// CHECK-NEXT:     %1 = stablehlo.transpose %0, dims = [1, 0, 2, 4, 3] : (tensor<1x2x4x5x3xf32>) -> tensor<2x1x4x3x5xf32>
// CHECK-NEXT:     return %1 : tensor<2x1x4x3x5xf32>
// CHECK-NEXT: }

func.func @fail2(%arg0: tensor<1x2x20x3xf32>) -> tensor<1x60x2xf32> {
    %0 = stablehlo.reshape %arg0 : (tensor<1x2x20x3xf32>) -> tensor<1x2x60xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1] : (tensor<1x2x60xf32>) -> tensor<1x60x2xf32>
    return %1 : tensor<1x60x2xf32>
}

// CHECK: func.func @fail2(%arg0: tensor<1x2x20x3xf32>) -> tensor<1x60x2xf32> {
// CHECK-NEXT:     %0 = stablehlo.reshape %arg0 : (tensor<1x2x20x3xf32>) -> tensor<1x2x60xf32>
// CHECK-NEXT:     %1 = stablehlo.transpose %0, dims = [0, 2, 1] : (tensor<1x2x60xf32>) -> tensor<1x60x2xf32>
// CHECK-NEXT:     return %1 : tensor<1x60x2xf32>
// CHECK-NEXT: }
