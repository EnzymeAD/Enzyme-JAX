// RUN: enzymexlamlir-opt --auto-batching --enzyme-hlo-opt %s | FileCheck %s

func.func @main(%arg0: tensor<3x4xf32>, %arg1: tensor<6x3xf32>) -> (tensor<4x5x6x3xf32>, tensor<4x5x6x3xf32>, tensor<4x5x6x3xf32>, tensor<4x5x6x3xf32>, tensor<4x5x6x3xf32>, tensor<4x5x6x3xf32>) {
    %0 = stablehlo.slice %arg1 [5:6, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %1 = stablehlo.slice %arg1 [4:5, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %2 = stablehlo.slice %arg1 [3:4, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %3 = stablehlo.slice %arg1 [2:3, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %4 = stablehlo.slice %arg1 [1:2, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %5 = stablehlo.slice %arg1 [0:1, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %6 = stablehlo.broadcast_in_dim %0, dims = [0, 3] : (tensor<1x3xf32>) -> tensor<4x5x6x3xf32>
    %7 = stablehlo.broadcast_in_dim %1, dims = [0, 3] : (tensor<1x3xf32>) -> tensor<4x5x6x3xf32>
    %8 = stablehlo.broadcast_in_dim %2, dims = [0, 3] : (tensor<1x3xf32>) -> tensor<4x5x6x3xf32>
    %9 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x3xf32>) -> tensor<4x5x6x3xf32>
    %10 = stablehlo.broadcast_in_dim %4, dims = [0, 3] : (tensor<1x3xf32>) -> tensor<4x5x6x3xf32>
    %11 = stablehlo.broadcast_in_dim %5, dims = [0, 3] : (tensor<1x3xf32>) -> tensor<4x5x6x3xf32>
    return %6, %7, %8, %9, %10, %11 : tensor<4x5x6x3xf32>, tensor<4x5x6x3xf32>, tensor<4x5x6x3xf32>, tensor<4x5x6x3xf32>, tensor<4x5x6x3xf32>, tensor<4x5x6x3xf32>
}

// CHECK:  func.func @main(%arg0: tensor<3x4xf32>, %arg1: tensor<6x3xf32>) -> (tensor<4x5x6x3xf32>, tensor<4x5x6x3xf32>, tensor<4x5x6x3xf32>, tensor<4x5x6x3xf32>, tensor<4x5x6x3xf32>, tensor<4x5x6x3xf32>) {
// CHECK-NEXT:    %0 = stablehlo.reshape %arg1 : (tensor<6x3xf32>) -> tensor<6x1x3xf32>
// CHECK-NEXT:    %1 = stablehlo.slice %0 [0:1, 0:1, 0:3] : (tensor<6x1x3xf32>) -> tensor<1x1x3xf32>
// CHECK-NEXT:    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 4] : (tensor<1x1x3xf32>) -> tensor<1x4x5x6x3xf32>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<1x4x5x6x3xf32>) -> tensor<4x5x6x3xf32>
// CHECK-NEXT:    %4 = stablehlo.slice %0 [1:2, 0:1, 0:3] : (tensor<6x1x3xf32>) -> tensor<1x1x3xf32>
// CHECK-NEXT:    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 4] : (tensor<1x1x3xf32>) -> tensor<1x4x5x6x3xf32>
// CHECK-NEXT:    %6 = stablehlo.reshape %5 : (tensor<1x4x5x6x3xf32>) -> tensor<4x5x6x3xf32>
// CHECK-NEXT:    %7 = stablehlo.slice %0 [2:3, 0:1, 0:3] : (tensor<6x1x3xf32>) -> tensor<1x1x3xf32>
// CHECK-NEXT:    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 4] : (tensor<1x1x3xf32>) -> tensor<1x4x5x6x3xf32>
// CHECK-NEXT:    %9 = stablehlo.reshape %8 : (tensor<1x4x5x6x3xf32>) -> tensor<4x5x6x3xf32>
// CHECK-NEXT:    %10 = stablehlo.slice %0 [3:4, 0:1, 0:3] : (tensor<6x1x3xf32>) -> tensor<1x1x3xf32>
// CHECK-NEXT:    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1, 4] : (tensor<1x1x3xf32>) -> tensor<1x4x5x6x3xf32>
// CHECK-NEXT:    %12 = stablehlo.reshape %11 : (tensor<1x4x5x6x3xf32>) -> tensor<4x5x6x3xf32>
// CHECK-NEXT:    %13 = stablehlo.slice %0 [4:5, 0:1, 0:3] : (tensor<6x1x3xf32>) -> tensor<1x1x3xf32>
// CHECK-NEXT:    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 4] : (tensor<1x1x3xf32>) -> tensor<1x4x5x6x3xf32>
// CHECK-NEXT:    %15 = stablehlo.reshape %14 : (tensor<1x4x5x6x3xf32>) -> tensor<4x5x6x3xf32>
// CHECK-NEXT:    %16 = stablehlo.slice %0 [5:6, 0:1, 0:3] : (tensor<6x1x3xf32>) -> tensor<1x1x3xf32>
// CHECK-NEXT:    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 4] : (tensor<1x1x3xf32>) -> tensor<1x4x5x6x3xf32>
// CHECK-NEXT:    %18 = stablehlo.reshape %17 : (tensor<1x4x5x6x3xf32>) -> tensor<4x5x6x3xf32>
// CHECK-NEXT:    return %18, %15, %12, %9, %6, %3 : tensor<4x5x6x3xf32>, tensor<4x5x6x3xf32>, tensor<4x5x6x3xf32>, tensor<4x5x6x3xf32>, tensor<4x5x6x3xf32>, tensor<4x5x6x3xf32>
// CHECK-NEXT:  }

func.func @main2(%arg0: tensor<3x4xf32>, %arg1: tensor<6x3xf32>) -> (tensor<1x1x3xf32>, tensor<1x1x3xf32>, tensor<1x1x3xf32>, tensor<1x1x3xf32>, tensor<1x1x3xf32>, tensor<1x1x3xf32>) {
    %0 = stablehlo.slice %arg1 [5:6, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %1 = stablehlo.slice %arg1 [4:5, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %2 = stablehlo.slice %arg1 [3:4, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %3 = stablehlo.slice %arg1 [2:3, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %4 = stablehlo.slice %arg1 [1:2, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %5 = stablehlo.slice %arg1 [0:1, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %6 = stablehlo.broadcast_in_dim %0, dims = [0, 2] : (tensor<1x3xf32>) -> tensor<1x1x3xf32>
    %7 = stablehlo.broadcast_in_dim %1, dims = [0, 2] : (tensor<1x3xf32>) -> tensor<1x1x3xf32>
    %8 = stablehlo.broadcast_in_dim %2, dims = [0, 2] : (tensor<1x3xf32>) -> tensor<1x1x3xf32>
    %9 = stablehlo.broadcast_in_dim %3, dims = [0, 2] : (tensor<1x3xf32>) -> tensor<1x1x3xf32>
    %10 = stablehlo.broadcast_in_dim %4, dims = [0, 2] : (tensor<1x3xf32>) -> tensor<1x1x3xf32>
    %11 = stablehlo.broadcast_in_dim %5, dims = [0, 2] : (tensor<1x3xf32>) -> tensor<1x1x3xf32>
    return %6, %7, %8, %9, %10, %11 : tensor<1x1x3xf32>, tensor<1x1x3xf32>, tensor<1x1x3xf32>, tensor<1x1x3xf32>, tensor<1x1x3xf32>, tensor<1x1x3xf32>
}

// CHECK: func.func @main2(%arg0: tensor<3x4xf32>, %arg1: tensor<6x3xf32>) -> (tensor<1x1x3xf32>, tensor<1x1x3xf32>, tensor<1x1x3xf32>, tensor<1x1x3xf32>, tensor<1x1x3xf32>, tensor<1x1x3xf32>) {
// CHECK-NEXT:   %0 = stablehlo.slice %arg1 [5:6, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
// CHECK-NEXT:   %1 = stablehlo.slice %arg1 [4:5, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
// CHECK-NEXT:   %2 = stablehlo.slice %arg1 [3:4, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
// CHECK-NEXT:   %3 = stablehlo.slice %arg1 [2:3, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
// CHECK-NEXT:   %4 = stablehlo.slice %arg1 [1:2, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
// CHECK-NEXT:   %5 = stablehlo.slice %arg1 [0:1, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
// CHECK-NEXT:   %6 = stablehlo.reshape %0 : (tensor<1x3xf32>) -> tensor<1x1x3xf32>
// CHECK-NEXT:   %7 = stablehlo.reshape %1 : (tensor<1x3xf32>) -> tensor<1x1x3xf32>
// CHECK-NEXT:   %8 = stablehlo.reshape %2 : (tensor<1x3xf32>) -> tensor<1x1x3xf32>
// CHECK-NEXT:   %9 = stablehlo.reshape %3 : (tensor<1x3xf32>) -> tensor<1x1x3xf32>
// CHECK-NEXT:   %10 = stablehlo.reshape %4 : (tensor<1x3xf32>) -> tensor<1x1x3xf32>
// CHECK-NEXT:   %11 = stablehlo.reshape %5 : (tensor<1x3xf32>) -> tensor<1x1x3xf32>
// CHECK-NEXT:   return %6, %7, %8, %9, %10, %11 : tensor<1x1x3xf32>, tensor<1x1x3xf32>, tensor<1x1x3xf32>, tensor<1x1x3xf32>, tensor<1x1x3xf32>, tensor<1x1x3xf32>
// CHECK-NEXT: }
