// RUN: enzymexlamlir-opt --auto-batching --inline --enzyme-hlo-opt %s | FileCheck %s

func.func @main(%arg0: tensor<3x4x2xf32>, %arg1: tensor<6x3x5xf32>) -> (tensor<5x3x1xf32>, tensor<5x3x1xf32>, tensor<5x3x1xf32>, tensor<5x3x1xf32>, tensor<5x3x1xf32>, tensor<5x3x1xf32>) {
    %0 = stablehlo.slice %arg1 [5:6, 0:3, 0:5] : (tensor<6x3x5xf32>) -> tensor<1x3x5xf32>
    %1 = stablehlo.slice %arg1 [4:5, 0:3, 0:5] : (tensor<6x3x5xf32>) -> tensor<1x3x5xf32>
    %2 = stablehlo.slice %arg1 [3:4, 0:3, 0:5] : (tensor<6x3x5xf32>) -> tensor<1x3x5xf32>
    %3 = stablehlo.slice %arg1 [2:3, 0:3, 0:5] : (tensor<6x3x5xf32>) -> tensor<1x3x5xf32>
    %4 = stablehlo.slice %arg1 [1:2, 0:3, 0:5] : (tensor<6x3x5xf32>) -> tensor<1x3x5xf32>
    %5 = stablehlo.slice %arg1 [0:1, 0:3, 0:5] : (tensor<6x3x5xf32>) -> tensor<1x3x5xf32>
    %6 = stablehlo.transpose %0, dims = [2, 1, 0] : (tensor<1x3x5xf32>) -> tensor<5x3x1xf32>
    %7 = stablehlo.transpose %1, dims = [2, 1, 0] : (tensor<1x3x5xf32>) -> tensor<5x3x1xf32>
    %8 = stablehlo.transpose %2, dims = [2, 1, 0] : (tensor<1x3x5xf32>) -> tensor<5x3x1xf32>
    %9 = stablehlo.transpose %3, dims = [2, 1, 0] : (tensor<1x3x5xf32>) -> tensor<5x3x1xf32>
    %10 = stablehlo.transpose %4, dims = [2, 1, 0] : (tensor<1x3x5xf32>) -> tensor<5x3x1xf32>
    %11 = stablehlo.transpose %5, dims = [2, 1, 0] : (tensor<1x3x5xf32>) -> tensor<5x3x1xf32>
    return %6, %7, %8, %9, %10, %11 : tensor<5x3x1xf32>, tensor<5x3x1xf32>, tensor<5x3x1xf32>, tensor<5x3x1xf32>, tensor<5x3x1xf32>, tensor<5x3x1xf32>
}

// CHECK: func.func @main(%arg0: tensor<3x4x2xf32>, %arg1: tensor<6x3x5xf32>) -> (tensor<5x3x1xf32>, tensor<5x3x1xf32>, tensor<5x3x1xf32>, tensor<5x3x1xf32>, tensor<5x3x1xf32>, tensor<5x3x1xf32>) {
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 2, 1] : (tensor<6x3x5xf32>) -> tensor<6x5x3x1xf32>
// CHECK-NEXT:    %1 = stablehlo.slice %0 [0:1, 0:5, 0:3, 0:1] : (tensor<6x5x3x1xf32>) -> tensor<1x5x3x1xf32>
// CHECK-NEXT:    %2 = stablehlo.reshape %1 : (tensor<1x5x3x1xf32>) -> tensor<5x3x1xf32>
// CHECK-NEXT:    %3 = stablehlo.slice %0 [1:2, 0:5, 0:3, 0:1] : (tensor<6x5x3x1xf32>) -> tensor<1x5x3x1xf32>
// CHECK-NEXT:    %4 = stablehlo.reshape %3 : (tensor<1x5x3x1xf32>) -> tensor<5x3x1xf32>
// CHECK-NEXT:    %5 = stablehlo.slice %0 [2:3, 0:5, 0:3, 0:1] : (tensor<6x5x3x1xf32>) -> tensor<1x5x3x1xf32>
// CHECK-NEXT:    %6 = stablehlo.reshape %5 : (tensor<1x5x3x1xf32>) -> tensor<5x3x1xf32>
// CHECK-NEXT:    %7 = stablehlo.slice %0 [3:4, 0:5, 0:3, 0:1] : (tensor<6x5x3x1xf32>) -> tensor<1x5x3x1xf32>
// CHECK-NEXT:    %8 = stablehlo.reshape %7 : (tensor<1x5x3x1xf32>) -> tensor<5x3x1xf32>
// CHECK-NEXT:    %9 = stablehlo.slice %0 [4:5, 0:5, 0:3, 0:1] : (tensor<6x5x3x1xf32>) -> tensor<1x5x3x1xf32>
// CHECK-NEXT:    %10 = stablehlo.reshape %9 : (tensor<1x5x3x1xf32>) -> tensor<5x3x1xf32>
// CHECK-NEXT:    %11 = stablehlo.slice %0 [5:6, 0:5, 0:3, 0:1] : (tensor<6x5x3x1xf32>) -> tensor<1x5x3x1xf32>
// CHECK-NEXT:    %12 = stablehlo.reshape %11 : (tensor<1x5x3x1xf32>) -> tensor<5x3x1xf32>
// CHECK-NEXT:    return %12, %10, %8, %6, %4, %2 : tensor<5x3x1xf32>, tensor<5x3x1xf32>, tensor<5x3x1xf32>, tensor<5x3x1xf32>, tensor<5x3x1xf32>, tensor<5x3x1xf32>
// CHECK-NEXT:  }
