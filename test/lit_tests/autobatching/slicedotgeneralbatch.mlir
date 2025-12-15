// RUN: enzymexlamlir-opt --auto-batching --inline --enzyme-hlo-opt %s | FileCheck %s

func.func @main(%arg0: tensor<3x4xf32>, %arg1: tensor<6x3xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
    %5 = stablehlo.slice %arg1 [0:1, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %6 = stablehlo.dot_general %arg0, %5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<1x3xf32>) -> tensor<4x1xf32>
    %7 = stablehlo.reshape %6 : (tensor<4x1xf32>) -> tensor<4xf32>

    %cst = stablehlo.constant dense<1.000000e+00> : tensor<3x4xf32>
    %pow1 = stablehlo.power %arg0, %arg0 : tensor<3x4xf32>
    %sub1 = stablehlo.subtract %arg0, %cst : tensor<3x4xf32>
    %add1 = stablehlo.add %arg0, %sub1 : tensor<3x4xf32>
    %4 = stablehlo.slice %arg1 [1:2, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %8 = stablehlo.dot_general %add1, %4, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<1x3xf32>) -> tensor<4x1xf32>
    %9 = stablehlo.reshape %8 : (tensor<4x1xf32>) -> tensor<4xf32>

    %2 = stablehlo.slice %arg1 [3:4, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>

    %3 = stablehlo.slice %arg1 [2:3, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %10 = stablehlo.dot_general %pow1, %3, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<1x3xf32>) -> tensor<4x1xf32>
    %11 = stablehlo.reshape %10 : (tensor<4x1xf32>) -> tensor<4xf32>

    %1 = stablehlo.slice %arg1 [4:5, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %14 = stablehlo.dot_general %arg0, %1, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<1x3xf32>) -> tensor<4x1xf32>
    %15 = stablehlo.reshape %14 : (tensor<4x1xf32>) -> tensor<4xf32>

    %12 = stablehlo.dot_general %arg0, %2, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<1x3xf32>) -> tensor<4x1xf32>
    %13 = stablehlo.reshape %12 : (tensor<4x1xf32>) -> tensor<4xf32>

    %0 = stablehlo.slice %arg1 [5:6, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %16 = stablehlo.dot_general %arg0, %0, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<1x3xf32>) -> tensor<4x1xf32>
    %17 = stablehlo.reshape %16 : (tensor<4x1xf32>) -> tensor<4xf32>

    return %7, %9, %11, %13, %15, %17 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
}

// CHECK: func.func @main(%arg0: tensor<3x4xf32>, %arg1: tensor<6x3xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<3x4xf32>
// CHECK-NEXT:    %0 = stablehlo.power %arg0, %arg0 : tensor<3x4xf32>
// CHECK-NEXT:    %1 = stablehlo.subtract %arg0, %cst : tensor<3x4xf32>
// CHECK-NEXT:    %2 = stablehlo.add %arg0, %1 : tensor<3x4xf32>
// CHECK-NEXT:    %3 = stablehlo.reshape %arg0 : (tensor<3x4xf32>) -> tensor<1x3x4xf32>
// CHECK-NEXT:    %4 = stablehlo.reshape %2 : (tensor<3x4xf32>) -> tensor<1x3x4xf32>
// CHECK-NEXT:    %5 = stablehlo.reshape %0 : (tensor<3x4xf32>) -> tensor<1x3x4xf32>
// CHECK-NEXT:    %6 = stablehlo.concatenate %3, %4, %5, %3, %3, %3, dim = 0 : (tensor<1x3x4xf32>, tensor<1x3x4xf32>, tensor<1x3x4xf32>, tensor<1x3x4xf32>, tensor<1x3x4xf32>, tensor<1x3x4xf32>) -> tensor<6x3x4xf32>
// CHECK-NEXT:    %7 = stablehlo.dot_general %6, %arg1, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6x3x4xf32>, tensor<6x3xf32>) -> tensor<6x4xf32>
// CHECK-NEXT:    %8 = stablehlo.slice %7 [0:1, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:    %9 = stablehlo.slice %7 [1:2, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:    %10 = stablehlo.slice %7 [2:3, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:    %11 = stablehlo.slice %7 [3:4, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:    %12 = stablehlo.slice %7 [4:5, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:    %13 = stablehlo.slice %7 [5:6, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:    %14 = stablehlo.reshape %8 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %15 = stablehlo.reshape %9 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %16 = stablehlo.reshape %10 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %17 = stablehlo.reshape %12 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %18 = stablehlo.reshape %11 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %19 = stablehlo.reshape %13 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    return %14, %15, %16, %18, %17, %19 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
// CHECK-NEXT: }

func.func @main2(%arg0: tensor<3x4xf32>, %arg1: tensor<6x3xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
    %0 = stablehlo.slice %arg1 [5:6, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %1 = stablehlo.slice %arg1 [4:5, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %2 = stablehlo.slice %arg1 [3:4, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %3 = stablehlo.slice %arg1 [2:3, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %4 = stablehlo.slice %arg1 [1:2, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %5 = stablehlo.slice %arg1 [0:1, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %6 = stablehlo.dot_general %arg0, %5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<1x3xf32>) -> tensor<4x1xf32>
    %8 = stablehlo.dot_general %arg0, %4, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<1x3xf32>) -> tensor<4x1xf32>
    %10 = stablehlo.dot_general %arg0, %3, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<1x3xf32>) -> tensor<4x1xf32>
    %12 = stablehlo.dot_general %arg0, %2, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<1x3xf32>) -> tensor<4x1xf32>
    %14 = stablehlo.dot_general %arg0, %1, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<1x3xf32>) -> tensor<4x1xf32>
    %16 = stablehlo.dot_general %arg0, %0, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<1x3xf32>) -> tensor<4x1xf32>
    %7 = stablehlo.reshape %6 : (tensor<4x1xf32>) -> tensor<4xf32>
    %9 = stablehlo.reshape %8 : (tensor<4x1xf32>) -> tensor<4xf32>
    %11 = stablehlo.reshape %10 : (tensor<4x1xf32>) -> tensor<4xf32>
    %13 = stablehlo.reshape %12 : (tensor<4x1xf32>) -> tensor<4xf32>
    %15 = stablehlo.reshape %14 : (tensor<4x1xf32>) -> tensor<4xf32>
    %17 = stablehlo.reshape %16 : (tensor<4x1xf32>) -> tensor<4xf32>
    return %7, %9, %11, %13, %15, %17 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
}

// CHECK: func.func @main2(%arg0: tensor<3x4xf32>, %arg1: tensor<6x3xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
// CHECK-NEXT:     %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] : (tensor<3x4xf32>) -> tensor<6x3x4xf32>
// CHECK-NEXT:     %1 = stablehlo.dot_general %0, %arg1, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6x3x4xf32>, tensor<6x3xf32>) -> tensor<6x4xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:1, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:     %3 = stablehlo.slice %1 [1:2, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:     %4 = stablehlo.slice %1 [2:3, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:     %5 = stablehlo.slice %1 [3:4, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:     %6 = stablehlo.slice %1 [4:5, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:     %7 = stablehlo.slice %1 [5:6, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:     %8 = stablehlo.reshape %2 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:     %9 = stablehlo.reshape %3 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:     %10 = stablehlo.reshape %4 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:     %11 = stablehlo.reshape %5 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:     %12 = stablehlo.reshape %6 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:     %13 = stablehlo.reshape %7 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:     return %8, %9, %10, %11, %12, %13 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
// CHECK-NEXT: }
