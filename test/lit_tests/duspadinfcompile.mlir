// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=dus_pad;dynamic_pad_to_pad" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

func.func @negative_mul_indexing(%arg0: tensor<5x20x6xf32>) -> (tensor<5x20x6xf32>, tensor<5x20x6xf32>) {
  %c = stablehlo.constant dense<13> : tensor<i32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %c_0 = stablehlo.constant dense<2> : tensor<i32>
  %c_1 = stablehlo.constant dense<1> : tensor<i32>
  %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<5x20x6xf32>) -> tensor<6x20x5xf32>
  %1 = stablehlo.slice %0 [0:4, 4:17:6, 0:2] : (tensor<6x20x5xf32>) -> tensor<4x3x2xf32>
  %2 = stablehlo.reverse %1, dims = [1] : tensor<4x3x2xf32>
  %3 = stablehlo.broadcast_in_dim %2, dims = [3, 0, 1] : (tensor<4x3x2xf32>) -> tensor<3x2x1x4xf32>
  %5 = stablehlo.reshape %3 : (tensor<3x2x1x4xf32>) -> tensor<3x2x4xf32>
  %6 = stablehlo.broadcast_in_dim %5, dims = [0, 2, 1] : (tensor<3x2x4xf32>) -> tensor<3x4x2xf32>
  %7 = stablehlo.slice %6 [0:3, 0:4, 0:2] : (tensor<3x4x2xf32>) -> tensor<3x4x2xf32>
  %9 = stablehlo.slice %7 [0:1, 0:4, 0:2] : (tensor<3x4x2xf32>) -> tensor<1x4x2xf32>
  %10 = stablehlo.reshape %9 : (tensor<1x4x2xf32>) -> tensor<4x2xf32>
  %11 = stablehlo.cosine %10 : tensor<4x2xf32>
  %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<4x2xf32>) -> tensor<4x1x2xf32>
  %13 = stablehlo.pad %12, %cst, low = [1, 17, 2], high = [1, 2, 1], interior = [0, 0, 0] : (tensor<4x1x2xf32>, tensor<f32>) -> tensor<6x20x5xf32>
  %14 = stablehlo.slice %6 [1:2, 0:4, 0:2] : (tensor<3x4x2xf32>) -> tensor<1x4x2xf32>
  %15 = stablehlo.reshape %14 : (tensor<1x4x2xf32>) -> tensor<4x2xf32>
  %17 = stablehlo.broadcast_in_dim %15, dims = [0, 2] : (tensor<4x2xf32>) -> tensor<4x1x2xf32>
  %18 = stablehlo.dynamic_update_slice %13, %17, %c_1, %c, %c_0 : (tensor<6x20x5xf32>, tensor<4x1x2xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x20x5xf32>
  %19 = stablehlo.transpose %18, dims = [2, 1, 0] : (tensor<6x20x5xf32>) -> tensor<5x20x6xf32>
  %20 = stablehlo.transpose %0, dims = [2, 1, 0] : (tensor<6x20x5xf32>) -> tensor<5x20x6xf32>
  return %19, %20 : tensor<5x20x6xf32>, tensor<5x20x6xf32>
}

// CHECK: func.func @negative_mul_indexing(%arg0: tensor<5x20x6xf32>) -> (tensor<5x20x6xf32>, tensor<5x20x6xf32>) {
// CHECK-NEXT:   %c = stablehlo.constant dense<13> : tensor<i32>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<2> : tensor<i32>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<5x20x6xf32>) -> tensor<6x20x5xf32>
// CHECK-NEXT:   %1 = stablehlo.slice %0 [0:4, 4:17:6, 0:2] : (tensor<6x20x5xf32>) -> tensor<4x3x2xf32>
// CHECK-NEXT:   %2 = stablehlo.reverse %1, dims = [1] : tensor<4x3x2xf32>
// CHECK-NEXT:   %3 = stablehlo.broadcast_in_dim %2, dims = [3, 0, 1] : (tensor<4x3x2xf32>) -> tensor<3x2x1x4xf32>
// CHECK-NEXT:   %4 = stablehlo.reshape %3 : (tensor<3x2x1x4xf32>) -> tensor<3x2x4xf32>
// CHECK-NEXT:   %5 = stablehlo.broadcast_in_dim %4, dims = [0, 2, 1] : (tensor<3x2x4xf32>) -> tensor<3x4x2xf32>
// CHECK-NEXT:   %6 = stablehlo.slice %5 [0:3, 0:4, 0:2] : (tensor<3x4x2xf32>) -> tensor<3x4x2xf32>
// CHECK-NEXT:   %7 = stablehlo.slice %6 [0:1, 0:4, 0:2] : (tensor<3x4x2xf32>) -> tensor<1x4x2xf32>
// CHECK-NEXT:   %8 = stablehlo.reshape %7 : (tensor<1x4x2xf32>) -> tensor<4x2xf32>
// CHECK-NEXT:   %9 = stablehlo.cosine %8 : tensor<4x2xf32>
// CHECK-NEXT:   %10 = stablehlo.broadcast_in_dim %9, dims = [0, 2] : (tensor<4x2xf32>) -> tensor<4x1x2xf32>
// CHECK-NEXT:   %11 = stablehlo.pad %10, %cst, low = [1, 17, 2], high = [1, 2, 1], interior = [0, 0, 0] : (tensor<4x1x2xf32>, tensor<f32>) -> tensor<6x20x5xf32>
// CHECK-NEXT:   %12 = stablehlo.slice %5 [1:2, 0:4, 0:2] : (tensor<3x4x2xf32>) -> tensor<1x4x2xf32>
// CHECK-NEXT:   %13 = stablehlo.reshape %12 : (tensor<1x4x2xf32>) -> tensor<4x2xf32>
// CHECK-NEXT:   %14 = stablehlo.broadcast_in_dim %13, dims = [0, 2] : (tensor<4x2xf32>) -> tensor<4x1x2xf32>
// CHECK-NEXT:   %15 = stablehlo.dynamic_update_slice %11, %14, %c_1, %c, %c_0 : (tensor<6x20x5xf32>, tensor<4x1x2xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x20x5xf32>
// CHECK-NEXT:   %16 = stablehlo.transpose %15, dims = [2, 1, 0] : (tensor<6x20x5xf32>) -> tensor<5x20x6xf32>
// CHECK-NEXT:   %17 = stablehlo.transpose %0, dims = [2, 1, 0] : (tensor<6x20x5xf32>) -> tensor<5x20x6xf32>
// CHECK-NEXT:   return %16, %17 : tensor<5x20x6xf32>, tensor<5x20x6xf32>
// CHECK-NEXT: }
