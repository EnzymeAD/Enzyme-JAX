// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<64x64xf32>, %arg3: tensor<64x64xf32>, %arg4: tensor<64x64xf32>, %arg5: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %0 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [0] x [1] : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    %1 = stablehlo.dot_general %0, %arg4, contracting_dims = [1] x [1] : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<64x1x1x64xf32>
    %3 = stablehlo.reshape %1 : (tensor<64x64xf32>) -> tensor<64x1x1x64xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<64x1x1x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg5, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64x1x1xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<64x64x1x1xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x64x1x1xf32>
    %8 = stablehlo.reshape %4 : (tensor<64x1x1x64xf32>) -> tensor<64x64x1x1xf32>
    %9 = stablehlo.add %8, %7 : tensor<64x64x1x1xf32>
    %10 = stablehlo.reshape %9 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<64x64x1x1xf32>) -> tensor<64x64xf32>
    %11 = stablehlo.transpose %10, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
    return %11 : tensor<64x64xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<64x64xf32>, %arg3: tensor<64x64xf32>, %arg4: tensor<64x64xf32>, %arg5: tensor<64x64xf32>) -> tensor<64x64xf32> {
// CHECK-NEXT:   %0 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [0] x [1] : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:   %1 = stablehlo.dot_general %0, %arg4, contracting_dims = [1] x [1] : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:   %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
// CHECK-NEXT:   %3 = stablehlo.multiply %2, %1 : tensor<64x64xf32>
// CHECK-NEXT:   %4 = stablehlo.transpose %arg5, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:   %5 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
// CHECK-NEXT:   %6 = stablehlo.multiply %4, %5 : tensor<64x64xf32>
// CHECK-NEXT:   %7 = stablehlo.add %3, %6 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<64x64xf32>
// CHECK-NEXT:   %8 = stablehlo.transpose %7, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:   return %8 : tensor<64x64xf32>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<64x64xf32>, %arg3: tensor<64x64xf32>, %arg4: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reshape %arg3 : (tensor<64x64xf32>) -> tensor<64x64x1x1xf32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<64x64x1x1xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<64x64x1x1xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %4 = stablehlo.multiply %arg2, %3 : tensor<64x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64x64x1x1xf32>
    %6 = stablehlo.broadcast_in_dim %2, dims = [1, 2, 3, 4] : (tensor<64x64x1x1xf32>) -> tensor<64x64x64x1x1xf32>
    %7 = stablehlo.multiply %6, %5 : tensor<64x64x64x1x1xf32>
    %8 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<64x64xf32>) -> tensor<64x64x64x1x1xf32>
    %9 = stablehlo.broadcast_in_dim %arg3, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64x64x1x1xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<64x64x64x1x1xf32>
    %11 = stablehlo.add %10, %7 : tensor<64x64x64x1x1xf32>
    %12 = stablehlo.reshape %11 : (tensor<64x64x64x1x1xf32>) -> tensor<64x1x64x64x1x1xf32>
    %13 = stablehlo.reduce(%12 init: %cst) applies stablehlo.add across dimensions = [2, 1, 5] : (tensor<64x1x64x64x1x1xf32>, tensor<f32>) -> tensor<64x64x1xf32>
    %14 = stablehlo.broadcast_in_dim %arg4, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64x1xf32>
    %15 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<64x64x1xf32>
    %16 = stablehlo.multiply %14, %15 : tensor<64x64x1xf32>
    %17 = stablehlo.add %13, %16 : tensor<64x64x1xf32>
    %18 = stablehlo.reshape %17 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<64x64x1xf32>) -> tensor<64x64xf32>
    %19 = stablehlo.transpose %18, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
    return %19 : tensor<64x64xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<64x64xf32>, %arg3: tensor<64x64xf32>, %arg4: tensor<64x64xf32>) -> tensor<64x64xf32> {
// CHECK-NEXT:   %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
// CHECK-NEXT:   %1 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [0] x [0] : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:   %2 = stablehlo.multiply %0, %1 : tensor<64x64xf32>
// CHECK-NEXT:   %3 = stablehlo.dot_general %arg3, %arg2, contracting_dims = [0] x [0] : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:   %4 = stablehlo.multiply %0, %3 : tensor<64x64xf32>
// CHECK-NEXT:   %5 = stablehlo.add %4, %2 : tensor<64x64xf32>
// CHECK-NEXT:   %6 = stablehlo.transpose %arg4, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:   %7 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
// CHECK-NEXT:   %8 = stablehlo.multiply %6, %7 : tensor<64x64xf32>
// CHECK-NEXT:   %9 = stablehlo.add %5, %8 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<64x64xf32>
// CHECK-NEXT:   %10 = stablehlo.transpose %9, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:   return %10 : tensor<64x64xf32>
// CHECK-NEXT: }
