// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main1(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<64x64xf32>, %arg3: tensor<64x64xf32>, %arg4: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = stablehlo.reshape %arg3 : (tensor<64x64xf32>) -> tensor<64x64x1x1xf32>
  %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<64x64x1x1xf32>
  %2 = stablehlo.multiply %0, %1 : tensor<64x64x1x1xf32>
  %3 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
  %4 = stablehlo.multiply %arg2, %3 : tensor<64x64xf32>
  %5 = stablehlo.broadcast_in_dim %2, dims = [2, 3, 4, 5] : (tensor<64x64x1x1xf32>) -> tensor<64x1x64x64x1x1xf32>
  %6 = stablehlo.broadcast_in_dim %arg2, dims = [2, 0] : (tensor<64x64xf32>) -> tensor<64x1x64x64x1x1xf32>
  %7 = stablehlo.multiply %5, %6 : tensor<64x1x64x64x1x1xf32>
  %8 = stablehlo.broadcast_in_dim %4, dims = [2, 3] : (tensor<64x64xf32>) -> tensor<64x1x64x64x1x1xf32>
  %9 = stablehlo.broadcast_in_dim %arg3, dims = [2, 0] : (tensor<64x64xf32>) -> tensor<64x1x64x64x1x1xf32>
  %10 = stablehlo.multiply %8, %9 : tensor<64x1x64x64x1x1xf32>
  %11 = stablehlo.add %10, %7 : tensor<64x1x64x64x1x1xf32>
  %12 = stablehlo.transpose %arg4, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
  %13 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
  %14 = stablehlo.multiply %12, %13 : tensor<64x64xf32>
  %15 = stablehlo.reduce(%11 init: %cst) applies stablehlo.add across dimensions = [1, 2, 4, 5] : (tensor<64x1x64x64x1x1xf32>, tensor<f32>) -> tensor<64x64xf32>
  %16 = stablehlo.add %15, %14 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<64x64xf32>
  %17 = stablehlo.transpose %16, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
  return %17 : tensor<64x64xf32>
}

// CHECK: func.func @main1(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<64x64xf32>, %arg3: tensor<64x64xf32>, %arg4: tensor<64x64xf32>) -> tensor<64x64xf32> {
// CHECK-NEXT:   %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
// CHECK-NEXT:   %1 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [0] x [0] : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:   %2 = stablehlo.multiply %0, %1 : tensor<64x64xf32>
// CHECK-NEXT:   %3 = stablehlo.dot_general %arg3, %arg2, contracting_dims = [0] x [0] : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:   %4 = stablehlo.multiply %0, %3 : tensor<64x64xf32>
// CHECK-NEXT:   %5 = stablehlo.transpose %arg4, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:   %6 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
// CHECK-NEXT:   %7 = stablehlo.multiply %5, %6 : tensor<64x64xf32>
// CHECK-NEXT:   %8 = stablehlo.add %4, %2 : tensor<64x64xf32>
// CHECK-NEXT:   %9 = stablehlo.add %8, %7 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<64x64xf32>
// CHECK-NEXT:   %10 = stablehlo.transpose %9, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:   return %10 : tensor<64x64xf32>
// CHECK-NEXT: }

// one of lhs/rhs has multiple uses
func.func @main2(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<64x64xf32>, %arg3: tensor<64x64xf32>, %arg4: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x1x64x64x1x1xf32>) {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = stablehlo.reshape %arg3 : (tensor<64x64xf32>) -> tensor<64x64x1x1xf32>
  %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<64x64x1x1xf32>
  %2 = stablehlo.multiply %0, %1 : tensor<64x64x1x1xf32>
  %3 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
  %4 = stablehlo.multiply %arg2, %3 : tensor<64x64xf32>
  %5 = stablehlo.broadcast_in_dim %2, dims = [2, 3, 4, 5] : (tensor<64x64x1x1xf32>) -> tensor<64x1x64x64x1x1xf32>
  %6 = stablehlo.broadcast_in_dim %arg2, dims = [2, 0] : (tensor<64x64xf32>) -> tensor<64x1x64x64x1x1xf32>
  %7 = stablehlo.multiply %5, %6 : tensor<64x1x64x64x1x1xf32>
  %8 = stablehlo.broadcast_in_dim %4, dims = [2, 3] : (tensor<64x64xf32>) -> tensor<64x1x64x64x1x1xf32>
  %9 = stablehlo.broadcast_in_dim %arg3, dims = [2, 0] : (tensor<64x64xf32>) -> tensor<64x1x64x64x1x1xf32>
  %10 = stablehlo.multiply %8, %9 : tensor<64x1x64x64x1x1xf32>
  %11 = stablehlo.add %10, %7 : tensor<64x1x64x64x1x1xf32>
  %12 = stablehlo.transpose %arg4, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
  %13 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
  %14 = stablehlo.multiply %12, %13 : tensor<64x64xf32>
  %15 = stablehlo.reduce(%11 init: %cst) applies stablehlo.add across dimensions = [1, 2, 4, 5] : (tensor<64x1x64x64x1x1xf32>, tensor<f32>) -> tensor<64x64xf32>
  %16 = stablehlo.add %15, %14 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<64x64xf32>
  %17 = stablehlo.transpose %16, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
  return %17, %10 : tensor<64x64xf32>, tensor<64x1x64x64x1x1xf32>
}

// CHECK: func.func @main2(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<64x64xf32>, %arg3: tensor<64x64xf32>, %arg4: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x1x64x64x1x1xf32>) {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:   %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
// CHECK-NEXT:   %1 = stablehlo.multiply %arg2, %0 : tensor<64x64xf32>
// CHECK-NEXT:   %2 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [0] x [0] : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:   %3 = stablehlo.multiply %0, %2 : tensor<64x64xf32>
// CHECK-NEXT:   %4 = stablehlo.broadcast_in_dim %1, dims = [2, 3] : (tensor<64x64xf32>) -> tensor<64x1x64x64x1x1xf32>
// CHECK-NEXT:   %5 = stablehlo.broadcast_in_dim %arg3, dims = [2, 0] : (tensor<64x64xf32>) -> tensor<64x1x64x64x1x1xf32>
// CHECK-NEXT:   %6 = stablehlo.multiply %4, %5 : tensor<64x1x64x64x1x1xf32>
// CHECK-NEXT:   %7 = stablehlo.reduce(%6 init: %cst) applies stablehlo.add across dimensions = [1, 2, 4, 5] : (tensor<64x1x64x64x1x1xf32>, tensor<f32>) -> tensor<64x64xf32>
// CHECK-NEXT:   %8 = stablehlo.transpose %arg4, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:   %9 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
// CHECK-NEXT:   %10 = stablehlo.multiply %8, %9 : tensor<64x64xf32>
// CHECK-NEXT:   %11 = stablehlo.add %7, %3 : tensor<64x64xf32>
// CHECK-NEXT:   %12 = stablehlo.add %11, %10 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<64x64xf32>
// CHECK-NEXT:   %13 = stablehlo.transpose %12, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:   return %13, %6 : tensor<64x64xf32>, tensor<64x1x64x64x1x1xf32>
// CHECK-NEXT: }

// lhs == rhs
func.func @main3(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<64x64xf32>, %arg3: tensor<64x64xf32>, %arg4: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
  %1 = stablehlo.multiply %arg2, %0 : tensor<64x64xf32>
  %2 = stablehlo.broadcast_in_dim %1, dims = [2, 3] : (tensor<64x64xf32>) -> tensor<64x1x64x64x1x1xf32>
  %3 = stablehlo.broadcast_in_dim %arg3, dims = [2, 0] : (tensor<64x64xf32>) -> tensor<64x1x64x64x1x1xf32>
  %4 = stablehlo.multiply %2, %3 : tensor<64x1x64x64x1x1xf32>
  %5 = stablehlo.add %4, %4 : tensor<64x1x64x64x1x1xf32>
  %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [1, 2, 4, 5] : (tensor<64x1x64x64x1x1xf32>, tensor<f32>) -> tensor<64x64xf32>
  return %6 : tensor<64x64xf32>
}

// CHECK: func.func @main3(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<64x64xf32>, %arg3: tensor<64x64xf32>, %arg4: tensor<64x64xf32>) -> tensor<64x64xf32> {
// CHECK-NEXT:   %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
// CHECK-NEXT:   %1 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [0] x [0] : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:   %2 = stablehlo.multiply %0, %1 : tensor<64x64xf32>
// CHECK-NEXT:   %3 = stablehlo.add %2, %2 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<64x64xf32>
// CHECK-NEXT:   %4 = stablehlo.transpose %3, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:   return %4 : tensor<64x64xf32>
// CHECK-NEXT: }
