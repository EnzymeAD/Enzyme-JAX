// RUN: enzymexlamlir-opt --enzyme-hlo-opt --enzyme-hlo-opt="passses=65536" --enzyme-hlo-opt="passses=131072" %s | FileCheck %s

func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<32x64xf32>, %arg3: tensor<64x32xf32>) -> tensor<64x64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.transpose %arg2, dims = [1, 0] : (tensor<32x64xf32>) -> tensor<64x32xf32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<64x32xf32>
    %2 = stablehlo.multiply %1, %0 : tensor<64x32xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<64x32xf32>) -> tensor<64x32x1x64x1xf32>
    %4 = stablehlo.broadcast_in_dim %arg3, dims = [3, 1] : (tensor<64x32xf32>) -> tensor<64x32x1x64x1xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x32x1x64x1xf32>
    %6 = stablehlo.reshape %5 : (tensor<64x32x1x64x1xf32>) -> tensor<64x32x64x1xf32>
    %7 = stablehlo.reduce(%6 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x32x64x1xf32>, tensor<f32>) -> tensor<64x64x1xf32>
    %8 = stablehlo.multiply %cst, %arg1 : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<64x64x1xf32>
    %10 = stablehlo.add %7, %9 : tensor<64x64x1xf32>
    %11 = stablehlo.reshape %10 : (tensor<64x64x1xf32>) -> tensor<64x64xf32>
    %12 = stablehlo.transpose %11, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
    return %12 : tensor<64x64xf32>
}

// CHECK: func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<32x64xf32>, %arg3: tensor<64x32xf32>) -> tensor<64x64xf32> {
// CHECK-NEXT:    %cst = stablehlo.constant {enzymexla.finite = [#enzymexla<guaranteed GUARANTEED>]} dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %0 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [0] x [1] {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<32x64xf32>, tensor<64x32xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:    %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:    %2 = stablehlo.broadcast_in_dim %arg0, dims = [] {enzymexla.symmetric_matrix = [#enzymexla<guaranteed GUARANTEED>]} : (tensor<f32>) -> tensor<64x64xf32>
// CHECK-NEXT:    %3 = stablehlo.multiply %2, %1 : tensor<64x64xf32>
// CHECK-NEXT:    %4 = stablehlo.multiply %cst, %arg1 {enzymexla.no_nan = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<f32>
// CHECK-NEXT:    %5 = stablehlo.broadcast_in_dim %4, dims = [] {enzymexla.symmetric_matrix = [#enzymexla<guaranteed GUARANTEED>]} : (tensor<f32>) -> tensor<64x64xf32>
// CHECK-NEXT:    %6 = stablehlo.add %3, %5 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<64x64xf32>
// CHECK-NEXT:    return %6 : tensor<64x64xf32>
// CHECK-NEXT: }
