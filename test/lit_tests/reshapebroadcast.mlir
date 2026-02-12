// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%a : tensor<f32>) -> tensor<1x2x1x1xf32> {
    %bc = stablehlo.broadcast_in_dim %a, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %conv = stablehlo.reshape %bc : (tensor<2xf32>) -> tensor<1x2x1x1xf32>
    return %conv : tensor<1x2x1x1xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<f32>) -> tensor<1x2x1x1xf32> {
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<1x2x1x1xf32>
// CHECK-NEXT:    return %0 : tensor<1x2x1x1xf32>
// CHECK-NEXT:  }

module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64x64xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64xf32>) -> (tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg6, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64x1x1xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<64xf32>) -> tensor<64x64x1x1xf32>
    %2 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<64x64x1x1xf32>
    %3 = stablehlo.reshape %1 : (tensor<64x64x1x1xf32>) -> tensor<64x64xf32>
    %4 = stablehlo.reshape %2 : (tensor<64x64x1x1xf32>) -> tensor<64x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x64xf32>
    %6 = stablehlo.reshape %0 : (tensor<64x64x1x1xf32>) -> tensor<64x64xf32>
    %7 = stablehlo.add %6, %5 : tensor<64x64xf32>
    %8 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<64xf32>) -> tensor<64x64x1x1xf32>
    %9 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<64xf32>) -> tensor<64x64x1x1xf32>
    %10 = stablehlo.reshape %8 : (tensor<64x64x1x1xf32>) -> tensor<64x64xf32>
    %11 = stablehlo.reshape %9 : (tensor<64x64x1x1xf32>) -> tensor<64x64xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<64x64xf32>
    %13 = stablehlo.add %7, %12 : tensor<64x64xf32>
    %14 = stablehlo.transpose %13, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %15 = stablehlo.dot_general %14, %arg8, contracting_dims = [1] x [0] : (tensor<64x64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %16 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %17 = stablehlo.multiply %16, %15 : tensor<64xf32>
    %18 = stablehlo.add %17, %arg7 : tensor<64xf32>
    %19 = stablehlo.add %18, %arg9 : tensor<64xf32>
    %20 = stablehlo.dot_general %13, %19, contracting_dims = [1] x [0] : (tensor<64x64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %21 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %22 = stablehlo.multiply %21, %20 : tensor<64xf32>
    return %22, %14, %19 : tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64x64xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64xf32>) -> (tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>) {
// CHECK-NEXT:     %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:     %1 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:     %2 = stablehlo.multiply %0, %1 : tensor<64x64xf32>
// CHECK-NEXT:     %3 = stablehlo.transpose %arg6, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:     %4 = stablehlo.add %3, %2 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<64x64xf32>
// CHECK-NEXT:     %5 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:     %6 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:     %7 = stablehlo.multiply %5, %6 : tensor<64x64xf32>
// CHECK-NEXT:     %8 = stablehlo.add %4, %7 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<64x64xf32>
// CHECK-NEXT:     %9 = stablehlo.transpose %8, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:     %10 = stablehlo.dot_general %8, %arg8, contracting_dims = [0] x [0] : (tensor<64x64xf32>, tensor<64xf32>) -> tensor<64xf32>
// CHECK-NEXT:     %11 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<64xf32>
// CHECK-NEXT:     %12 = stablehlo.multiply %11, %10 : tensor<64xf32>
// CHECK-NEXT:     %13 = stablehlo.add %12, %arg7 : tensor<64xf32>
// CHECK-NEXT:     %14 = stablehlo.add %13, %arg9 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<64xf32>
// CHECK-NEXT:     %15 = stablehlo.dot_general %8, %14, contracting_dims = [1] x [0] : (tensor<64x64xf32>, tensor<64xf32>) -> tensor<64xf32>
// CHECK-NEXT:     %16 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<64xf32>
// CHECK-NEXT:     %17 = stablehlo.multiply %16, %15 : tensor<64xf32>
// CHECK-NEXT:     return %17, %9, %14 : tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>
// CHECK-NEXT:   }
