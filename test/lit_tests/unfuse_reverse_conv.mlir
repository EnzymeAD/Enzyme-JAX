// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%arg0: tensor<4x5x1024xf32>, %arg1: tensor<4x2x2xf32>) -> (tensor<5x2x1023xf32>) {
    %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0]x[i, o, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], lhs_dilate = [1], rhs_dilate = [1], reverse = [true]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<4x5x1024xf32>, tensor<4x2x2xf32>) -> tensor<5x2x1023xf32>
    return %0 : tensor<5x2x1023xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<4x5x1024xf32>, %arg1: tensor<4x2x2xf32>) -> tensor<5x2x1023xf32> {
// CHECK-NEXT:     %0 = stablehlo.reverse %arg1, dims = [2] : tensor<4x2x2xf32>
// CHECK-NEXT{LITERAL}:     %1 = stablehlo.convolution(%arg0, %0) dim_numbers = [f, b, 0]x[i, o, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], lhs_dilate = [1], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<4x5x1024xf32>, tensor<4x2x2xf32>) -> tensor<5x2x1023xf32>
// CHECK-NEXT:     return %1 : tensor<5x2x1023xf32>
// CHECK-NEXT: }
