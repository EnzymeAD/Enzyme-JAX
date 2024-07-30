// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{max_constant_expansion=1})" %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<5x3x224x224xf32>, %arg1: tensor<2x3x10x10xf32>) -> tensor<215x215x2x5xf32> {
    %0 = stablehlo.transpose %arg0, dims = [3, 2, 1, 0] : (tensor<5x3x224x224xf32>) -> tensor<224x224x3x5xf32>
    %1 = stablehlo.transpose %arg1, dims = [3, 2, 1, 0] : (tensor<2x3x10x10xf32>) -> tensor<10x10x3x2xf32>
    %2 = stablehlo.convolution(%0, %1) dim_numbers = [0, 1, f, b]x[0, 1, i, o]->[b, f, 1, 0], window = {stride = [1, 1], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<224x224x3x5xf32>, tensor<10x10x3x2xf32>) -> tensor<5x2x215x215xf32>
    %3 = stablehlo.transpose %2, dims = [3, 2, 1, 0] : (tensor<5x2x215x215xf32>) -> tensor<215x215x2x5xf32>
    return %3 :tensor<215x215x2x5xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<5x3x224x224xf32>, %arg1: tensor<2x3x10x10xf32>) -> tensor<215x215x2x5xf32> {
// CHECK-NEXT:    %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 1, 0]x[o, i, 1, 0]->[0, 1, f, b], window = {stride = [1, 1], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<5x3x224x224xf32>, tensor<2x3x10x10xf32>) -> tensor<215x215x2x5xf32>
// CHECK-NEXT:    return %0 : tensor<215x215x2x5xf32>
// CHECK-NEXT:  }
