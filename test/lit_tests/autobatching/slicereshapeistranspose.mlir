// RUN: enzymexlamlir-opt --auto-batching --inline --enzyme-hlo-opt %s | FileCheck %s
// TODO

module {
  func.func @main(%arg0: tensor<3x5x10xf32>, %arg1: tensor<3x5x10xf32>) -> tensor<5x3x10xf32> attributes {enzymexla.memory_effects = []} {
    %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<3x5x10xf32>) -> tensor<10x5x3xf32>
    %1 = stablehlo.slice %0 [0:10, 0:1, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %2 = stablehlo.slice %0 [0:10, 1:2, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %3 = stablehlo.slice %0 [0:10, 2:3, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %4 = stablehlo.slice %0 [0:10, 3:4, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %5 = stablehlo.slice %0 [0:10, 4:5, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %6 = stablehlo.reshape %5 : (tensor<10x1x3xf32>) -> tensor<1x10x3xf32>
    %7 = stablehlo.reshape %4 : (tensor<10x1x3xf32>) -> tensor<1x10x3xf32>
    %8 = stablehlo.reshape %3 : (tensor<10x1x3xf32>) -> tensor<1x10x3xf32>
    %9 = stablehlo.reshape %2 : (tensor<10x1x3xf32>) -> tensor<1x10x3xf32>
    %10 = stablehlo.reshape %1 : (tensor<10x1x3xf32>) -> tensor<1x10x3xf32>
    %11 = stablehlo.transpose %6, dims = [0, 2, 1] : (tensor<1x10x3xf32>) -> tensor<1x3x10xf32>
    %12 = stablehlo.transpose %7, dims = [0, 2, 1] : (tensor<1x10x3xf32>) -> tensor<1x3x10xf32>
    %13 = stablehlo.transpose %8, dims = [0, 2, 1] : (tensor<1x10x3xf32>) -> tensor<1x3x10xf32>
    %14 = stablehlo.transpose %9, dims = [0, 2, 1] : (tensor<1x10x3xf32>) -> tensor<1x3x10xf32>
    %15 = stablehlo.transpose %10, dims = [0, 2, 1] : (tensor<1x10x3xf32>) -> tensor<1x3x10xf32>
    %16 = stablehlo.concatenate %11, %12, %13, %14, %15, dim = 0 : (tensor<1x3x10xf32>, tensor<1x3x10xf32>, tensor<1x3x10xf32>, tensor<1x3x10xf32>, tensor<1x3x10xf32>) -> tensor<5x3x10xf32>
    return %16 : tensor<5x3x10xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<3x5x10xf32>, %arg1: tensor<3x5x10xf32>) -> tensor<5x3x10xf32> attributes {enzymexla.memory_effects = []} {
