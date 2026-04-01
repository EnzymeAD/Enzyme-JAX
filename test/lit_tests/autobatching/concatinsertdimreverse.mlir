// RUN: enzymexlamlir-opt --auto-batching --enzyme-hlo-opt %s | FileCheck %s

module @reactant_loop1 attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<4x2x3xf32> {enzymexla.memory_effects = []}) -> tensor<4x2x3xf32> attributes {enzymexla.memory_effects = []} {
    %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<4x2x3xf32>) -> tensor<3x2x4xf32>
    %1 = stablehlo.slice %0 [0:3, 0:1, 0:4] : (tensor<3x2x4xf32>) -> tensor<3x1x4xf32>
    %2 = stablehlo.reshape %1 : (tensor<3x1x4xf32>) -> tensor<3x4xf32>
    %3 = stablehlo.reverse %2, dims = [0, 1] : tensor<3x4xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2, 0] : (tensor<3x4xf32>) -> tensor<4x1x3xf32>
    %5 = stablehlo.slice %0 [0:3, 1:2, 0:4] : (tensor<3x2x4xf32>) -> tensor<3x1x4xf32>
    %6 = stablehlo.reshape %5 : (tensor<3x1x4xf32>) -> tensor<3x4xf32>
    %7 = stablehlo.reverse %6, dims = [0, 1] : tensor<3x4xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [2, 0] : (tensor<3x4xf32>) -> tensor<4x1x3xf32>
    %9 = stablehlo.concatenate %4, %8, dim = 1 : (tensor<4x1x3xf32>, tensor<4x1x3xf32>) -> tensor<4x2x3xf32>
    return %9 : tensor<4x2x3xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<4x2x3xf32> {enzymexla.memory_effects = []}) -> tensor<4x2x3xf32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [1, 2, 0] : (tensor<4x2x3xf32>) -> tensor<2x3x4xf32>
// CHECK-NEXT:     %1 = stablehlo.reverse %0, dims = [1, 2] : tensor<2x3x4xf32>
// CHECK-NEXT:     %2 = stablehlo.transpose %1, dims = [2, 0, 1] : (tensor<2x3x4xf32>) -> tensor<4x2x3xf32>
// CHECK-NEXT:     return %2 : tensor<4x2x3xf32>
// CHECK-NEXT: }
