// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module @reactant_kernel_... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<512x1024x256xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 0 : i32}, %arg1: tensor<512x512xf32> {enzymexla.memory_effects = []}) -> tensor<512x1024x256xf32> attributes {enzymexla.memory_effects = []} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [1] : (tensor<512x1024x256xf32>, tensor<512x512xf32>) -> tensor<1024x256x512xf32>

    %1 = stablehlo.broadcast_in_dim %0, dims = [2, 1, 3] : (tensor<1024x256x512xf32>) -> tensor<1x256x1024x512xf32>
    %2 = stablehlo.reshape %1 : (tensor<1x256x1024x512xf32>) -> tensor<256x1024x512x1xf32>

    // %2 = stablehlo.broadcast_in_dim %0, dims = [1, 0, 2] : (tensor<1024x256x512xf32>) -> tensor<256x1024x512x1xf32>

    %3 = stablehlo.transpose %2, dims = [2, 1, 0, 3] : (tensor<256x1024x512x1xf32>) -> tensor<512x1024x256x1xf32>
    %4 = stablehlo.reshape %3 : (tensor<512x1024x256x1xf32>) -> tensor<512x1024x256xf32>
    return %4 : tensor<512x1024x256xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<512x1024x256xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 0 : i32}, %arg1: tensor<512x512xf32> {enzymexla.memory_effects = []}) -> tensor<512x1024x256xf32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [0] : (tensor<512x512xf32>, tensor<512x1024x256xf32>) -> tensor<512x1024x256xf32>
// CHECK-NEXT:    return %0 : tensor<512x1024x256xf32>
// CHECK-NEXT:  }
