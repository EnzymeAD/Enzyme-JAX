// RUN: enzymexlamlir-opt --enzyme-batch --inline %s | FileCheck %s

module {
  func.func private @unbatched_fn(%arg0: tensor<4x4x2x3xf32> {enzymexla.memory_effects = []}) -> tensor<4x4x2x3xf32> attributes {enzymexla.memory_effects = []} {
    %0 = stablehlo.reverse %arg0, dims = [0, 1] : tensor<4x4x2x3xf32>
    return %0 : tensor<4x4x2x3xf32>
  }
  func.func @main(%arg0: tensor<6x3x2x5x4x4xf32>) -> tensor<6x3x2x5x4x4xf32> {
    %0 = stablehlo.transpose %arg0, dims = [3, 0, 5, 4, 2, 1] : (tensor<6x3x2x5x4x4xf32>) -> tensor<5x6x4x4x2x3xf32>
    %1 = enzyme.batch @unbatched_fn(%0) {batch_shape = array<i64: 5, 6>} : (tensor<5x6x4x4x2x3xf32>) -> tensor<5x6x4x4x2x3xf32>
    %2 = stablehlo.transpose %1, dims = [1, 5, 4, 0, 3, 2] : (tensor<5x6x4x4x2x3xf32>) -> tensor<6x3x2x5x4x4xf32>
    return %2 : tensor<6x3x2x5x4x4xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<6x3x2x5x4x4xf32>) -> tensor<6x3x2x5x4x4xf32> {
// CHECK-NEXT:    %0 = stablehlo.transpose %arg0, dims = [3, 0, 5, 4, 2, 1] : (tensor<6x3x2x5x4x4xf32>) -> tensor<5x6x4x4x2x3xf32>
// CHECK-NEXT:    %1 = stablehlo.reverse %0, dims = [2, 3] : tensor<5x6x4x4x2x3xf32>
// CHECK-NEXT:    %2 = stablehlo.transpose %1, dims = [1, 5, 4, 0, 3, 2] : (tensor<5x6x4x4x2x3xf32>) -> tensor<6x3x2x5x4x4xf32>
// CHECK-NEXT:    return %2 : tensor<6x3x2x5x4x4xf32>
// CHECK-NEXT:  }
