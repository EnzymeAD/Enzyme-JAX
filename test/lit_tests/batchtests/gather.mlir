// RUN: enzymexlamlir-opt %s --enzyme-batch | FileCheck %s

module @reactant_fn1_bat... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func private @unbatched_fn1(%arg0: tensor<4x6x5xf32>, %arg1: tensor<6xi64>, %arg2: tensor<7xi64>) -> tensor<6x6x7xf32> {
    %c = stablehlo.constant dense<1> : tensor<42x2xi64>
    %0 = stablehlo.concatenate %arg1, %arg1, %arg1, %arg1, %arg1, %arg1, %arg1, dim = 0 : (tensor<6xi64>, tensor<6xi64>, tensor<6xi64>, tensor<6xi64>, tensor<6xi64>, tensor<6xi64>, tensor<6xi64>) -> tensor<42xi64>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<7xi64>) -> tensor<7x6xi64>
    %2 = stablehlo.reshape %0 : (tensor<42xi64>) -> tensor<42x1xi64>
    %3 = stablehlo.reshape %1 : (tensor<7x6xi64>) -> tensor<42x1xi64>
    %4 = stablehlo.concatenate %2, %3, dim = 1 : (tensor<42x1xi64>, tensor<42x1xi64>) -> tensor<42x2xi64>
    %5 = stablehlo.subtract %4, %c : tensor<42x2xi64>
    %6 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 2], start_index_map = [0, 2], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 6, 1>}> : (tensor<4x6x5xf32>, tensor<42x2xi64>) -> tensor<42x6xf32>
    %7 = stablehlo.transpose %6, dims = [1, 0] : (tensor<42x6xf32>) -> tensor<6x42xf32>
    %8 = stablehlo.reshape %7 : (tensor<6x42xf32>) -> tensor<6x7x6xf32>
    %9 = stablehlo.transpose %8, dims = [2, 0, 1] : (tensor<6x7x6xf32>) -> tensor<6x6x7xf32>
    return %9 : tensor<6x6x7xf32>
  }
  func.func @main(%arg0: tensor<5x6x4x3x2xf32>, %arg1: tensor<6x3x2xi64>, %arg2: tensor<7x3x2xi64>) -> tensor<7x6x6x3x2xf32> {
    %0 = stablehlo.transpose %arg0, dims = [4, 3, 2, 1, 0] : (tensor<5x6x4x3x2xf32>) -> tensor<2x3x4x6x5xf32>
    %1 = stablehlo.transpose %arg1, dims = [2, 1, 0] : (tensor<6x3x2xi64>) -> tensor<2x3x6xi64>
    %2 = stablehlo.transpose %arg2, dims = [2, 1, 0] : (tensor<7x3x2xi64>) -> tensor<2x3x7xi64>
    %3 = enzyme.batch @unbatched_fn1(%0, %1, %2) {batch_shape = array<i64: 2, 3>} : (tensor<2x3x4x6x5xf32>, tensor<2x3x6xi64>, tensor<2x3x7xi64>) -> tensor<2x3x6x6x7xf32>
    %4 = stablehlo.transpose %3, dims = [4, 3, 2, 1, 0] : (tensor<2x3x6x6x7xf32>) -> tensor<7x6x6x3x2xf32>
    return %4 : tensor<7x6x6x3x2xf32>
  }
}

// CHECK: func.func private @batched_unbatched_fn1(%arg0: tensor<2x3x4x6x5xf32>, %arg1: tensor<2x3x6xi64>, %arg2: tensor<2x3x7xi64>) -> tensor<2x3x6x6x7xf32> {
// CHECK-NEXT:     %c = stablehlo.constant dense<1> : tensor<2x3x42x2xi64>
// CHECK-NEXT:     %0 = stablehlo.concatenate %arg1, %arg1, %arg1, %arg1, %arg1, %arg1, %arg1, dim = 2 : (tensor<2x3x6xi64>, tensor<2x3x6xi64>, tensor<2x3x6xi64>, tensor<2x3x6xi64>, tensor<2x3x6xi64>, tensor<2x3x6xi64>, tensor<2x3x6xi64>) -> tensor<2x3x42xi64>
// CHECK-NEXT:     %1 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2] : (tensor<2x3x7xi64>) -> tensor<2x3x7x6xi64>
// CHECK-NEXT:     %2 = stablehlo.reshape %0 : (tensor<2x3x42xi64>) -> tensor<2x3x42x1xi64>
// CHECK-NEXT:     %3 = stablehlo.reshape %1 : (tensor<2x3x7x6xi64>) -> tensor<2x3x42x1xi64>
// CHECK-NEXT:     %4 = stablehlo.concatenate %2, %3, dim = 3 : (tensor<2x3x42x1xi64>, tensor<2x3x42x1xi64>) -> tensor<2x3x42x2xi64>
// CHECK-NEXT:     %5 = stablehlo.subtract %4, %c : tensor<2x3x42x2xi64>
// CHECK-NEXT:     %6 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [3], collapsed_slice_dims = [2, 4], operand_batching_dims = [0, 1], start_indices_batching_dims = [0, 1], start_index_map = [2, 4], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1, 6, 1>}> : (tensor<2x3x4x6x5xf32>, tensor<2x3x42x2xi64>) -> tensor<2x3x42x6xf32>
// CHECK-NEXT:     %7 = stablehlo.transpose %6, dims = [0, 1, 3, 2] : (tensor<2x3x42x6xf32>) -> tensor<2x3x6x42xf32>
// CHECK-NEXT:     %8 = stablehlo.reshape %7 : (tensor<2x3x6x42xf32>) -> tensor<2x3x6x7x6xf32>
// CHECK-NEXT:     %9 = stablehlo.transpose %8, dims = [0, 1, 4, 2, 3] : (tensor<2x3x6x7x6xf32>) -> tensor<2x3x6x6x7xf32>
// CHECK-NEXT:     return %9 : tensor<2x3x6x6x7xf32>
// CHECK-NEXT: }
