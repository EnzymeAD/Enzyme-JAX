// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --canonicalize --arith-raise --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%output: memref<3x3xi64>, %values: memref<3x3xi64>) {
    affine.parallel (%i, %j) = (0, 0) to (3, 3) {
        %val = memref.load %values[%i, %j] : memref<3x3xi64>
        affine.store %val, %output[%i, %j] : memref<3x3xi64>
    }
    return
  }
}
// CHECK:  func.func private @main_raised(%arg0: tensor<3x3xi64>, %arg1: tensor<3x3xi64>) -> (tensor<3x3xi64>, tensor<3x3xi64>) {
// CHECK-NEXT{LITERAL}:    %c = stablehlo.constant dense<[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]> : tensor<9x2xi64>
// CHECK-NEXT:    %0 = "stablehlo.gather"(%arg1, %c) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<3x3xi64>, tensor<9x2xi64>) -> tensor<9xi64>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<9xi64>) -> tensor<3x3xi64>
// CHECK-NEXT:    return %1, %arg1 : tensor<3x3xi64>, tensor<3x3xi64>
// CHECK-NEXT:  }

module {
  func.func @single_dim(%output: memref<3xi64>, %values: memref<3xi64>) {
    affine.parallel (%i) = (0) to (3) {
        %val = memref.load %values[%i] : memref<3xi64>
        affine.store %val, %output[%i] : memref<3xi64>
    }
    return
  }
}
// CHECK:  func.func private @single_dim_raised(%arg0: tensor<3xi64>, %arg1: tensor<3xi64>) -> (tensor<3xi64>, tensor<3xi64>) {
// CHECK-NEXT{LITERAL}:    %c = stablehlo.constant dense<[[0], [1], [2]]> : tensor<3x1xi64>
// CHECK-NEXT:    %0 = "stablehlo.gather"(%arg1, %c) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<3xi64>, tensor<3x1xi64>) -> tensor<3xi64>
// CHECK-NEXT:    return %0, %arg1 : tensor<3xi64>, tensor<3xi64>
// CHECK-NEXT:  }

module {
  func.func @with_constant(%output: memref<3x3xi64>, %values: memref<3x3xi64>) {
    %c = arith.constant 0 : index
    affine.parallel (%i) = (0) to (3) {
        %val = memref.load %values[%i, %c] : memref<3x3xi64>
        affine.store %val, %output[%i, %c] : memref<3x3xi64>
    }
    return
  }
}
// CHECK:  func.func private @with_constant_raised(%arg0: tensor<3x3xi64>, %arg1: tensor<3x3xi64>) -> (tensor<3x3xi64>, tensor<3x3xi64>) {
// CHECK-NEXT{LITERAL}:    %c = stablehlo.constant dense<[[0, 0], [1, 0], [2, 0]]> : tensor<3x2xi64>
// CHECK-NEXT:    %0 = "stablehlo.gather"(%arg1, %c) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<3x3xi64>, tensor<3x2xi64>) -> tensor<3xi64>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<3xi64>) -> tensor<3x1xi64>
// CHECK-NEXT:    %2 = stablehlo.slice %arg0 [0:3, 1:3] : (tensor<3x3xi64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %3 = stablehlo.concatenate %1, %2, dim = 1 : (tensor<3x1xi64>, tensor<3x2xi64>) -> tensor<3x3xi64>
// CHECK-NEXT:    return %3, %arg1 : tensor<3x3xi64>, tensor<3x3xi64>
// CHECK-NEXT:  }

module {
  func.func @load_indirect(%output: memref<3x3xi64>, %values: memref<3x3xi64>, %indices: memref<3xi64>) {
    affine.parallel (%i, %j) = (0, 0) to (3, 3) {
      %idx_i64 = affine.load %indices[%j] : memref<3xi64>
      %idx = arith.index_cast %idx_i64 : i64 to index
      %val = memref.load %values[%i, %idx] : memref<3x3xi64>
      affine.store %val, %output[%j, %i] : memref<3x3xi64>
    }
    return
  }
}
// CHECK:  func.func private @load_indirect_raised(%arg0: tensor<3x3xi64>, %arg1: tensor<3x3xi64>, %arg2: tensor<3xi64>) -> (tensor<3x3xi64>, tensor<3x3xi64>, tensor<3xi64>) {
// CHECK-NEXT{LITERAL}:    %c = stablehlo.constant dense<[[0], [0], [0], [1], [1], [1], [2], [2], [2]]> : tensor<9x1xi64>
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<3xi64>) -> tensor<3x3xi64>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<3x3xi64>) -> tensor<9x1xi64>
// CHECK-NEXT:    %2 = stablehlo.concatenate %c, %1, dim = 1 : (tensor<9x1xi64>, tensor<9x1xi64>) -> tensor<9x2xi64>
// CHECK-NEXT:    %3 = "stablehlo.gather"(%arg1, %2) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<3x3xi64>, tensor<9x2xi64>) -> tensor<9xi64>
// CHECK-NEXT:    %4 = stablehlo.reshape %3 : (tensor<9xi64>) -> tensor<3x3xi64>
// CHECK-NEXT:    %5 = stablehlo.transpose %4, dims = [1, 0] : (tensor<3x3xi64>) -> tensor<3x3xi64>
// CHECK-NEXT:    return %5, %arg1, %arg2 : tensor<3x3xi64>, tensor<3x3xi64>, tensor<3xi64>
// CHECK-NEXT:  }
