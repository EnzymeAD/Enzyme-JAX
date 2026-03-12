// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --canonicalize --enzyme-hlo-opt=max_constant_expansion=0 | FileCheck %s

module {
  func.func @main(%arg0: memref<100xf32>, %arg1: memref<100xf32>) {
    affine.parallel (%i, %j) = (0, 0) to (10, 10) step (1, 1) {
      %0 = affine.load %arg1[%i * 10 + %j] : memref<100xf32>
      affine.store %0, %arg0[%i * 10 + %j] : memref<100xf32>
    }
    return
  }
}

// CHECK:  func.func private @main_raised(%arg0: tensor<100xf32>, %arg1: tensor<100xf32>) -> (tensor<100xf32>, tensor<100xf32>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<10> : tensor<10xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<99> : tensor<100x1xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<100x1xi64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<10xi64>
// CHECK-NEXT:    %1 = stablehlo.multiply %0, %c : tensor<10xi64>
// CHECK-NEXT:    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<10xi64>) -> tensor<10x10xi64>
// CHECK-NEXT:    %3 = stablehlo.iota dim = 1 : tensor<10x10xi64>
// CHECK-NEXT:    %4 = stablehlo.add %2, %3 : tensor<10x10xi64>
// CHECK-NEXT:    %5 = stablehlo.reshape %4 : (tensor<10x10xi64>) -> tensor<100x1xi64>
// CHECK-NEXT:    %6 = stablehlo.maximum %5, %c_1 : tensor<100x1xi64>
// CHECK-NEXT:    %7 = stablehlo.minimum %6, %c_0 : tensor<100x1xi64>
// CHECK-NEXT:    %8 = "stablehlo.gather"(%arg1, %7) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<100xf32>, tensor<100x1xi64>) -> tensor<100xf32>
// CHECK-NEXT:    %9 = "stablehlo.scatter"(%arg0, %7, %8) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// CHECK-NEXT:      stablehlo.return %arg3 : tensor<f32>
// CHECK-NEXT:    }) : (tensor<100xf32>, tensor<100x1xi64>, tensor<100xf32>) -> tensor<100xf32>
// CHECK-NEXT:    return %9, %arg1 : tensor<100xf32>, tensor<100xf32>
// CHECK-NEXT:  }
