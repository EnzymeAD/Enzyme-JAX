// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --canonicalize --enzyme-hlo-opt=max_constant_expansion=0 | FileCheck %s

module {
  func.func @main(%arg0: memref<4x10xf32>, %arg1: memref<16x10xf32>) {
    affine.parallel (%i) = (0) to (10) {
      affine.for %j = 0 to 4 {
        %0 = affine.load %arg0[%j, %i] : memref<4x10xf32>
        %1 = arith.mulf %0, %0 : f32
        affine.store %1, %arg1[4 * %j, %i] : memref<16x10xf32>
      }
    }
    return
  }
}

// CHECK:  func.func private @main_raised(%arg0: tensor<4x10xf32>, %arg1: tensor<16x10xf32>) -> (tensor<4x10xf32>, tensor<16x10xf32>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0:3 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg0, %iterArg_3 = %arg1) : tensor<i64>, tensor<4x10xf32>, tensor<16x10xf32>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %1 = stablehlo.broadcast_in_dim %iterArg, dims = [] : (tensor<i64>) -> tensor<10x1xi64>
// CHECK-NEXT:      %2 = stablehlo.iota dim = 0 : tensor<10x1xi64>
// CHECK-NEXT:      %3 = stablehlo.concatenate %1, %2, dim = 1 : (tensor<10x1xi64>, tensor<10x1xi64>) -> tensor<10x2xi64>
// CHECK-NEXT:      %4 = "stablehlo.gather"(%iterArg_2, %3) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<4x10xf32>, tensor<10x2xi64>) -> tensor<10xf32>
// CHECK-NEXT:      %5 = arith.mulf %4, %4 : tensor<10xf32>
// CHECK-NEXT:      %6 = stablehlo.multiply %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:      %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i64>) -> tensor<10x1xi64>
// CHECK-NEXT:      %8 = stablehlo.concatenate %7, %2, dim = 1 : (tensor<10x1xi64>, tensor<10x1xi64>) -> tensor<10x2xi64>
// CHECK-NEXT:      %9 = "stablehlo.scatter"(%iterArg_3, %8, %5) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// CHECK-NEXT:        stablehlo.return %arg3 : tensor<f32>
// CHECK-NEXT:      }) : (tensor<16x10xf32>, tensor<10x2xi64>, tensor<10xf32>) -> tensor<16x10xf32>
// CHECK-NEXT:      %10 = stablehlo.add %iterArg, %c : tensor<i64>
// CHECK-NEXT:      stablehlo.return %10, %iterArg_2, %9 : tensor<i64>, tensor<4x10xf32>, tensor<16x10xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %0#1, %0#2 : tensor<4x10xf32>, tensor<16x10xf32>
// CHECK-NEXT:  }
