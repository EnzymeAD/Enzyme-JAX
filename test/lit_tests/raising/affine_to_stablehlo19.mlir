// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --canonicalize --enzyme-hlo-opt=max_constant_expansion=0 | FileCheck %s

module {
  func.func @broadcast_test(%arg0: memref<1x20xf32>, %arg1: memref<10x10xf32>) {
    affine.parallel (%arg2, %arg3) = (0, 0) to (10, 10) {
       // Force emitLoadAsGather using complex index on Dim 1 (multiple IVs).
       // Dim 0 is Constant 0.
       %1 = affine.load %arg0[0, %arg2 + %arg3] : memref<1x20xf32>
       %2 = affine.load %arg0[%arg2 + %arg3, 0] : memref<1x20xf32>
       %3 = arith.addf %1, %2 : f32
       affine.store %3, %arg1[%arg2, %arg3] : memref<10x10xf32>
    }
    return
  }
}

// CHECK:  func.func private @broadcast_test_raised(%arg0: tensor<1x20xf32>, %arg1: tensor<10x10xf32>) -> (tensor<1x20xf32>, tensor<10x10xf32>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[v0:.+]] = stablehlo.iota dim = 0 : tensor<10x10x1xi64>
// CHECK-NEXT:    %[[v1:.+]] = stablehlo.iota dim = 1 : tensor<10x10x1xi64>
// CHECK-NEXT:    %[[v2:.+]] = stablehlo.add %[[v0]], %[[v1]] : tensor<10x10x1xi64>
// CHECK-NEXT:    %[[v3:.+]] = stablehlo.pad %[[v2]], %c, low = [0, 0, 1], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<10x10x1xi64>, tensor<i64>) -> tensor<10x10x2xi64>
// CHECK-NEXT:    %[[v4:.+]] = stablehlo.reshape %[[v3]] : (tensor<10x10x2xi64>) -> tensor<100x2xi64>
// CHECK-NEXT:    %[[v5:.+]] = "stablehlo.gather"(%arg0, %[[v4]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<1x20xf32>, tensor<100x2xi64>) -> tensor<100xf32>
// CHECK-NEXT:    %[[v6:.+]] = stablehlo.reshape %[[v5]] : (tensor<100xf32>) -> tensor<10x10xf32>
// CHECK-NEXT:    %[[v7:.+]] = stablehlo.pad %[[v2]], %c, low = [0, 0, 0], high = [0, 0, 1], interior = [0, 0, 0] : (tensor<10x10x1xi64>, tensor<i64>) -> tensor<10x10x2xi64>
// CHECK-NEXT:    %[[v8:.+]] = stablehlo.reshape %[[v7]] : (tensor<10x10x2xi64>) -> tensor<100x2xi64>
// CHECK-NEXT:    %[[v9:.+]] = "stablehlo.gather"(%arg0, %[[v8]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<1x20xf32>, tensor<100x2xi64>) -> tensor<100xf32>
// CHECK-NEXT:    %[[v10:.+]] = stablehlo.reshape %[[v9]] : (tensor<100xf32>) -> tensor<10x10xf32>
// CHECK-NEXT:    %[[v11:.+]] = arith.addf %[[v6]], %[[v10]] : tensor<10x10xf32>
// CHECK-NEXT:    return %arg0, %[[v11]] : tensor<1x20xf32>, tensor<10x10xf32>
// CHECK-NEXT:  }
