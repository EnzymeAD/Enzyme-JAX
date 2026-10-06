// RUN: enzymexlamlir-opt %s "--pass-pipeline=builtin.module(raise-affine-to-stablehlo{enable_lockstep_for=true})" | FileCheck %s

// A loop of constant count whose accesses are offset by a product of
// symbols: the lockstep check now builds its dependence problems as the
// parallel check does, with that product abstracted, and the loop runs in
// lock step (one gather and one scatter) rather than as a while.
module {
  func.func private @copy(%a: memref<64xf64, 1>, %b: memref<64xf64, 1>, %p: memref<2xi64, 1>) {
    %s64 = affine.load %p[0] : memref<2xi64, 1>
    %t64 = affine.load %p[1] : memref<2xi64, 1>
    %s = arith.index_cast %s64 : i64 to index
    %t = arith.index_cast %t64 : i64 to index
    affine.parallel (%i) = (0) to (4) {
      affine.for %k = 0 to 3 {
        %v = affine.load %a[%k + %i * 3 + symbol(%s) * symbol(%t)] : memref<64xf64, 1>
        affine.store %v, %b[%k + %i * 3 + symbol(%s) * symbol(%t)] : memref<64xf64, 1>
      }
    }
    return
  }
}

// CHECK:  func.func private @copy_raised(%arg0: tensor<64xf64>, %arg1: tensor<64xf64>, %arg2: tensor<2xi64>) -> (tensor<64xf64>, tensor<64xf64>, tensor<2xi64>) {
// CHECK-NEXT:  %0 = stablehlo.slice %arg2 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-NEXT:  %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
// CHECK-NEXT:  %2 = stablehlo.slice %arg2 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-NEXT:  %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
// CHECK-NEXT:  %4 = stablehlo.iota dim = 0 : tensor<4xi64>
// CHECK-NEXT:  %c = stablehlo.constant dense<0> : tensor<4xi64>
// CHECK-NEXT:  %5 = stablehlo.add %4, %c : tensor<4xi64>
// CHECK-NEXT:  %c_0 = stablehlo.constant dense<1> : tensor<4xi64>
// CHECK-NEXT:  %6 = stablehlo.multiply %5, %c_0 : tensor<4xi64>
// CHECK-NEXT:  %7 = stablehlo.iota dim = 0 : tensor<3xi64>
// CHECK-NEXT:  %c_1 = stablehlo.constant dense<0> : tensor<3xi64>
// CHECK-NEXT:  %8 = stablehlo.add %7, %c_1 : tensor<3xi64>
// CHECK-NEXT:  %c_2 = stablehlo.constant dense<1> : tensor<3xi64>
// CHECK-NEXT:  %9 = stablehlo.multiply %8, %c_2 : tensor<3xi64>
// CHECK-NEXT:  %c_3 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:  %10 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:  %11 = stablehlo.multiply %6, %10 : tensor<4xi64>
// CHECK-NEXT:  %12 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<3xi64>) -> tensor<3x4xi64>
// CHECK-NEXT:  %13 = stablehlo.broadcast_in_dim %11, dims = [1] : (tensor<4xi64>) -> tensor<3x4xi64>
// CHECK-NEXT:  %14 = stablehlo.add %12, %13 : tensor<3x4xi64>
// CHECK-NEXT:  %15 = stablehlo.multiply %1, %3 : tensor<i64>
// CHECK-NEXT:  %16 = stablehlo.broadcast_in_dim %15, dims = [] : (tensor<i64>) -> tensor<3x4xi64>
// CHECK-NEXT:  %17 = stablehlo.add %14, %16 : tensor<3x4xi64>
// CHECK-NEXT:  %18 = stablehlo.reshape %17 : (tensor<3x4xi64>) -> tensor<3x4x1xi64>
// CHECK-NEXT:  %19 = "stablehlo.gather"(%arg0, %18) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<64xf64>, tensor<3x4x1xi64>) -> tensor<3x4xf64>
// CHECK-NEXT:  %c_4 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:  %20 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:  %21 = stablehlo.multiply %6, %20 : tensor<4xi64>
// CHECK-NEXT:  %22 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<3xi64>) -> tensor<3x4xi64>
// CHECK-NEXT:  %23 = stablehlo.broadcast_in_dim %21, dims = [1] : (tensor<4xi64>) -> tensor<3x4xi64>
// CHECK-NEXT:  %24 = stablehlo.add %22, %23 : tensor<3x4xi64>
// CHECK-NEXT:  %25 = stablehlo.multiply %1, %3 : tensor<i64>
// CHECK-NEXT:  %26 = stablehlo.broadcast_in_dim %25, dims = [] : (tensor<i64>) -> tensor<3x4xi64>
// CHECK-NEXT:  %27 = stablehlo.add %24, %26 : tensor<3x4xi64>
// CHECK-NEXT:  %28 = stablehlo.reshape %27 : (tensor<3x4xi64>) -> tensor<3x4x1xi64>
// CHECK-NEXT:  %29 = "stablehlo.scatter"(%arg1, %28, %19) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = true}> ({
// CHECK-NEXT:  ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:    stablehlo.return %arg4 : tensor<f64>
// CHECK-NEXT:  }) : (tensor<64xf64>, tensor<3x4x1xi64>, tensor<3x4xf64>) -> tensor<64xf64>
// CHECK-NEXT:  return %arg0, %29, %arg2 : tensor<64xf64>, tensor<64xf64>, tensor<2xi64>
// CHECK-NEXT: }
