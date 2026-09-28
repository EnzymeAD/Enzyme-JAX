// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo="prefer_while_raising=true" | FileCheck %s
// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo="prefer_while_raising=true specialize_loop_bounds=false" | FileCheck %s --check-prefix=PLAIN

// The kernel's element loop runs to `ne`, a scalar it reads but never
// writes. With specialize_loop_bounds the scalar moves to the end of the
// raised function's arguments, is not returned, and the wrapper counts it
// among the scalars the runtime specializes the executable over: every pass
// after that sees a constant trip count.

func.func @scale(%out: memref<?xf64, 1>, %nem: memref<i32, 1>, %in: memref<?xf64, 1>, %unused: index) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %ne = affine.load %nem[] : memref<i32, 1>
  %nei = arith.index_cast %ne : i32 to index
  %c0_i32 = arith.constant 0 : i32
  %ok = arith.cmpi sgt, %ne, %c0_i32 : i32
  scf.if %ok {
  %0 = "enzymexla.gpu_wrapper"(%nei, %c1, %c1, %c2, %c1, %c1) ({
    affine.parallel (%e) = (0) to (symbol(%nei)) {
      affine.parallel (%t) = (0) to (2) {
        %v = affine.load %in[%e * 2 + %t] : memref<?xf64, 1>
        %w = arith.mulf %v, %v : f64
        affine.store %w, %out[%e * 2 + %t] : memref<?xf64, 1>
      }
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  }
  return
}

// CHECK:  func.func @scale(%arg0: memref<?xf64, 1>, %arg1: memref<i32, 1>, %arg2: memref<?xf64, 1>, %arg3: index) {
// CHECK-NEXT:   %alloca = memref.alloca() : memref<i32>
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %c2 = arith.constant 2 : index
// CHECK-NEXT:   %0 = affine.load %arg1[] : memref<i32, 1>
// CHECK-NEXT:   %1 = arith.index_cast %0 : i32 to index
// CHECK-NEXT:   %2 = arith.cmpi sgt, %0, %c0_i32 : i32
// CHECK-NEXT:   scf.if %2 {
// CHECK-NEXT:     affine.store %0, %alloca[] : memref<i32>
// CHECK-NEXT:     %c4 = arith.constant 4 : index
// CHECK-NEXT:     enzymexla.xla_wrapper @rxla$raised_0 (%arg2, %arg0, %0) {num_specialized = 1 : i64} : (memref<?xf64, 1>, memref<?xf64, 1>, i32) -> ()
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
// CHECK-NEXT: func.func private @rxla$raised_0(%arg0: tensor<?xf64>, %arg1: tensor<?xf64>, %arg2: tensor<i32>) -> (tensor<?xf64>, tensor<?xf64>) {
// CHECK-NEXT:   %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<i64>
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %1:3 = stablehlo.while(%iterArg = %c, %iterArg_1 = %arg0, %iterArg_2 = %arg1) : tensor<i64>, tensor<?xf64>, tensor<?xf64> attributes {enzymexla.parallel}
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %2 = stablehlo.compare LT, %iterArg, %0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %2 = stablehlo.iota dim = 0 : tensor<2xi64>
// CHECK-NEXT:     %c_3 = stablehlo.constant dense<0> : tensor<2xi64>
// CHECK-NEXT:     %3 = stablehlo.add %2, %c_3 : tensor<2xi64>
// CHECK-NEXT:     %c_4 = stablehlo.constant dense<1> : tensor<2xi64>
// CHECK-NEXT:     %4 = stablehlo.multiply %3, %c_4 : tensor<2xi64>
// CHECK-NEXT:     %c_5 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:     %5 = stablehlo.multiply %iterArg, %c_5 : tensor<i64>
// CHECK-NEXT:     %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:     %7 = stablehlo.add %6, %4 : tensor<2xi64>
// CHECK-NEXT:     %8 = stablehlo.reshape %7 : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK-NEXT:     %9 = "stablehlo.gather"(%iterArg_1, %8) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<?xf64>, tensor<2x1xi64>) -> tensor<2xf64>
// CHECK-NEXT:     %10 = arith.mulf %9, %9 : tensor<2xf64>
// CHECK-NEXT:     %c_6 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:     %11 = stablehlo.multiply %iterArg, %c_6 : tensor<i64>
// CHECK-NEXT:     %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:     %13 = stablehlo.add %12, %4 : tensor<2xi64>
// CHECK-NEXT:     %14 = stablehlo.reshape %13 : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK-NEXT:     %15 = "stablehlo.scatter"(%iterArg_2, %14, %10) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:     ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:       stablehlo.return %arg4 : tensor<f64>
// CHECK-NEXT:     }) : (tensor<?xf64>, tensor<2x1xi64>, tensor<2xf64>) -> tensor<?xf64>
// CHECK-NEXT:     %16 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %16, %iterArg_1, %15 : tensor<i64>, tensor<?xf64>, tensor<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %1#1, %1#2 : tensor<?xf64>, tensor<?xf64>
// CHECK-NEXT: }

// PLAIN:  func.func @scale(%arg0: memref<?xf64, 1>, %arg1: memref<i32, 1>, %arg2: memref<?xf64, 1>, %arg3: index) {
// PLAIN-NEXT:   %alloca = memref.alloca() : memref<i32>
// PLAIN-NEXT:   %c0_i32 = arith.constant 0 : i32
// PLAIN-NEXT:   %c1 = arith.constant 1 : index
// PLAIN-NEXT:   %c2 = arith.constant 2 : index
// PLAIN-NEXT:   %0 = affine.load %arg1[] : memref<i32, 1>
// PLAIN-NEXT:   %1 = arith.index_cast %0 : i32 to index
// PLAIN-NEXT:   %2 = arith.cmpi sgt, %0, %c0_i32 : i32
// PLAIN-NEXT:   scf.if %2 {
// PLAIN-NEXT:     %memref = gpu.alloc  () : memref<i32, 1>
// PLAIN-NEXT:     affine.store %0, %alloca[] : memref<i32>
// PLAIN-NEXT:     %c4 = arith.constant 4 : index
// PLAIN-NEXT:     enzymexla.memcpy  %memref, %alloca, %c4 : memref<i32, 1>, memref<i32>
// PLAIN-NEXT:     enzymexla.xla_wrapper @rxla$raised_0 (%arg2, %arg0, %memref) : (memref<?xf64, 1>, memref<?xf64, 1>, memref<i32, 1>) -> ()
// PLAIN-NEXT:     %c0 = arith.constant 0 : index
// PLAIN-NEXT:     gpu.dealloc  %memref : memref<i32, 1>
// PLAIN-NEXT:   }
// PLAIN-NEXT:   return
// PLAIN-NEXT: }
// PLAIN-NEXT: func.func private @rxla$raised_0(%arg0: tensor<?xf64>, %arg1: tensor<?xf64>, %arg2: tensor<i32>) -> (tensor<?xf64>, tensor<?xf64>, tensor<i32>) {
// PLAIN-NEXT:   %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<i64>
// PLAIN-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// PLAIN-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i64>
// PLAIN-NEXT:   %1:3 = stablehlo.while(%iterArg = %c, %iterArg_1 = %arg0, %iterArg_2 = %arg1) : tensor<i64>, tensor<?xf64>, tensor<?xf64> attributes {enzymexla.parallel}
// PLAIN-NEXT:   cond {
// PLAIN-NEXT:     %2 = stablehlo.compare LT, %iterArg, %0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// PLAIN-NEXT:     stablehlo.return %2 : tensor<i1>
// PLAIN-NEXT:   } do {
// PLAIN-NEXT:     %2 = stablehlo.iota dim = 0 : tensor<2xi64>
// PLAIN-NEXT:     %c_3 = stablehlo.constant dense<0> : tensor<2xi64>
// PLAIN-NEXT:     %3 = stablehlo.add %2, %c_3 : tensor<2xi64>
// PLAIN-NEXT:     %c_4 = stablehlo.constant dense<1> : tensor<2xi64>
// PLAIN-NEXT:     %4 = stablehlo.multiply %3, %c_4 : tensor<2xi64>
// PLAIN-NEXT:     %c_5 = stablehlo.constant dense<2> : tensor<i64>
// PLAIN-NEXT:     %5 = stablehlo.multiply %iterArg, %c_5 : tensor<i64>
// PLAIN-NEXT:     %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<i64>) -> tensor<2xi64>
// PLAIN-NEXT:     %7 = stablehlo.add %6, %4 : tensor<2xi64>
// PLAIN-NEXT:     %8 = stablehlo.reshape %7 : (tensor<2xi64>) -> tensor<2x1xi64>
// PLAIN-NEXT:     %9 = "stablehlo.gather"(%iterArg_1, %8) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<?xf64>, tensor<2x1xi64>) -> tensor<2xf64>
// PLAIN-NEXT:     %10 = arith.mulf %9, %9 : tensor<2xf64>
// PLAIN-NEXT:     %c_6 = stablehlo.constant dense<2> : tensor<i64>
// PLAIN-NEXT:     %11 = stablehlo.multiply %iterArg, %c_6 : tensor<i64>
// PLAIN-NEXT:     %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<i64>) -> tensor<2xi64>
// PLAIN-NEXT:     %13 = stablehlo.add %12, %4 : tensor<2xi64>
// PLAIN-NEXT:     %14 = stablehlo.reshape %13 : (tensor<2xi64>) -> tensor<2x1xi64>
// PLAIN-NEXT:     %15 = "stablehlo.scatter"(%iterArg_2, %14, %10) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
// PLAIN-NEXT:     ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// PLAIN-NEXT:       stablehlo.return %arg4 : tensor<f64>
// PLAIN-NEXT:     }) : (tensor<?xf64>, tensor<2x1xi64>, tensor<2xf64>) -> tensor<?xf64>
// PLAIN-NEXT:     %16 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// PLAIN-NEXT:     stablehlo.return %16, %iterArg_1, %15 : tensor<i64>, tensor<?xf64>, tensor<?xf64>
// PLAIN-NEXT:   }
// PLAIN-NEXT:   return %1#1, %1#2, %arg2 : tensor<?xf64>, tensor<?xf64>, tensor<i32>
// PLAIN-NEXT: }
