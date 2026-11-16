// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo="prefer_while_raising=true" | FileCheck %s

// The element loop runs to `min(ne, chunk)`, an expression of two scalars the
// kernel reads but never writes: both are specialized, and the bound folds to
// a constant once the runtime replaces them.

func.func @scale(%out: memref<?xf64, 1>, %ne: i32, %chunk: i32, %in: memref<?xf64, 1>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %n = arith.minsi %ne, %chunk : i32
  %ni = arith.index_cast %n : i32 to index
  %c0_i32 = arith.constant 0 : i32
  %ok = arith.cmpi sgt, %n, %c0_i32 : i32
  scf.if %ok {
  %0 = "enzymexla.gpu_wrapper"(%ni, %c1, %c1, %c2, %c1, %c1) ({
    %m = arith.minsi %ne, %chunk : i32
    %mi = arith.index_cast %m : i32 to index
    affine.parallel (%e) = (0) to (symbol(%mi)) {
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

// CHECK:  func.func @scale(%arg0: memref<?xf64, 1>, %arg1: i32, %arg2: i32, %arg3: memref<?xf64, 1>) {
// CHECK-NEXT:   %alloca = memref.alloca() : memref<i32>
// CHECK-NEXT:   %alloca_0 = memref.alloca() : memref<i32>
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %c2 = arith.constant 2 : index
// CHECK-NEXT:   %0 = arith.minsi %arg1, %arg2 : i32
// CHECK-NEXT:   %1 = arith.index_cast %0 : i32 to index
// CHECK-NEXT:   %2 = arith.cmpi sgt, %0, %c0_i32 : i32
// CHECK-NEXT:   scf.if %2 {
// CHECK-NEXT:     affine.store %arg1, %alloca_0[] : memref<i32>
// CHECK-NEXT:     %c4 = arith.constant 4 : index
// CHECK-NEXT:     affine.store %arg2, %alloca[] : memref<i32>
// CHECK-NEXT:     %c4_1 = arith.constant 4 : index
// CHECK-NEXT:     enzymexla.xla_wrapper @rxla$raised_0 (%arg3, %arg0, %arg1, %arg2) <num_specialized = 2> : (memref<?xf64, 1>, memref<?xf64, 1>, i32, i32) -> ()
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK:  func.func private @rxla$raised_0(%arg0: tensor<?xf64>, %arg1: tensor<?xf64>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<?xf64>, tensor<?xf64>) {
// CHECK-NEXT:   %0 = arith.minsi %arg2, %arg3 : tensor<i32>
// CHECK-NEXT:   %1 = stablehlo.convert %0 : (tensor<i32>) -> tensor<i64>
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %2:3 = stablehlo.while(%iterArg = %c, %iterArg_1 = %arg0, %iterArg_2 = %arg1) : tensor<i64>, tensor<?xf64>, tensor<?xf64> attributes {enzymexla.parallel}
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %3 = stablehlo.compare LT, %iterArg, %1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %3 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %3 = stablehlo.iota dim = 0 : tensor<2xi64>
// CHECK-NEXT:     %c_3 = stablehlo.constant dense<0> : tensor<2xi64>
// CHECK-NEXT:     %4 = stablehlo.add %3, %c_3 : tensor<2xi64>
// CHECK-NEXT:     %c_4 = stablehlo.constant dense<1> : tensor<2xi64>
// CHECK-NEXT:     %5 = stablehlo.multiply %4, %c_4 : tensor<2xi64>
// CHECK-NEXT:     %c_5 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:     %6 = stablehlo.multiply %iterArg, %c_5 : tensor<i64>
// CHECK-NEXT:     %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:     %8 = stablehlo.add %7, %5 : tensor<2xi64>
// CHECK-NEXT:     %9 = stablehlo.reshape %8 : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK-NEXT:     %10 = "stablehlo.gather"(%iterArg_1, %9) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<?xf64>, tensor<2x1xi64>) -> tensor<2xf64>
// CHECK-NEXT:     %11 = arith.mulf %10, %10 : tensor<2xf64>
// CHECK-NEXT:     %c_6 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:     %12 = stablehlo.multiply %iterArg, %c_6 : tensor<i64>
// CHECK-NEXT:     %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:     %14 = stablehlo.add %13, %5 : tensor<2xi64>
// CHECK-NEXT:     %15 = stablehlo.reshape %14 : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK-NEXT:     %16 = "stablehlo.scatter"(%iterArg_2, %15, %11) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:     ^bb0(%arg4: tensor<f64>, %arg5: tensor<f64>):
// CHECK-NEXT:       stablehlo.return %arg5 : tensor<f64>
// CHECK-NEXT:     }) : (tensor<?xf64>, tensor<2x1xi64>, tensor<2xf64>) -> tensor<?xf64>
// CHECK-NEXT:     %17 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %17, %iterArg_1, %16 : tensor<i64>, tensor<?xf64>, tensor<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %2#1, %2#2 : tensor<?xf64>, tensor<?xf64>
