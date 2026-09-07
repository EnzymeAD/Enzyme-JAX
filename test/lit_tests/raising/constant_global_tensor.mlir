// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s

// A kernel-local constant table promoted to a rodata global reads as a
// stablehlo.constant of its initializer: constant indices slice it, runtime
// indices gather from it, and the raised kernel never captures its address.

module {
  llvm.mlir.global private constant @map(dense<[0, 3, 6, 1]> : tensor<4xi32>) {addr_space = 0 : i32} : !llvm.array<4 x i32>
  func.func private @table(%out: memref<16xf64, 1>) {
    %g = llvm.mlir.addressof @map : !llvm.ptr
    %m = "enzymexla.pointer2memref"(%g) : (!llvm.ptr) -> memref<?xi32>
    affine.parallel (%t) = (0) to (16) {
      %c = affine.load %m[2] : memref<?xi32>
      %d = affine.load %m[%t mod 4] : memref<?xi32>
      %s = arith.addi %c, %d : i32
      %f = arith.sitofp %s : i32 to f64
      affine.store %f, %out[%t] : memref<16xf64, 1>
    }
    return
  }
}

// CHECK:  module {
// CHECK-NEXT:  llvm.mlir.global private constant @map(dense<[0, 3, 6, 1]> : tensor<4xi32>) {addr_space = 0 : i32} : !llvm.array<4 x i32>
// CHECK-NEXT:  func.func private @table_raised(%arg0: tensor<16xf64>) -> tensor<16xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<[0, 3, 6, 1]> : tensor<4xi32>
// CHECK-NEXT:    %0 = stablehlo.bitcast_convert %c : (tensor<4xi32>) -> tensor<4xi32>
// CHECK-NEXT:    %1 = stablehlo.iota dim = 0 : tensor<16xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<16xi64>
// CHECK-NEXT:    %2 = stablehlo.add %1, %c_0 : tensor<16xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<1> : tensor<16xi64>
// CHECK-NEXT:    %3 = stablehlo.multiply %2, %c_1 : tensor<16xi64>
// CHECK-NEXT:    %4 = stablehlo.slice %0 [2:3] : (tensor<4xi32>) -> tensor<1xi32>
// CHECK-NEXT:    %5 = stablehlo.reshape %4 : (tensor<1xi32>) -> tensor<i32>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:    %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<16xi64>
// CHECK-NEXT:    %7 = stablehlo.remainder %3, %6 : tensor<16xi64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<0> : tensor<16xi64>
// CHECK-NEXT:    %8 = stablehlo.compare LT, %7, %c_3 : (tensor<16xi64>, tensor<16xi64>) -> tensor<16xi1>
// CHECK-NEXT:    %9 = stablehlo.add %7, %6 : tensor<16xi64>
// CHECK-NEXT:    %10 = stablehlo.select %8, %9, %7 : tensor<16xi1>, tensor<16xi64>
// CHECK-NEXT:    %11 = stablehlo.reshape %10 : (tensor<16xi64>) -> tensor<16x1xi64>
// CHECK-NEXT:    %12 = "stablehlo.gather"(%0, %11) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<4xi32>, tensor<16x1xi64>) -> tensor<16xi32>
// CHECK-NEXT:    %13 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<i32>) -> tensor<16xi32>
// CHECK-NEXT:    %14 = arith.addi %13, %12 : tensor<16xi32>
// CHECK-NEXT:    %15 = arith.sitofp %14 : tensor<16xi32> to tensor<16xf64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_8 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_9 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %16 = stablehlo.broadcast_in_dim %15, dims = [0] : (tensor<16xf64>) -> tensor<16xf64>
// CHECK-NEXT:    %17 = stablehlo.dynamic_update_slice %arg0, %16, %c_9 : (tensor<16xf64>, tensor<16xf64>, tensor<i64>) -> tensor<16xf64>
// CHECK-NEXT:    return %17 : tensor<16xf64>
// CHECK-NEXT:  }

// -----

// In the host form the view of the table is taken outside the wrapper; it
// moves inside instead of becoming a kernel operand.

module {
  llvm.mlir.global private constant @map(dense<[0, 3, 6, 1, 4, 7, 2, 5, 8]> : tensor<9xi32>) {addr_space = 0 : i32} : !llvm.array<9 x i32>
  func.func @host(%out: memref<9xf64, 1>, %in: memref<9xf64, 1>, %grid: index) {
    %c1 = arith.constant 1 : index
    %g = llvm.mlir.addressof @map : !llvm.ptr
    %m = "enzymexla.pointer2memref"(%g) : (!llvm.ptr) -> memref<?xi32>
    "enzymexla.gpu_wrapper"(%grid, %c1, %c1, %c1, %c1, %c1) ({
      affine.parallel (%i) = (0) to (9) {
        %j = affine.load %m[%i] : memref<?xi32>
        %ji = arith.index_cast %j : i32 to index
        %v = memref.load %in[%ji] : memref<9xf64, 1>
        affine.store %v, %out[%i] : memref<9xf64, 1>
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// CHECK:  module {
// CHECK-NEXT:  llvm.mlir.global private constant @map(dense<[0, 3, 6, 1, 4, 7, 2, 5, 8]> : tensor<9xi32>) {addr_space = 0 : i32} : !llvm.array<9 x i32>
// CHECK-NEXT:  func.func @host(%arg0: memref<9xf64, 1>, %arg1: memref<9xf64, 1>, %arg2: index) {
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %0 = llvm.mlir.addressof @map : !llvm.ptr
// CHECK-NEXT:    %1 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr) -> memref<?xi32>
// CHECK-NEXT:    enzymexla.xla_wrapper @rxla$raised_0 (%arg1, %arg0) : (memref<9xf64, 1>, memref<9xf64, 1>) -> ()
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func private @rxla$raised_0(%arg0: tensor<9xf64>, %arg1: tensor<9xf64>) -> (tensor<9xf64>, tensor<9xf64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<[0, 3, 6, 1, 4, 7, 2, 5, 8]> : tensor<9xi32>
// CHECK-NEXT:    %0 = stablehlo.bitcast_convert %c : (tensor<9xi32>) -> tensor<9xi32>
// CHECK-NEXT:    %1 = stablehlo.iota dim = 0 : tensor<9xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<9xi64>
// CHECK-NEXT:    %2 = stablehlo.add %1, %c_0 : tensor<9xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<1> : tensor<9xi64>
// CHECK-NEXT:    %3 = stablehlo.multiply %2, %c_1 : tensor<9xi64>
// CHECK-NEXT:    %4 = stablehlo.reshape %0 : (tensor<9xi32>) -> tensor<9xi32>
// CHECK-NEXT:    %5 = stablehlo.convert %4 : (tensor<9xi32>) -> tensor<9xi64>
// CHECK-NEXT:    %6 = stablehlo.reshape %5 : (tensor<9xi64>) -> tensor<9x1xi64>
// CHECK-NEXT:    %7 = "stablehlo.gather"(%arg0, %6) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<9xf64>, tensor<9x1xi64>) -> tensor<9xf64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<9xf64>) -> tensor<9xf64>
// CHECK-NEXT:    %9 = stablehlo.dynamic_update_slice %arg1, %8, %c_7 : (tensor<9xf64>, tensor<9xf64>, tensor<i64>) -> tensor<9xf64>
// CHECK-NEXT:    return %arg0, %9 : tensor<9xf64>, tensor<9xf64>
// CHECK-NEXT:  }

// -----

// The view may also be taken inside the wrapper, leaving the table's address
// as the operand: the same move applies to the pointer.

module {
  llvm.mlir.global private constant @map(dense<[0, 3, 6, 1, 4, 7, 2, 5, 8]> : tensor<9xi32>) {addr_space = 0 : i32} : !llvm.array<9 x i32>
  func.func @host_ptr(%out: memref<9xf64, 1>, %in: memref<9xf64, 1>, %grid: index) {
    %c1 = arith.constant 1 : index
    %g = llvm.mlir.addressof @map : !llvm.ptr
    "enzymexla.gpu_wrapper"(%grid, %c1, %c1, %c1, %c1, %c1) ({
      %m = "enzymexla.pointer2memref"(%g) : (!llvm.ptr) -> memref<?xi32>
      affine.parallel (%i) = (0) to (9) {
        %j = affine.load %m[%i] : memref<?xi32>
        %ji = arith.index_cast %j : i32 to index
        %v = memref.load %in[%ji] : memref<9xf64, 1>
        affine.store %v, %out[%i] : memref<9xf64, 1>
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// CHECK:  module {
// CHECK-NEXT:  llvm.mlir.global private constant @map(dense<[0, 3, 6, 1, 4, 7, 2, 5, 8]> : tensor<9xi32>) {addr_space = 0 : i32} : !llvm.array<9 x i32>
// CHECK-NEXT:  func.func @host_ptr(%arg0: memref<9xf64, 1>, %arg1: memref<9xf64, 1>, %arg2: index) {
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %0 = llvm.mlir.addressof @map : !llvm.ptr
// CHECK-NEXT:    enzymexla.xla_wrapper @rxla$raised_0 (%arg1, %arg0) : (memref<9xf64, 1>, memref<9xf64, 1>) -> ()
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func private @rxla$raised_0(%arg0: tensor<9xf64>, %arg1: tensor<9xf64>) -> (tensor<9xf64>, tensor<9xf64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<[0, 3, 6, 1, 4, 7, 2, 5, 8]> : tensor<9xi32>
// CHECK-NEXT:    %0 = stablehlo.bitcast_convert %c : (tensor<9xi32>) -> tensor<9xi32>
// CHECK-NEXT:    %1 = stablehlo.iota dim = 0 : tensor<9xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<9xi64>
// CHECK-NEXT:    %2 = stablehlo.add %1, %c_0 : tensor<9xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<1> : tensor<9xi64>
// CHECK-NEXT:    %3 = stablehlo.multiply %2, %c_1 : tensor<9xi64>
// CHECK-NEXT:    %4 = stablehlo.reshape %0 : (tensor<9xi32>) -> tensor<9xi32>
// CHECK-NEXT:    %5 = stablehlo.convert %4 : (tensor<9xi32>) -> tensor<9xi64>
// CHECK-NEXT:    %6 = stablehlo.reshape %5 : (tensor<9xi64>) -> tensor<9x1xi64>
// CHECK-NEXT:    %7 = "stablehlo.gather"(%arg0, %6) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<9xf64>, tensor<9x1xi64>) -> tensor<9xf64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<9xf64>) -> tensor<9xf64>
// CHECK-NEXT:    %9 = stablehlo.dynamic_update_slice %arg1, %8, %c_7 : (tensor<9xf64>, tensor<9xf64>, tensor<i64>) -> tensor<9xf64>
// CHECK-NEXT:    return %arg0, %9 : tensor<9xf64>, tensor<9xf64>
// CHECK-NEXT:  }
