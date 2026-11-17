// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s
// RUN: enzymexlamlir-opt %s --sort-memory --raise-affine-to-stablehlo --split-input-file | FileCheck %s --check-prefix=SORT

// Scalar constant LLVM globals (such as Julia's @_j_const_N literals) and
// initializer-region constant globals raise to 1-D stablehlo.constant tensors.

module {
  llvm.mlir.global private unnamed_addr constant @_j_const_1(0.5 : f64) {addr_space = 0 : i32, alignment = 8 : i64} : f64
  func.func private @scalar_attr_global(%out: memref<4xf64, 1>) {
    %g = llvm.mlir.addressof @_j_const_1 : !llvm.ptr
    %m = "enzymexla.pointer2memref"(%g) : (!llvm.ptr) -> memref<?xf64>
    affine.parallel (%t) = (0) to (4) {
      %v = affine.load %m[0] : memref<?xf64>
      affine.store %v, %out[%t] : memref<4xf64, 1>
    }
    return
  }
}

// CHECK-LABEL: func.func private @scalar_attr_global_raised(
// CHECK: %cst = stablehlo.constant dense<5.000000e-01> : tensor<1xf64>
// CHECK: stablehlo.dynamic_update_slice

// -----

module {
  llvm.mlir.global private constant @_j_const_region() {addr_space = 0 : i32} : i32 {
    %c = llvm.mlir.constant(42 : i32) : i32
    llvm.return %c : i32
  }
  func.func private @scalar_region_global(%out: memref<4xi32, 1>) {
    %g = llvm.mlir.addressof @_j_const_region : !llvm.ptr
    %m = "enzymexla.pointer2memref"(%g) : (!llvm.ptr) -> memref<?xi32>
    affine.parallel (%t) = (0) to (4) {
      %v = affine.load %m[0] : memref<?xi32>
      affine.store %v, %out[%t] : memref<4xi32, 1>
    }
    return
  }
}

// CHECK-LABEL: func.func private @scalar_region_global_raised(
// CHECK: %c = stablehlo.constant dense<42> : tensor<1xi32>
// CHECK: stablehlo.dynamic_update_slice

// -----

module {
  llvm.mlir.global private constant @_j_const_pair() {addr_space = 0 : i32} : !llvm.struct<(f64, f64)> {
    %u = llvm.mlir.undef : !llvm.struct<(f64, f64)>
    %c0 = llvm.mlir.constant(1.25 : f64) : f64
    %c1 = llvm.mlir.constant(2.75 : f64) : f64
    %s0 = llvm.insertvalue %c0, %u[0] : !llvm.struct<(f64, f64)>
    %s1 = llvm.insertvalue %c1, %s0[1] : !llvm.struct<(f64, f64)>
    llvm.return %s1 : !llvm.struct<(f64, f64)>
  }
  func.func private @struct_insertvalue_global(%out: memref<2xf64, 1>) {
    %g = llvm.mlir.addressof @_j_const_pair : !llvm.ptr
    %m = "enzymexla.pointer2memref"(%g) : (!llvm.ptr) -> memref<?xf64>
    affine.parallel (%t) = (0) to (2) {
      %v = affine.load %m[%t] : memref<?xf64>
      affine.store %v, %out[%t] : memref<2xf64, 1>
    }
    return
  }
}

// CHECK-LABEL: func.func private @struct_insertvalue_global_raised(
// CHECK: %cst = stablehlo.constant dense<[1.250000e+00, 2.750000e+00]> : tensor<2xf64>
// CHECK: stablehlo.dynamic_update_slice

// -----

// sort-memory must not hoist cross-lane scratch loads above stores across an
// enzymexla.barrier.
func.func @barrier_swap(%out: memref<4xf64, 1>, %in: memref<4xf64, 1>) {
  %scr = memref.alloca() : memref<4xf64>
  affine.parallel (%t) = (0) to (4) {
    %v = affine.load %in[%t] : memref<4xf64, 1>
    affine.store %v, %scr[%t] : memref<4xf64>
    "enzymexla.barrier"(%t) : (index) -> ()
    %x = affine.load %scr[3 - %t] : memref<4xf64>
    affine.store %x, %out[%t] : memref<4xf64, 1>
  }
  return
}

// SORT-LABEL: func.func private @barrier_swap_raised(
// SORT-NOT: enzymexla.barrier
// SORT: stablehlo.reverse
