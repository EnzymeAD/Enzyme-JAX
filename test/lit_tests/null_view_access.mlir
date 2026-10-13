// RUN: enzymexlamlir-opt %s --canonicalize --split-input-file | FileCheck %s

// An optional buffer arriving as null: its accesses sit behind a runtime
// flag, so they can only execute as undefined behavior. Loads through the
// null view read as zero, stores through it drop, and the view dies.

func.func @optional(%out: memref<?xf64>, %flag: i1, %v: f64) {
  %null = llvm.mlir.zero : !llvm.ptr
  %fview = "enzymexla.pointer2memref"(%null) : (!llvm.ptr) -> memref<?xf64>
  %iview = "enzymexla.pointer2memref"(%null) : (!llvm.ptr) -> memref<?xi32>
  scf.if %flag {
    %m = affine.load %fview[3] : memref<?xf64>
    %i = affine.load %iview[1] : memref<?xi32>
    %fi = arith.sitofp %i : i32 to f64
    %s = arith.addf %m, %fi : f64
    affine.store %s, %out[0] : memref<?xf64>
    affine.store %v, %fview[2] : memref<?xf64>
  }
  return
}

// CHECK:  module {
// CHECK-NEXT:  func.func @optional(%arg0: memref<?xf64>, %arg1: i1, %arg2: f64) {
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    scf.if %arg1 {
// CHECK-NEXT:      affine.store %cst, %arg0[0] : memref<?xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// -----

// A pointer loaded through a null view has no arith zero; it reads as llvm
// null. memref accesses fold the same way as affine ones.

func.func @optionalptr(%i: index, %v: i64) -> !llvm.ptr {
  %null = llvm.mlir.zero : !llvm.ptr
  %view = "enzymexla.pointer2memref"(%null) : (!llvm.ptr) -> memref<?x!llvm.ptr>
  %iview = "enzymexla.pointer2memref"(%null) : (!llvm.ptr) -> memref<?xi64>
  %p = memref.load %view[%i] : memref<?x!llvm.ptr>
  memref.store %v, %iview[%i] : memref<?xi64>
  return %p : !llvm.ptr
}

// CHECK:  module {
// CHECK-NEXT:  func.func @optionalptr(%arg0: index, %arg1: i64) -> !llvm.ptr {
// CHECK-NEXT:    %0 = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    return %0 : !llvm.ptr
// CHECK-NEXT:  }

// -----

// An offset off the null folds into the view's index first, then the access
// through the null view folds.

func.func @offset(%out: memref<?xf64>, %v: f64) {
  %null = llvm.mlir.zero : !llvm.ptr
  %p = llvm.getelementptr %null[3] : (!llvm.ptr) -> !llvm.ptr, f64
  %view = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?xf64>
  %m = affine.load %view[1] : memref<?xf64>
  affine.store %m, %out[0] : memref<?xf64>
  affine.store %v, %view[0] : memref<?xf64>
  return
}

// CHECK:  module {
// CHECK-NEXT:  func.func @offset(%arg0: memref<?xf64>, %arg1: f64) {
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    affine.store %cst, %arg0[0] : memref<?xf64>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
