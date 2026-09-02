// RUN: enzymexlamlir-opt %s --llvm-to-affine-access | FileCheck %s

// An optional buffer arriving as null: its accesses sit behind a runtime
// flag, so they can only execute as undefined behavior. Loads fold to zero,
// stores drop, and the null views disappear.
module {
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
}

// CHECK-LABEL: func.func @optional(
// CHECK-SAME: %[[OUT:.+]]: memref<?xf64>, %[[FLAG:.+]]: i1, %[[V:.+]]: f64
// CHECK-DAG: %[[FZ:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NOT: llvm.mlir.zero
// CHECK-NOT: pointer2memref
// CHECK: scf.if %[[FLAG]] {
// CHECK-NEXT: affine.store %[[FZ]], %[[OUT]][0] : memref<?xf64>
// CHECK-NEXT: }
// CHECK-NEXT: return

// A pointer loaded from a null view has no arith zero attribute; it folds to
// llvm null instead.
// CHECK-LABEL: func.func @optionalptr(
// CHECK: llvm.mlir.zero
// CHECK-NOT: arith.constant {{.*}} !llvm.ptr
func.func @optionalptr(%flag: i1) -> !llvm.ptr {
  %null = llvm.mlir.zero : !llvm.ptr
  %view = "enzymexla.pointer2memref"(%null) : (!llvm.ptr) -> memref<?x!llvm.ptr>
  %p = affine.load %view[3] : memref<?x!llvm.ptr>
  return %p : !llvm.ptr
}

// The same through raw llvm accesses at an offset from the null.
// CHECK-LABEL: func.func @offset(
// CHECK-SAME: %[[OUT:.+]]: memref<?xf64>, %[[FLAG:.+]]: i1, %[[V:.+]]: f64
// CHECK-DAG: %[[FZ:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NOT: llvm.load
// CHECK-NOT: llvm.store
// CHECK: affine.store %[[FZ]], %[[OUT]][0] : memref<?xf64>
func.func @offset(%out: memref<?xf64>, %flag: i1, %v: f64) {
  %null = llvm.mlir.zero : !llvm.ptr
  scf.if %flag {
    %p = llvm.getelementptr %null[3] : (!llvm.ptr) -> !llvm.ptr, f64
    %m = llvm.load %p : !llvm.ptr -> f64
    affine.store %m, %out[0] : memref<?xf64>
    llvm.store %v, %p : f64, !llvm.ptr
  }
  return
}

// A null chosen against a real pointer is not an access through the null:
// the load through the real buffer survives, only the null arm folds.
// CHECK-LABEL: func.func @chosen(
// CHECK-SAME: %[[BUF:.+]]: !llvm.ptr, %[[FLAG:.+]]: i1
// CHECK: %[[Z:.+]] = arith.constant 0.000000e+00 : f64
// CHECK: %[[M:.+]] = "enzymexla.pointer2memref"(%[[BUF]])
// CHECK: %[[R:.+]] = scf.if %[[FLAG]] -> (f64) {
// CHECK:   %[[L:.+]] = affine.load %[[M]][0]
// CHECK:   scf.yield %[[L]]
// CHECK: } else {
// CHECK:   scf.yield %[[Z]]
// CHECK: }
// CHECK: return %[[R]]
func.func @chosen(%buf: !llvm.ptr, %flag: i1) -> f64 {
  %null = llvm.mlir.zero : !llvm.ptr
  %p = arith.select %flag, %buf, %null : !llvm.ptr
  %m = llvm.load %p : !llvm.ptr -> f64
  return %m : f64
}
