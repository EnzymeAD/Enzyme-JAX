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
