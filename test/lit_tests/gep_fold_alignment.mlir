// RUN: enzymexlamlir-opt %s --canonicalize | FileCheck %s

// Folding a getelementptr into the index of an access through a view of it
// keeps the alignment the access promised: it is the same address through
// another base. Without it the access is emitted at the element's ABI
// alignment -- an i128 (16) behind a pointer the access knows only 8 of.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>>} {
  func.func @store_wide(%base: !llvm.ptr, %v: i128) {
    %c0 = arith.constant 0 : index
    %g = llvm.getelementptr inbounds|nuw %base[256] : (!llvm.ptr) -> !llvm.ptr, i8
    %m = "enzymexla.pointer2memref"(%g) : (!llvm.ptr) -> memref<?xi128>
    memref.store %v, %m[%c0] alignment(8) : memref<?xi128>
    return
  }

  func.func @load_wide(%base: !llvm.ptr) -> i128 {
    %c0 = arith.constant 0 : index
    %g = llvm.getelementptr inbounds|nuw %base[256] : (!llvm.ptr) -> !llvm.ptr, i8
    %m = "enzymexla.pointer2memref"(%g) : (!llvm.ptr) -> memref<?xi128>
    %v = memref.load %m[%c0] alignment(8) : memref<?xi128>
    return %v : i128
  }

  func.func @affine_store_wide(%base: !llvm.ptr, %v: i128) {
    %g = llvm.getelementptr inbounds|nuw %base[256] : (!llvm.ptr) -> !llvm.ptr, i8
    %m = "enzymexla.pointer2memref"(%g) : (!llvm.ptr) -> memref<?xi128>
    affine.store %v, %m[0] {alignment = 8 : i64} : memref<?xi128>
    return
  }

  func.func @affine_load_wide(%base: !llvm.ptr) -> i128 {
    %g = llvm.getelementptr inbounds|nuw %base[256] : (!llvm.ptr) -> !llvm.ptr, i8
    %m = "enzymexla.pointer2memref"(%g) : (!llvm.ptr) -> memref<?xi128>
    %v = affine.load %m[0] {alignment = 8 : i64} : memref<?xi128>
    return %v : i128
  }
}

// CHECK-LABEL: func.func @store_wide(%arg0: !llvm.ptr, %arg1: i128) {
// CHECK-NEXT:   %c16 = arith.constant 16 : index
// CHECK-NEXT:   %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi128>
// CHECK-NEXT:   memref.store %arg1, %0[%c16] alignment(8) : memref<?xi128>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK-LABEL: func.func @load_wide(%arg0: !llvm.ptr) -> i128 {
// CHECK-NEXT:   %c16 = arith.constant 16 : index
// CHECK-NEXT:   %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi128>
// CHECK-NEXT:   %1 = memref.load %0[%c16] alignment(8) : memref<?xi128>
// CHECK-NEXT:   return %1 : i128
// CHECK-NEXT: }

// CHECK-LABEL: func.func @affine_store_wide(%arg0: !llvm.ptr, %arg1: i128) {
// CHECK-NEXT:   %c16 = arith.constant 16 : index
// CHECK-NEXT:   %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi128>
// CHECK-NEXT:   memref.store %arg1, %0[%c16] alignment(8) : memref<?xi128>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK-LABEL: func.func @affine_load_wide(%arg0: !llvm.ptr) -> i128 {
// CHECK-NEXT:   %c16 = arith.constant 16 : index
// CHECK-NEXT:   %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi128>
// CHECK-NEXT:   %1 = memref.load %0[%c16] alignment(8) : memref<?xi128>
// CHECK-NEXT:   return %1 : i128
// CHECK-NEXT: }
