// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access)" | FileCheck %s

// Rebuilding the byte allocation at the pointer2memref's element type keeps
// the allocation's memory space, while the pointer2memref names space 0.
// memref.cast can reshape but not change the memory space; the space change
// is said as the memory_space_cast it is.

module {
  func.func @spacealloc(%v: f64) -> f64 {
    %sh = memref.alloca() : memref<512xi8, 5>
    %p = "enzymexla.memref2pointer"(%sh) : (memref<512xi8, 5>) -> !llvm.ptr
    %m = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?xf64>
    %c0 = arith.constant 0 : index
    memref.store %v, %m[%c0] : memref<?xf64>
    %x = memref.load %m[%c0] : memref<?xf64>
    return %x : f64
  }
}

// CHECK-LABEL: func.func @spacealloc
// CHECK: %[[SH:.+]] = memref.alloca() : memref<64xf64, 5>
// CHECK: %[[MR:.+]] = memref.memory_space_cast %[[SH]] : memref<64xf64, 5> to memref<64xf64>
// CHECK: memref.store %{{.+}}, %[[MR]][%{{.+}}] : memref<64xf64>
// CHECK: memref.load %[[MR]][%{{.+}}] : memref<64xf64>
