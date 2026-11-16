// RUN: enzymexlamlir-opt %s --canonicalize-parallel | FileCheck %s

// A store of undef or poison does nothing: the memory may hold any value
// afterwards, and what it held before is one. Stores of defined values stay.

func.func @padding(%arg0: !llvm.ptr, %arg1: memref<?xi32>, %arg2: i32) {
  %undef = llvm.mlir.undef : i32
  %poison = ub.poison : i32
  %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi32>
  affine.store %undef, %0[3] : memref<?xi32>
  affine.store %arg2, %0[4] : memref<?xi32>
  %c0 = arith.constant 0 : index
  memref.store %poison, %arg1[%c0] : memref<?xi32>
  memref.store %arg2, %arg1[%c0] : memref<?xi32>
  return
}

// CHECK:    func.func @padding(%arg0: !llvm.ptr, %arg1: memref<?xi32>, %arg2: i32) {
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi32>
// CHECK-NEXT:    affine.store %arg2, %0[4] : memref<?xi32>
// CHECK-NEXT:    memref.store %arg2, %arg1[%c0] : memref<?xi32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
