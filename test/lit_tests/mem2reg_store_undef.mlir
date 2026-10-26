// RUN: enzymexlamlir-opt --polygeist-mem2reg %s | FileCheck %s

// A store of undef leaves the slot holding any value, and what it held
// before is one: the load forwards the store before it, not the undef.

func.func @after(%arg0: i32) -> i32 {
  %alloca = memref.alloca() : memref<i32>
  %undef = llvm.mlir.undef : i32
  memref.store %arg0, %alloca[] : memref<i32>
  memref.store %undef, %alloca[] : memref<i32>
  %0 = memref.load %alloca[] : memref<i32>
  return %0 : i32
}

// CHECK:    func.func @after(%arg0: i32) -> i32 {
// CHECK-NEXT:    %0 = llvm.mlir.undef : i32
// CHECK-NEXT:    return %arg0 : i32
// CHECK-NEXT:  }
