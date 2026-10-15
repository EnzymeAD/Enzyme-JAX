// RUN: enzymexlamlir-opt %s -allow-unregistered-dialect --canonicalize | FileCheck %s

// An allocation in a serial loop leaves it, out of every serial loop around
// it up to the function; one in a parallel loop stays private to each
// iteration. The enzymexla op only loads the dialect whose canonicalization
// patterns these are.

func.func @nested(%n: index, %b: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    affine.for %j = 0 to 4 {
      %a = memref.alloca() : memref<9xi1>
      memref.store %b, %a[%c0] : memref<9xi1>
      %p = "enzymexla.memref2pointer"(%a) : (memref<9xi1>) -> !llvm.ptr
      "test.use"(%p) : (!llvm.ptr) -> ()
    }
  }
  return
}

func.func @parallel(%n: index, %b: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%i) = (%c0) to (%n) step (%c1) {
    %a = memref.alloca() : memref<9xi1>
    memref.store %b, %a[%c0] : memref<9xi1>
    %p = "enzymexla.memref2pointer"(%a) : (memref<9xi1>) -> !llvm.ptr
    "test.use"(%p) : (!llvm.ptr) -> ()
  }
  return
}

// The size it reads is computed in the loop from values outside it: it
// leaves with the allocation.
func.func @sized(%n: index, %m: index, %b: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %s = arith.addi %m, %c1 : index
    %a = memref.alloca(%s) : memref<?xi1>
    memref.store %b, %a[%c0] : memref<?xi1>
    %p = "enzymexla.memref2pointer"(%a) : (memref<?xi1>) -> !llvm.ptr
    "test.use"(%p) : (!llvm.ptr) -> ()
  }
  return
}

// CHECK: func.func @nested(%arg0: index, %arg1: i1) {
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %alloca = memref.alloca() : memref<9xi1>
// CHECK-NEXT:   scf.for %arg2 = %c0 to %arg0 step %c1 {
// CHECK-NEXT:     affine.for %arg3 = 0 to 4 {
// CHECK-NEXT:       memref.store %arg1, %alloca[%c0] : memref<9xi1>
// CHECK-NEXT:       %0 = "enzymexla.memref2pointer"(%alloca) : (memref<9xi1>) -> !llvm.ptr
// CHECK-NEXT:       "test.use"(%0) : (!llvm.ptr) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK: func.func @parallel(%arg0: index, %arg1: i1) {
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   scf.parallel (%arg2) = (%c0) to (%arg0) step (%c1) {
// CHECK-NEXT:     %alloca = memref.alloca() : memref<9xi1>
// CHECK-NEXT:     memref.store %arg1, %alloca[%c0] : memref<9xi1>
// CHECK-NEXT:     %0 = "enzymexla.memref2pointer"(%alloca) : (memref<9xi1>) -> !llvm.ptr
// CHECK-NEXT:     "test.use"(%0) : (!llvm.ptr) -> ()
// CHECK-NEXT:     scf.reduce
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK: func.func @sized(%arg0: index, %arg1: index, %arg2: i1) {
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %0 = arith.addi %arg1, %c1 : index
// CHECK-NEXT:   %alloca = memref.alloca(%0) : memref<?xi1>
// CHECK-NEXT:   scf.for %arg3 = %c0 to %arg0 step %c1 {
// CHECK-NEXT:     memref.store %arg2, %alloca[%c0] : memref<?xi1>
// CHECK-NEXT:     %1 = "enzymexla.memref2pointer"(%alloca) : (memref<?xi1>) -> !llvm.ptr
// CHECK-NEXT:     "test.use"(%1) : (!llvm.ptr) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
