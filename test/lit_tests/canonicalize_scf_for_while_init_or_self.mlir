// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(canonicalize-scf-for)" | FileCheck %s

// %a starts at %n and each iteration either keeps it or resets it to %n, so
// it is %n throughout, and the loop counts %k from 1 while %k < %n. This is
// the shape MFEM's `for (int qz = 0; qz < Q1D; qz++)` around
// MFEM_FOREACH_THREAD_DIRECT takes once clang hoists the thread guard.

func.func @init_or_self(%n: i32, %m: i32, %p: memref<?xf64>) {
  %c1 = arith.constant 1 : i32
  %cst = arith.constant 1.0 : f64
  %r:2 = scf.while (%a = %n, %k = %c1) : (i32, i32) -> (i32, i32) {
    %g = arith.cmpi sgt, %a, %m : i32
    %s = arith.select %g, %n, %a : i32
    scf.if %g {
      %i = arith.index_cast %k : i32 to index
      memref.store %cst, %p[%i] : memref<?xf64>
    }
    %c = arith.cmpi slt, %k, %s : i32
    scf.condition(%c) %s, %k : i32, i32
  } do {
  ^bb0(%a2: i32, %k2: i32):
    %k3 = arith.addi %k2, %c1 : i32
    scf.yield %a2, %k3 : i32, i32
  }
  return
}

// CHECK:      func.func @init_or_self(%arg0: i32, %arg1: i32, %arg2: memref<?xf64>) {
// CHECK-NEXT:   %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:   %cst = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:   %0 = arith.maxsi %arg0, %c1_i32 : i32
// CHECK-NEXT:   %1 = arith.addi %0, %c1_i32 : i32
// CHECK-NEXT:   scf.for %arg3 = %c1_i32 to %1 step %c1_i32  : i32 {
// CHECK-NEXT:     %2 = arith.cmpi sgt, %arg0, %arg1 : i32
// CHECK-NEXT:     scf.if %2 {
// CHECK-NEXT:       %3 = arith.index_cast %arg3 : i32 to index
// CHECK-NEXT:       memref.store %cst, %arg2[%3] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
