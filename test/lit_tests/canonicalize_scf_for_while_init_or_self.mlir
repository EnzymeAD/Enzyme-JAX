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

// Two carried values start at %n and each iteration resets both to %n or to
// the first, or keeps both, through scf.if results; the loop condition reads
// the second. This is the shape of MFEM's dfem quadrature-function loops once
// clang hoists their element guard: both are %n throughout.

func.func @through_if(%n: i32, %m: i32, %p: memref<?xf64>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %cst = arith.constant 1.0 : f64
  %r:3 = scf.while (%a = %n, %b = %n, %k = %c1) : (i32, i32, i32) -> (i32, i32, i32) {
    %g = arith.cmpi slt, %b, %m : i32
    %ab:2 = scf.if %g -> (i32, i32) {
      %q = arith.cmpi sgt, %m, %c0 : i32
      %x = scf.if %q -> (i32) {
        %i = arith.index_cast %k : i32 to index
        memref.store %cst, %p[%i] : memref<?xf64>
        scf.yield %n : i32
      } else {
        scf.yield %a : i32
      }
      scf.yield %x, %x : i32, i32
    } else {
      scf.yield %a, %b : i32, i32
    }
    %c = arith.cmpi slt, %k, %ab#1 : i32
    scf.condition(%c) %ab#0, %ab#1, %k : i32, i32, i32
  } do {
  ^bb0(%a2: i32, %b2: i32, %k2: i32):
    %k3 = arith.addi %k2, %c1 : i32
    scf.yield %a2, %b2, %k3 : i32, i32, i32
  }
  return
}

// CHECK:      func.func @through_if(%arg0: i32, %arg1: i32, %arg2: memref<?xf64>) {
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:   %cst = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:   %0 = arith.maxsi %arg0, %c1_i32 : i32
// CHECK-NEXT:   %1 = arith.addi %0, %c1_i32 : i32
// CHECK-NEXT:   scf.for %arg3 = %c1_i32 to %1 step %c1_i32  : i32 {
// CHECK-NEXT:     %2 = arith.cmpi slt, %arg0, %arg1 : i32
// CHECK-NEXT:     scf.if %2 {
// CHECK-NEXT:       %3 = arith.cmpi sgt, %arg1, %c0_i32 : i32
// CHECK-NEXT:       scf.if %3 {
// CHECK-NEXT:         %4 = arith.index_cast %arg3 : i32 to index
// CHECK-NEXT:         memref.store %cst, %arg2[%4] : memref<?xf64>
// CHECK-NEXT:       } else {
// CHECK-NEXT:       }
// CHECK-NEXT:     } else {
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// The same once affine-cfg has made the branches affine.if: the kernel
// re-reads its bound from a by-value argument, which is only the host's read
// (the init) after parallel-licm, late in the pipeline.

func.func @through_affine_if(%n: i32, %m: i32, %q: index, %p: memref<?xf64>) {
  %c1 = arith.constant 1 : i32
  %cst = arith.constant 1.0 : f64
  %r:3 = scf.while (%a = %n, %b = %n, %k = %c1) : (i32, i32, i32) -> (i32, i32, i32) {
    %g = arith.cmpi slt, %b, %m : i32
    %ab:2 = scf.if %g -> (i32, i32) {
      %x = affine.if affine_set<()[s0] : (s0 - 1 >= 0)>()[%q] -> i32 {
        %i = arith.index_cast %k : i32 to index
        memref.store %cst, %p[%i] : memref<?xf64>
        affine.yield %n : i32
      } else {
        affine.yield %a : i32
      }
      scf.yield %x, %x : i32, i32
    } else {
      scf.yield %a, %b : i32, i32
    }
    %c = arith.cmpi slt, %k, %ab#1 : i32
    scf.condition(%c) %ab#0, %ab#1, %k : i32, i32, i32
  } do {
  ^bb0(%a2: i32, %b2: i32, %k2: i32):
    %k3 = arith.addi %k2, %c1 : i32
    scf.yield %a2, %b2, %k3 : i32, i32, i32
  }
  return
}

// CHECK:      func.func @through_affine_if(%arg0: i32, %arg1: i32, %arg2: index, %arg3: memref<?xf64>) {
// CHECK-NEXT:   %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:   %cst = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:   %0 = arith.maxsi %arg0, %c1_i32 : i32
// CHECK-NEXT:   %1 = arith.addi %0, %c1_i32 : i32
// CHECK-NEXT:   scf.for %arg4 = %c1_i32 to %1 step %c1_i32  : i32 {
// CHECK-NEXT:     %2 = arith.cmpi slt, %arg0, %arg1 : i32
// CHECK-NEXT:     scf.if %2 {
// CHECK-NEXT:       affine.if #set()[%arg2] {
// CHECK-NEXT:         %3 = arith.index_cast %arg4 : i32 to index
// CHECK-NEXT:         memref.store %cst, %arg3[%3] : memref<?xf64>
// CHECK-NEXT:       } else {
// CHECK-NEXT:       }
// CHECK-NEXT:     } else {
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// An affine.if result both branches yield the same value for is that value;
// a result they differ on stays.

func.func @affine_if_same_yield(%n: i32, %m: i32, %q: index, %p: memref<?xf64>) -> (i32, i32) {
  %cst = arith.constant 1.0 : f64
  %r:2 = affine.if affine_set<()[s0] : (s0 - 1 >= 0)>()[%q] -> (i32, i32) {
    memref.store %cst, %p[%q] : memref<?xf64>
    affine.yield %n, %n : i32, i32
  } else {
    affine.yield %n, %m : i32, i32
  }
  return %r#0, %r#1 : i32, i32
}

// CHECK:      func.func @affine_if_same_yield(%arg0: i32, %arg1: i32, %arg2: index, %arg3: memref<?xf64>) -> (i32, i32) {
// CHECK-NEXT:   %cst = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:   %0 = affine.if #set()[%arg2] -> i32 {
// CHECK-NEXT:     memref.store %cst, %arg3[%arg2] : memref<?xf64>
// CHECK-NEXT:     affine.yield %arg0 : i32
// CHECK-NEXT:   } else {
// CHECK-NEXT:     affine.yield %arg1 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   return %arg0, %0 : i32, i32
// CHECK-NEXT: }
