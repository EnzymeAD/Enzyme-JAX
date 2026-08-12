// RUN: enzymexlamlir-opt %s --canonicalize-parallel -split-input-file | FileCheck %s

// A barrier under a thread-dependent condition must stay there: outside it every
// thread runs it, inside it only the taken ones do, and a thread that skips the
// guard entirely is meant to reach no barrier at all.
module {
  func.func @divergent(%n: index, %m: memref<?xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 1.000000e+00 : f64
    scf.parallel (%i, %j, %k) = (%c0, %c0, %c0) to (%c2, %c2, %c2) step (%c1, %c1, %c1) {
      %d = arith.subi %n, %k : index
      %c = arith.cmpi sge, %d, %c0 : index
      scf.if %c {
        memref.store %cst, %m[%i] : memref<?xf64>
        "enzymexla.barrier"(%i, %j, %k) : (index, index, index) -> ()
      }
      scf.reduce
    }
    return
  }
}

// CHECK-LABEL: func.func @divergent
// CHECK: scf.if
// CHECK-NEXT: memref.store
// CHECK-NEXT: "enzymexla.barrier"

// -----

// A condition no thread index reaches is uniform across the block, so the
// barrier may still be lifted out of it.
module {
  func.func @uniform(%c: i1, %m: memref<?xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 1.000000e+00 : f64
    scf.parallel (%i, %j, %k) = (%c0, %c0, %c0) to (%c2, %c2, %c2) step (%c1, %c1, %c1) {
      scf.if %c {
        memref.store %cst, %m[%i] : memref<?xf64>
        "enzymexla.barrier"(%i, %j, %k) : (index, index, index) -> ()
      }
      scf.reduce
    }
    return
  }
}

// CHECK-LABEL: func.func @uniform
// CHECK: scf.if
// CHECK-NEXT: memref.store
// CHECK-NEXT: }
// CHECK-NEXT: "enzymexla.barrier"
