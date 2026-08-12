// RUN: enzymexlamlir-opt %s --canonicalize-parallel -split-input-file | FileCheck %s

// One barrier, and every thread runs it once it is out here: lifting it is a
// legalization, not a miscompile.
module {
  func.func @lone(%n: index, %m: memref<?xf64>) {
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

// CHECK-LABEL: func.func @lone
// CHECK: scf.if
// CHECK-NEXT: memref.store
// CHECK-NEXT: }
// CHECK-NEXT: "enzymexla.barrier"

// -----

// A second barrier stays behind whatever happens to this one -- the store
// between them pins it -- so lifting only the trailing one leaves a thread that
// skips the branch waiting out here while a thread that takes it waits in
// there, neither ever reaching the other's.
module {
  func.func @two(%n: index, %m: memref<?xf64>) {
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
        memref.store %cst, %m[%j] : memref<?xf64>
        "enzymexla.barrier"(%i, %j, %k) : (index, index, index) -> ()
      }
      scf.reduce
    }
    return
  }
}

// CHECK-LABEL: func.func @two
// CHECK: scf.if
// CHECK: "enzymexla.barrier"
// CHECK: memref.store
// CHECK-NEXT: "enzymexla.barrier"
// CHECK-NEXT: }

// -----

// The condition reads a serial loop's index, and that loop's own bound is a
// thread index: a block argument from inside the thread parallel says nothing
// about how the threads decide the branch.
module {
  func.func @serial_iv(%m: memref<?xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 1.000000e+00 : f64
    scf.parallel (%i, %j, %k) = (%c0, %c0, %c0) to (%c2, %c2, %c2) step (%c1, %c1, %c1) {
      scf.for %t = %c0 to %k step %c1 {
        %c = arith.cmpi sge, %t, %c1 : index
        scf.if %c {
          memref.store %cst, %m[%i] : memref<?xf64>
          "enzymexla.barrier"(%i, %j, %k) : (index, index, index) -> ()
          memref.store %cst, %m[%j] : memref<?xf64>
          "enzymexla.barrier"(%i, %j, %k) : (index, index, index) -> ()
        }
      }
      scf.reduce
    }
    return
  }
}

// CHECK-LABEL: func.func @serial_iv
// CHECK: scf.if
// CHECK: "enzymexla.barrier"
// CHECK: memref.store
// CHECK-NEXT: "enzymexla.barrier"
// CHECK-NEXT: }
