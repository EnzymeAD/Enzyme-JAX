// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

// A rotated strided-copy do-while (`k = tid; do { copy(k); k += stride; }
// while (k < n)` with the increment folded) yields only pass-throughs of
// condition-forwarded values: every iteration past the second would repeat
// the second exactly, so it unrolls to body(init); if (cond) body(next).

// CHECK-LABEL: @dowhile
// CHECK-NOT: scf.while

module {
  func.func private @dowhile(%buf: memref<32xf64, 1>, %nbuf: memref<1xi32, 1>) {
    %c16 = arith.constant 16 : index
    %cst = arith.constant 2.0 : f64
    %n = affine.load %nbuf[0] : memref<1xi32, 1>
    %ni = arith.index_cast %n : i32 to index
    affine.parallel (%t) = (0) to (16) {
      %r = scf.while (%k = %t) : (index) -> index {
        memref.store %cst, %buf[%k] : memref<32xf64, 1>
        %next = arith.addi %k, %c16 : index
        %cond = arith.cmpi slt, %next, %ni : index
        scf.condition(%cond) %next : index
      } do {
      ^bb0(%j: index):
        scf.yield %j : index
      }
    }
    return
  }
}

// A run-exactly-twice loop: the carried flag starts true and the after
// region yields false — invariant — while carrying a side store between
// the two body halves.

// CHECK-LABEL: @sideflag
// CHECK-NOT: scf.while

module {
  func.func private @sideflag(%out: memref<16xf64, 1>, %side: memref<4xf64, 1>) {
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant 2.0 : f64
    affine.parallel (%t) = (0) to (16) {
      scf.while (%f = %true) : (i1) -> () {
        %v = affine.load %out[%t] : memref<16xf64, 1>
        %a = arith.addf %v, %cst : f64
        affine.store %a, %out[%t] : memref<16xf64, 1>
        scf.condition(%f)
      } do {
        affine.store %cst, %side[0] : memref<4xf64, 1>
        scf.yield %false : i1
      }
    }
    return
  }
}
