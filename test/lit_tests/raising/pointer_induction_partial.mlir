// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(raise-affine-to-stablehlo{err_if_not_fully_raised=false})" | FileCheck %s

// The rotated nest a do-while lowering produces: the inner loop carries the
// walking pointer plus a poison-initialized lagging copy that re-yields the
// same advance; the outer loop is carried through the lagging copy's result.
// The pointer machinery rewrites away entirely: a static whole-trip stride
// composes into the affine access map, a runtime stride (dynamic inner trip)
// becomes an arith-indexed access. Raising the remaining scalar nest is a
// separate concern; this file checks the rewrite.

// CHECK-LABEL: @nest
// CHECK: affine.load %{{.*}}[%{{.*}} + %{{.*}} + %{{.*}} * 4] : memref<?xf64, 1>
// CHECK-NOT: llvm.getelementptr

// CHECK-LABEL: @nest_dynstride
// CHECK: arith.muli
// CHECK: memref.load
// CHECK-NOT: llvm.getelementptr

module {
  func.func private @nest(%in: memref<256xf64, 1>, %dst: memref<16xf64, 1>, %nbuf: memref<1xi32, 1>) {
    %c0 = arith.constant 0.0 : f64
    %pz = ub.poison : !llvm.ptr<1>
    %n = affine.load %nbuf[0] : memref<1xi32, 1>
    %ni = arith.index_cast %n : i32 to index
    %p0 = "enzymexla.memref2pointer"(%in) : (memref<256xf64, 1>) -> !llvm.ptr<1>
    affine.parallel (%t) = (0) to (16) {
      %o:2 = affine.for %i = 0 to %ni iter_args(%oacc = %c0, %p = %p0) -> (f64, !llvm.ptr<1>) {
        %r:3 = affine.for %j = 0 to 4 iter_args(%acc = %oacc, %q = %p, %lag = %pz) -> (f64, !llvm.ptr<1>, !llvm.ptr<1>) {
          %view = "enzymexla.pointer2memref"(%q) : (!llvm.ptr<1>) -> memref<?xf64, 1>
          %v = affine.load %view[%t] : memref<?xf64, 1>
          %a = arith.addf %acc, %v : f64
          %adv = llvm.getelementptr %q[1] : (!llvm.ptr<1>) -> !llvm.ptr<1>, f64
          affine.yield %a, %adv, %adv : f64, !llvm.ptr<1>, !llvm.ptr<1>
        }
        affine.yield %r#0, %r#2 : f64, !llvm.ptr<1>
      }
      affine.store %o#0, %dst[%t] : memref<16xf64, 1>
    }
    return
  }
}

module {
  func.func private @nest_dynstride(%in: memref<256xf64, 1>, %dst: memref<16xf64, 1>, %nbuf: memref<1xi32, 1>) {
    %c0 = arith.constant 0.0 : f64
    %pz = ub.poison : !llvm.ptr<1>
    %n = affine.load %nbuf[0] : memref<1xi32, 1>
    %ni = arith.index_cast %n : i32 to index
    %p0 = "enzymexla.memref2pointer"(%in) : (memref<256xf64, 1>) -> !llvm.ptr<1>
    affine.parallel (%t) = (0) to (16) {
      %o:2 = affine.for %i = 0 to 8 iter_args(%oacc = %c0, %p = %p0) -> (f64, !llvm.ptr<1>) {
        %r:3 = affine.for %j = 0 to %ni iter_args(%acc = %oacc, %q = %p, %lag = %pz) -> (f64, !llvm.ptr<1>, !llvm.ptr<1>) {
          %view = "enzymexla.pointer2memref"(%q) : (!llvm.ptr<1>) -> memref<?xf64, 1>
          %v = affine.load %view[%t] : memref<?xf64, 1>
          %a = arith.addf %acc, %v : f64
          %adv = llvm.getelementptr %q[1] : (!llvm.ptr<1>) -> !llvm.ptr<1>, f64
          affine.yield %a, %adv, %adv : f64, !llvm.ptr<1>, !llvm.ptr<1>
        }
        affine.yield %r#0, %r#2 : f64, !llvm.ptr<1>
      }
      affine.store %o#0, %dst[%t] : memref<16xf64, 1>
    }
    return
  }
}
