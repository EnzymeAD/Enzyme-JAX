// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

// A loop that walks a pointer forward by a constant stride each iteration
// carries it as an iter arg no tensor can stand for; the pointer is a pure
// function of the induction variable, so accesses rebase onto the init
// pointer and the carried pointer disappears.

// CHECK-LABEL: @walk_raised
// CHECK: stablehlo.while
// CHECK-NOT: !llvm.ptr

module {
  func.func private @walk(%out: memref<16xf64, 1>, %in: memref<64xf64, 1>, %nbuf: memref<1xi32, 1>) {
    %c0 = arith.constant 0.0 : f64
    %n = affine.load %nbuf[0] : memref<1xi32, 1>
    %ni = arith.index_cast %n : i32 to index
    %p0 = "enzymexla.memref2pointer"(%in) : (memref<64xf64, 1>) -> !llvm.ptr<1>
    affine.parallel (%t) = (0) to (16) {
      %sum:2 = affine.for %i = 0 to %ni iter_args(%acc = %c0, %p = %p0) -> (f64, !llvm.ptr<1>) {
        %view = "enzymexla.pointer2memref"(%p) : (!llvm.ptr<1>) -> memref<?xf64, 1>
        %v = affine.load %view[%t] : memref<?xf64, 1>
        %a = arith.addf %acc, %v : f64
        %adv = llvm.getelementptr %p[1] : (!llvm.ptr<1>) -> !llvm.ptr<1>, f64
        affine.yield %a, %adv : f64, !llvm.ptr<1>
      }
      affine.store %sum#0, %out[%t] : memref<16xf64, 1>
    }
    return
  }
}

// A used final pointer is the init advanced by the whole trip count; the
// following access rebases onto the init with the trip folded in.

// CHECK-LABEL: @final_used
// CHECK: stablehlo.while
// CHECK-NOT: !llvm.ptr

module {
  func.func private @final_used(%in: memref<64xf64, 1>, %dst: memref<16xf64, 1>, %nbuf: memref<1xi32, 1>) {
    %c0 = arith.constant 0.0 : f64
    %n = affine.load %nbuf[0] : memref<1xi32, 1>
    %ni = arith.index_cast %n : i32 to index
    %p0 = "enzymexla.memref2pointer"(%in) : (memref<64xf64, 1>) -> !llvm.ptr<1>
    affine.parallel (%t) = (0) to (16) {
      %r:2 = affine.for %i = 0 to %ni iter_args(%acc = %c0, %p = %p0) -> (f64, !llvm.ptr<1>) {
        %view = "enzymexla.pointer2memref"(%p) : (!llvm.ptr<1>) -> memref<?xf64, 1>
        %v = affine.load %view[%t] : memref<?xf64, 1>
        %a = arith.addf %acc, %v : f64
        %adv = llvm.getelementptr %p[1] : (!llvm.ptr<1>) -> !llvm.ptr<1>, f64
        affine.yield %a, %adv : f64, !llvm.ptr<1>
      }
      %viewf = "enzymexla.pointer2memref"(%r#1) : (!llvm.ptr<1>) -> memref<?xf64, 1>
      %w = affine.load %viewf[%t] : memref<?xf64, 1>
      %b = arith.addf %r#0, %w : f64
      affine.store %b, %dst[%t] : memref<16xf64, 1>
    }
    return
  }
}

