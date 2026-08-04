// RUN: enzymexlamlir-opt --polygeist-mem2reg --split-input-file %s | FileCheck %s
// XFAIL: *
// A byte-counted transfer meeting a dimension-counted load still does not
// forward; the zero-offset fix does not reach this path.

// The shape XSBench's seed array has: everything but element 0 zeroed through
// a pointer at byte 8, element 0 stored through the memref, and a piece read
// back out of the zeroed part.

llvm.func @memset_tail_then_read() -> f64 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %z8 = llvm.mlir.constant(0 : i8) : i8
  %n32 = llvm.mlir.constant(32 : i64) : i64
  %one = arith.constant 1.000000e+00 : f64
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %a = llvm.alloca %c1 x !llvm.array<5 x f64> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %g = llvm.getelementptr inbounds|nuw %a[8] : (!llvm.ptr) -> !llvm.ptr, i8
  "llvm.intr.memset"(%g, %z8, %n32) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
  %v = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf64>
  memref.store %one, %v[%c0] : memref<?xf64>
  %r = memref.load %v[%c3] : memref<?xf64>
  llvm.return %r : f64
}

// CHECK-LABEL: llvm.func @memset_tail_then_read
// CHECK-NOT:     memref.load
