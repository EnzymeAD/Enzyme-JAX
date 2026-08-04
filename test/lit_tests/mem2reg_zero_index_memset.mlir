// RUN: enzymexlamlir-opt --polygeist-mem2reg --split-input-file %s | FileCheck %s

// A fill reaching over the whole of a slot says what is in it: zero bytes are
// a zero of whatever the slot holds, however far past it they run.

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
// CHECK:         llvm.mlir.zero

// -----

// The slot need not be all the fill wrote: a piece of what a whole-allocation
// fill covered is zero like the rest of it.

llvm.func @memset_all_read_piece() -> f64 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %z8 = llvm.mlir.constant(0 : i8) : i8
  %n40 = llvm.mlir.constant(40 : i64) : i64
  %a = llvm.alloca %c1 x !llvm.array<5 x f64> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  "llvm.intr.memset"(%a, %z8, %n40) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
  %r = llvm.load %a : !llvm.ptr -> f64
  llvm.return %r : f64
}

// CHECK-LABEL: llvm.func @memset_all_read_piece
// CHECK-NOT:     llvm.load
// CHECK:         llvm.mlir.zero

// -----

// A fill of the whole allocation reaches an element counted in dimensions the
// same as one counted in bytes does.

llvm.func @memset_all_read_dim() -> f64 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %z8 = llvm.mlir.constant(0 : i8) : i8
  %n40 = llvm.mlir.constant(40 : i64) : i64
  %c3 = arith.constant 3 : index
  %a = llvm.alloca %c1 x !llvm.array<5 x f64> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  "llvm.intr.memset"(%a, %z8, %n40) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
  %v = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf64>
  %r = memref.load %v[%c3] : memref<?xf64>
  llvm.return %r : f64
}

// CHECK-LABEL: llvm.func @memset_all_read_dim
// CHECK-NOT:     memref.load
// CHECK:         llvm.mlir.zero

// -----

// Reaching it is also overwriting it: a fill after a store is what is read
// back, not what the store put there.

llvm.func @store_then_memset_all() -> f64 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %z8 = llvm.mlir.constant(0 : i8) : i8
  %n40 = llvm.mlir.constant(40 : i64) : i64
  %one = arith.constant 1.000000e+00 : f64
  %c3 = arith.constant 3 : index
  %a = llvm.alloca %c1 x !llvm.array<5 x f64> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %v = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf64>
  memref.store %one, %v[%c3] : memref<?xf64>
  "llvm.intr.memset"(%a, %z8, %n40) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
  %r = memref.load %v[%c3] : memref<?xf64>
  llvm.return %r : f64
}

// CHECK-LABEL: llvm.func @store_then_memset_all
// CHECK-NOT:     memref.load
// CHECK:         %[[Z:.+]] = llvm.mlir.zero : f64
// CHECK:         llvm.return %[[Z]] : f64

// -----

// A fill that stops short of the slot says nothing about the rest of it, so
// what a store put there is still what is read back.

llvm.func @memset_part_keeps_rest(%v: !llvm.array<5 x f64>) -> f64 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %z8 = llvm.mlir.constant(0 : i8) : i8
  %n8 = llvm.mlir.constant(8 : i64) : i64
  %c3 = arith.constant 3 : index
  %a = llvm.alloca %c1 x !llvm.array<5 x f64> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  llvm.store %v, %a : !llvm.array<5 x f64>, !llvm.ptr
  "llvm.intr.memset"(%a, %z8, %n8) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
  %m = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf64>
  %r = memref.load %m[%c3] : memref<?xf64>
  llvm.return %r : f64
}

// CHECK-LABEL: llvm.func @memset_part_keeps_rest
// CHECK:         "llvm.intr.memset"
// CHECK:         memref.load

// -----

// Nor does a fill of anything but zero, whatever it covers.

llvm.func @memset_nonzero_keeps_load() -> f64 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %b = llvm.mlir.constant(7 : i8) : i8
  %n40 = llvm.mlir.constant(40 : i64) : i64
  %c3 = arith.constant 3 : index
  %a = llvm.alloca %c1 x !llvm.array<5 x f64> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  "llvm.intr.memset"(%a, %b, %n40) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
  %m = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf64>
  %r = memref.load %m[%c3] : memref<?xf64>
  llvm.return %r : f64
}

// CHECK-LABEL: llvm.func @memset_nonzero_keeps_load
// CHECK:         memref.load
