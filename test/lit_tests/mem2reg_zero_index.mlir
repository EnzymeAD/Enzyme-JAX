// RUN: enzymexlamlir-opt --polygeist-mem2reg --split-input-file %s | FileCheck %s

// A store at index 0 and a store at index 2 are two different places, so the
// one at 0 is what a load at 0 reads. Naming the start of an allocation is
// naming no offset into it at all, so the access at [0] carries no indices --
// and an offset that goes nowhere still says where it is.

llvm.func @load_index_zero() -> f64 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %one = arith.constant 1.000000e+00 : f64
  %two = arith.constant 2.000000e+00 : f64
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %a = llvm.alloca %c1 x !llvm.array<5 x f64> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %v = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf64>
  memref.store %one, %v[%c0] : memref<?xf64>
  memref.store %two, %v[%c2] : memref<?xf64>
  %r = memref.load %v[%c0] : memref<?xf64>
  llvm.return %r : f64
}

// CHECK-LABEL: llvm.func @load_index_zero
// CHECK-NOT:     memref.load
// CHECK:         llvm.return %cst

// -----

// The same the other way round, which already worked: the last store before
// the load names the index the load names.

llvm.func @load_index_nonzero() -> f64 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %one = arith.constant 1.000000e+00 : f64
  %two = arith.constant 2.000000e+00 : f64
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %a = llvm.alloca %c1 x !llvm.array<5 x f64> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %v = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf64>
  memref.store %one, %v[%c0] : memref<?xf64>
  memref.store %two, %v[%c2] : memref<?xf64>
  %r = memref.load %v[%c2] : memref<?xf64>
  llvm.return %r : f64
}

// CHECK-LABEL: llvm.func @load_index_nonzero
// CHECK-NOT:     memref.load
// CHECK:         llvm.return %cst

// -----

// A static view is no different: it is the index, not the shape, that decides.

llvm.func @load_index_zero_static() -> f64 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %one = arith.constant 1.000000e+00 : f64
  %two = arith.constant 2.000000e+00 : f64
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %a = llvm.alloca %c1 x !llvm.array<5 x f64> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %v = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<5xf64>
  memref.store %one, %v[%c0] : memref<5xf64>
  memref.store %two, %v[%c2] : memref<5xf64>
  %r = memref.load %v[%c0] : memref<5xf64>
  llvm.return %r : f64
}

// CHECK-LABEL: llvm.func @load_index_zero_static
// CHECK-NOT:     memref.load
// CHECK:         llvm.return %cst
