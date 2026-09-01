// RUN: enzymexlamlir-opt %s --canonicalize-scf-for --canonicalize | FileCheck %s

// mfem walks a pointer as `TC *c = Cdata;` advanced with `c++` in an inner
// loop, so the enclosing loop carries it while the inner loop's own copy is
// dead. The inner result is the yielded value of the last iteration.
func.func @walk(%init: !llvm.ptr, %dead: !llvm.ptr, %bw: i64, %aw: i64, %v: f64) {
  %idx0 = arith.constant 0 : index
  %c1 = arith.constant 1 : i64
  %cm1 = arith.constant -1 : i64
  %c8 = arith.constant 8 : i64
  %outer = scf.for %i = %c1 to %bw step %c1 iter_args(%c = %init) -> (!llvm.ptr) : i64 {
    %inner = scf.for %j = %c1 to %aw step %c1 iter_args(%lag = %dead) -> (!llvm.ptr) : i64 {
      %jm = arith.addi %j, %cm1 : i64
      %off = arith.muli %jm, %c8 : i64
      %at = llvm.getelementptr inbounds|nuw %c[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %m = "enzymexla.pointer2memref"(%at) : (!llvm.ptr) -> memref<?xf64>
      %old = memref.load %m[%idx0] : memref<?xf64>
      %new = arith.addf %old, %v : f64
      memref.store %new, %m[%idx0] : memref<?xf64>
      %next = llvm.getelementptr inbounds|nuw %at[8] : (!llvm.ptr) -> !llvm.ptr, i8
      scf.yield %next : !llvm.ptr
    }
    scf.yield %inner : !llvm.ptr
  }
  return
}

// With bounds the folder can settle, the guard goes too and the enclosing
// loop stops carrying the pointer as well.
func.func @walk_const(%init: !llvm.ptr, %dead: !llvm.ptr, %v: f64) {
  %idx0 = arith.constant 0 : index
  %c1 = arith.constant 1 : i64
  %cm1 = arith.constant -1 : i64
  %c8 = arith.constant 8 : i64
  %c5 = arith.constant 5 : i64
  %c9 = arith.constant 9 : i64
  %outer = scf.for %i = %c1 to %c9 step %c1 iter_args(%c = %init) -> (!llvm.ptr) : i64 {
    %inner = scf.for %j = %c1 to %c5 step %c1 iter_args(%lag = %dead) -> (!llvm.ptr) : i64 {
      %jm = arith.addi %j, %cm1 : i64
      %off = arith.muli %jm, %c8 : i64
      %at = llvm.getelementptr inbounds|nuw %c[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %m = "enzymexla.pointer2memref"(%at) : (!llvm.ptr) -> memref<?xf64>
      %old = memref.load %m[%idx0] : memref<?xf64>
      %new = arith.addf %old, %v : f64
      memref.store %new, %m[%idx0] : memref<?xf64>
      %next = llvm.getelementptr inbounds|nuw %at[8] : (!llvm.ptr) -> !llvm.ptr, i8
      scf.yield %next : !llvm.ptr
    }
    scf.yield %inner : !llvm.ptr
  }
  return
}

// An iter arg the body reads is accumulating something: left alone.
func.func @live_iter_arg(%n: i64, %x: f64) -> f64 {
  %c1 = arith.constant 1 : i64
  %cst = arith.constant 0.0 : f64
  %r = scf.for %i = %c1 to %n step %c1 iter_args(%acc = %cst) -> (f64) : i64 {
    %s = arith.addf %acc, %x : f64
    scf.yield %s : f64
  }
  return %r : f64
}

// CHECK-LABEL: func.func @walk(
// The inner loop no longer carries a pointer; its result is the last value.
// CHECK: arith.select
// CHECK: scf.for
// CHECK-NOT: iter_args

// CHECK-LABEL: func.func @walk_const(
// CHECK-NOT: iter_args
// CHECK-NOT: arith.select

// CHECK-LABEL: func.func @live_iter_arg(
// CHECK: iter_args
