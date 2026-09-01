// RUN: enzymexlamlir-opt %s --canonicalize-scf-for | FileCheck %s

// A pointer advanced by a constant gep each iteration is an induction
// variable: on iteration n it is the init advanced by n times that offset.
func.func @ptr_induction(%p: !llvm.ptr, %n: index, %x: f64) -> !llvm.ptr {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%q = %p) -> (!llvm.ptr) {
    llvm.store %x, %q : f64, !llvm.ptr
    %next = llvm.getelementptr inbounds|nuw %q[8] : (!llvm.ptr) -> !llvm.ptr, i8
    scf.yield %next : !llvm.ptr
  }
  return %r : !llvm.ptr
}

// A step of more than one divides into the count.
func.func @strided(%p: !llvm.ptr, %n: index, %x: f64) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %i = %c0 to %n step %c4 iter_args(%q = %p) -> (!llvm.ptr) {
    llvm.store %x, %q : f64, !llvm.ptr
    %next = llvm.getelementptr %q[8] : (!llvm.ptr) -> !llvm.ptr, i8
    scf.yield %next : !llvm.ptr
  }
  return
}

// An advance the loop does not vary is a fixed distance, even as a value.
func.func @dynamic_advance(%p: !llvm.ptr, %n: index, %k: i64, %x: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%q = %p) -> (!llvm.ptr) {
    llvm.store %x, %q : f64, !llvm.ptr
    %next = llvm.getelementptr %q[%k] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    scf.yield %next : !llvm.ptr
  }
  return
}

// Advancing something other than the carried pointer is not an induction.
func.func @other_base(%p: !llvm.ptr, %o: !llvm.ptr, %n: index, %x: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%q = %p) -> (!llvm.ptr) {
    llvm.store %x, %q : f64, !llvm.ptr
    %next = llvm.getelementptr %o[8] : (!llvm.ptr) -> !llvm.ptr, i8
    scf.yield %next : !llvm.ptr
  }
  return
}

// CHECK-LABEL: func.func @ptr_induction(
// CHECK-NOT: iter_args
// CHECK: llvm.getelementptr inbounds|nuw %arg0[%{{.+}}]

// CHECK-LABEL: func.func @strided(
// CHECK-NOT: iter_args

// CHECK-LABEL: func.func @dynamic_advance(
// CHECK-NOT: iter_args

// CHECK-LABEL: func.func @other_base(
// CHECK: iter_args

// -----
// RUN: enzymexlamlir-opt %s --canonicalize-scf-for | FileCheck %s --check-prefix=CHAIN

// The advance can be a chain, and a step of it can be a value the loop does
// not vary.
func.func @chained_invariant(%p: !llvm.ptr, %n: i64, %k: i64, %x: f64) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%q = %p) -> (!llvm.ptr) : i64 {
    %mid = llvm.getelementptr inbounds|nuw %q[%k] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %x, %mid : f64, !llvm.ptr
    %next = llvm.getelementptr inbounds|nuw %mid[8] : (!llvm.ptr) -> !llvm.ptr, i8
    scf.yield %next : !llvm.ptr
  }
  return
}

// A stride the loop varies is not a fixed distance: left alone.
func.func @varying_stride(%p: !llvm.ptr, %n: i64, %x: f64) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%q = %p) -> (!llvm.ptr) : i64 {
    %mid = llvm.getelementptr inbounds|nuw %q[%i] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %x, %mid : f64, !llvm.ptr
    %next = llvm.getelementptr inbounds|nuw %mid[8] : (!llvm.ptr) -> !llvm.ptr, i8
    scf.yield %next : !llvm.ptr
  }
  return
}

// CHAIN-LABEL: func.func @chained_invariant(
// CHAIN-NOT: iter_args

// CHAIN-LABEL: func.func @varying_stride(
// CHAIN: iter_args
