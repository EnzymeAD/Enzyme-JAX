// RUN: enzymexlamlir-opt %s --canonicalize-scf-for --split-input-file | FileCheck %s

// The loop leaves either because a dependency's flag was clear (%ok false) or
// because it scanned them all (%i2b reached %n) with every flag set. Exiting
// only says the conjunction failed -- at least one conjunct is false, with no
// say in which -- so the forwarded %ok must not be folded to false. This is
// NCMesh::DofFinalizable after LICM: with its result pinned false, no DoF was
// ever finalizable and BuildParallelConformingInterpolation spun forever.
//
// (The condition arrives as select(%ok, %more, false) and reaches
// WhileLogicalNegation as andi %ok, %more; the duplicated increment is as the
// frontend spelled it, and keeps MoveWhileToFor from converting the loop.)

llvm.func @all_set(%deps: !llvm.ptr, %flags: !llvm.ptr, %n: i64) -> i1 {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %r:2 = scf.while (%i = %c0) : (i64) -> (i64, i1) {
    %p = llvm.getelementptr inbounds %deps[%i] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %d = llvm.load %p {alignment = 4 : i64} : !llvm.ptr -> i32
    %d64 = arith.extsi %d : i32 to i64
    %q = llvm.getelementptr inbounds %flags[%d64] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %f8 = llvm.load %q {alignment = 1 : i64} : !llvm.ptr -> i8
    %ok = arith.trunci %f8 : i8 to i1
    %i2 = arith.addi %i, %c1 : i64
    %i2b = arith.addi %i, %c1 : i64
    %more = arith.cmpi ne, %i2b, %n : i64
    %more32 = arith.extui %more : i1 to i32
    %cond32 = arith.select %ok, %more32, %c0_i32 : i32
    %cond = arith.trunci %cond32 : i32 to i1
    scf.condition(%cond) %i2, %ok : i64, i1
  } do {
  ^bb0(%i2: i64, %ok2: i1):
    scf.yield %i2 : i64
  }
  llvm.return %r#1 : i1
}

// CHECK-LABEL: llvm.func @all_set
// CHECK:         %[[R:.+]]:2 = scf.while
// CHECK:         llvm.return %[[R]]#1 : i1

// -----

// When the forwarded value is the root condition itself, the exit does decide
// it: the loop ran until the condition failed, so the result is false.

llvm.func @exhausted(%deps: !llvm.ptr, %n: i64) -> i1 {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %r:2 = scf.while (%i = %c0) : (i64) -> (i64, i1) {
    %p = llvm.getelementptr inbounds %deps[%i] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %d = llvm.load %p {alignment = 4 : i64} : !llvm.ptr -> i32
    %d64 = arith.extsi %d : i32 to i64
    %i2 = arith.addi %i, %d64 : i64
    %cont = arith.cmpi ne, %i2, %n : i64
    scf.condition(%cont) %i2, %cont : i64, i1
  } do {
  ^bb0(%i2: i64, %c: i1):
    scf.yield %i2 : i64
  }
  llvm.return %r#1 : i1
}

// CHECK-LABEL: llvm.func @exhausted
// CHECK:         %[[F:.+]] = arith.constant false
// CHECK:         llvm.return %[[F]] : i1
