// RUN: enzymexlamlir-opt %s -tessera-lower-guards -split-input-file -verify-diagnostics | FileCheck %s

// The body of an scf.for, scf.if or affine.for is one block, so a guard there
// cannot become a branch between blocks. It becomes an scf.if on the check,
// with the guard's regions as its own, and a short-circuit in the check
// becomes an scf.if yielding the result.

!mpfr = !llvm.struct<(i64, i32, i64, ptr)>

module {
  tessera.define private @mpfr.nan_p(!llvm.ptr) -> i32 attributes {argModes = [{dir = #tessera.dir<in>, type = !mpfr}], pure = true}
  tessera.define private @mpfr.signbit(!llvm.ptr) -> i32 attributes {argModes = [{dir = #tessera.dir<in>, type = !mpfr}], pure = true}
  tessera.define private @mpfr.cmp_d(!llvm.ptr, f64) -> i32 attributes {argModes = [{dir = #tessera.dir<in>, type = !mpfr}, unit], pure = false}
  tessera.define private @mpfr.pow(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> i32 attributes {argModes = [{dir = #tessera.dir<inout>, type = !mpfr}, {dir = #tessera.dir<in>, type = !mpfr}, {dir = #tessera.dir<in>, type = !mpfr}, unit], pure = false}
  tessera.define private @mpfr.sqrt(!llvm.ptr, !llvm.ptr, i32) -> i32 attributes {argModes = [{dir = #tessera.dir<inout>, type = !mpfr}, {dir = #tessera.dir<in>, type = !mpfr}, unit], pure = false}

  // CHECK-LABEL: llvm.func @in_scf_for
  llvm.func @in_scf_for(%rop: !mpfr, %x: !mpfr, %y: !mpfr, %rnd: i32, %out: !llvm.ptr, %len: i64) {
    %n = arith.index_cast %len : i64 to index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // CHECK: scf.for
    // CHECK-NOT: llvm.cond_br
    // CHECK: %[[P:.*]] = llvm.icmp "sgt" %arg3, %{{.*}} : i32
    // CHECK: %[[R:.*]]:2 = scf.if %[[P]] -> ({{.*}}, i32) {
    // CHECK: %[[S:.*]]:2 = tessera.call @mpfr.sqrt
    // CHECK: scf.yield %[[S]]#0, %[[S]]#1
    // CHECK: } else {
    // CHECK: %[[W:.*]]:2 = tessera.call @mpfr.pow
    // CHECK: scf.yield %[[W]]#0, %[[W]]#1
    // CHECK: }
    // CHECK: llvm.store %[[R]]#1, %arg4
    scf.for %i = %c0 to %n step %c1 {
      %0:2 = tessera.guard "rnd > 3" args(%rop, %x, %y, %rnd) {argNames = ["rop", "x", "y", "rnd"]} : (!mpfr, !mpfr, !mpfr, i32) -> (!mpfr, i32) {
        %1:2 = tessera.call @mpfr.sqrt(%rop, %x, %rnd) : (!mpfr, !mpfr, i32) -> (!mpfr, i32)
        tessera.yield %1#0, %1#1 : !mpfr, i32
      } else {
        %2:2 = tessera.call @mpfr.pow(%rop, %x, %y, %rnd) : (!mpfr, !mpfr, !mpfr, i32) -> (!mpfr, i32)
        tessera.yield %2#0, %2#1 : !mpfr, i32
      }
      llvm.store %0#1, %out : i32, !llvm.ptr
    }
    llvm.return
  }

  // The calls on the right of each '&&' are made only in the branch where
  // what comes before holds, as at function level.
  // CHECK-LABEL: llvm.func @short_circuit_in_affine_for
  llvm.func @short_circuit_in_affine_for(%rop: !mpfr, %x: !mpfr, %y: !mpfr, %rnd: i32, %out: !llvm.ptr, %len: i64) {
    %n = arith.index_cast %len : i64 to index
    // CHECK: affine.for
    // CHECK-NOT: llvm.cond_br
    // CHECK: %[[N:.*]] = tessera.call @mpfr.nan_p(%arg2)
    // CHECK: %[[NOTNAN:.*]] = llvm.icmp "eq" %[[N]], %{{.*}} : i32
    // CHECK: %[[A:.*]] = scf.if %[[NOTNAN]] -> (i1) {
    // CHECK: %[[C:.*]] = tessera.call @mpfr.cmp_d(%arg2
    // CHECK: %[[HALF:.*]] = llvm.icmp "eq" %[[C]], %{{.*}} : i32
    // CHECK: scf.yield %[[HALF]] : i1
    // CHECK: } else {
    // CHECK: %[[F1:.*]] = llvm.mlir.constant(false) : i1
    // CHECK: scf.yield %[[F1]] : i1
    // CHECK: }
    // CHECK: %[[B:.*]] = scf.if %[[A]] -> (i1) {
    // CHECK: tessera.call @mpfr.signbit(%arg1)
    // CHECK: } else {
    // CHECK: llvm.mlir.constant(false) : i1
    // CHECK: }
    // CHECK: scf.if %[[B]] -> ({{.*}}, i32) {
    // CHECK: tessera.call @mpfr.sqrt
    // CHECK: } else {
    // CHECK: tessera.call @mpfr.pow
    // CHECK: }
    affine.for %i = 0 to %n {
      %0:2 = tessera.guard "mpfr.nan_p(y) == 0 && mpfr.cmp_d(y, 0.5) == 0 && mpfr.signbit(x) == 0" args(%rop, %x, %y, %rnd) {argNames = ["rop", "x", "y", "rnd"]} : (!mpfr, !mpfr, !mpfr, i32) -> (!mpfr, i32) {
        %1:2 = tessera.call @mpfr.sqrt(%rop, %x, %rnd) : (!mpfr, !mpfr, i32) -> (!mpfr, i32)
        tessera.yield %1#0, %1#1 : !mpfr, i32
      } else {
        %2:2 = tessera.call @mpfr.pow(%rop, %x, %y, %rnd) : (!mpfr, !mpfr, !mpfr, i32) -> (!mpfr, i32)
        tessera.yield %2#0, %2#1 : !mpfr, i32
      }
      llvm.store %0#1, %out : i32, !llvm.ptr
    }
    llvm.return
  }

  // For '||' the left side deciding means it is true, so the scf.if yields
  // true on that side and evaluates the right side in its else branch.
  // CHECK-LABEL: llvm.func @short_circuit_or_in_scf_if
  llvm.func @short_circuit_or_in_scf_if(%rop: !mpfr, %x: !mpfr, %y: !mpfr, %rnd: i32, %out: !llvm.ptr, %c: i1) {
    // CHECK: scf.if %arg5 {
    // CHECK: %[[L:.*]] = llvm.icmp "sgt" %arg3, %{{.*}} : i32
    // CHECK: %[[V:.*]] = scf.if %[[L]] -> (i1) {
    // CHECK: %[[T:.*]] = llvm.mlir.constant(true) : i1
    // CHECK: scf.yield %[[T]] : i1
    // CHECK: } else {
    // CHECK: tessera.call @mpfr.nan_p(%arg2)
    // CHECK: }
    // CHECK: scf.if %[[V]] -> ({{.*}}, i32) {
    scf.if %c {
      %0:2 = tessera.guard "rnd > 3 || mpfr.nan_p(y) != 0" args(%rop, %x, %y, %rnd) {argNames = ["rop", "x", "y", "rnd"]} : (!mpfr, !mpfr, !mpfr, i32) -> (!mpfr, i32) {
        %1:2 = tessera.call @mpfr.sqrt(%rop, %x, %rnd) : (!mpfr, !mpfr, i32) -> (!mpfr, i32)
        tessera.yield %1#0, %1#1 : !mpfr, i32
      } else {
        %2:2 = tessera.call @mpfr.pow(%rop, %x, %y, %rnd) : (!mpfr, !mpfr, !mpfr, i32) -> (!mpfr, i32)
        tessera.yield %2#0, %2#1 : !mpfr, i32
      }
      llvm.store %0#1, %out : i32, !llvm.ptr
    }
    llvm.return
  }

  // CHECK-NOT: tessera.guard
}

// -----

// Giving up part way through a structured check, after the call on the left
// and the scf.if for the right have been built, leaves nothing behind.

!mpfr = !llvm.struct<(i64, i32, i64, ptr)>

module {
  tessera.define private @mpfr.nan_p(!llvm.ptr) -> i32 attributes {argModes = [{dir = #tessera.dir<in>, type = !mpfr}], pure = true}
  tessera.define private @mpfr.cmp_d(!llvm.ptr, f64) -> i32 attributes {argModes = [{dir = #tessera.dir<in>, type = !mpfr}, unit], pure = false}
  tessera.define private @mpfr.pow(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> i32 attributes {argModes = [{dir = #tessera.dir<inout>, type = !mpfr}, {dir = #tessera.dir<in>, type = !mpfr}, {dir = #tessera.dir<in>, type = !mpfr}, unit], pure = false}
  tessera.define private @mpfr.sqrt(!llvm.ptr, !llvm.ptr, i32) -> i32 attributes {argModes = [{dir = #tessera.dir<inout>, type = !mpfr}, {dir = #tessera.dir<in>, type = !mpfr}, unit], pure = false}

  // CHECK-LABEL: llvm.func @gives_up_part_way
  llvm.func @gives_up_part_way(%rop: !mpfr, %x: !mpfr, %y: !mpfr, %rnd: i32, %out: !llvm.ptr, %len: i64) {
    %n = arith.index_cast %len : i64 to index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // CHECK: scf.for
    // CHECK-NOT: tessera.call @mpfr.nan_p
    // CHECK-NOT: scf.if
    // CHECK: tessera.call @mpfr.pow
    // CHECK-NEXT: llvm.store
    scf.for %i = %c0 to %n step %c1 {
      // expected-warning @+1 {{optimization rule not applied here, the original call is kept: argument 1 of 'mpfr.cmp_d' in a condition is 'i32', which it does not take}}
      %0:2 = tessera.guard "mpfr.nan_p(y) == 0 && mpfr.cmp_d(y, rnd) == 0" args(%rop, %x, %y, %rnd) {argNames = ["rop", "x", "y", "rnd"]} : (!mpfr, !mpfr, !mpfr, i32) -> (!mpfr, i32) {
        %1:2 = tessera.call @mpfr.sqrt(%rop, %x, %rnd) : (!mpfr, !mpfr, i32) -> (!mpfr, i32)
        tessera.yield %1#0, %1#1 : !mpfr, i32
      } else {
        %2:2 = tessera.call @mpfr.pow(%rop, %x, %y, %rnd) : (!mpfr, !mpfr, !mpfr, i32) -> (!mpfr, i32)
        tessera.yield %2#0, %2#1 : !mpfr, i32
      }
      llvm.store %0#1, %out : i32, !llvm.ptr
    }
    llvm.return
  }
}
