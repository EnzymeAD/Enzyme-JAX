// RUN: enzymexlamlir-opt %s -tessera-lower-guards -split-input-file -verify-diagnostics | FileCheck %s

// A condition can call a tessera op and compare what it returns. The call is
// a query about values the rule matched -- here whether an MPFR exponent is
// exactly 0.5 -- and becomes a tessera.call ahead of the branch, lowered with
// every other call by tessera-to-llvm.

!mpfr = !llvm.struct<(i64, i32, i64, ptr)>

module {
  tessera.define private @mpfr.nan_p(!llvm.ptr) -> i32 attributes {argModes = [{dir = #tessera.dir<in>, type = !mpfr}], pure = true}
  tessera.define private @mpfr.signbit(!llvm.ptr) -> i32 attributes {argModes = [{dir = #tessera.dir<in>, type = !mpfr}], pure = true}
  tessera.define private @mpfr.cmp_d(!llvm.ptr, f64) -> i32 attributes {argModes = [{dir = #tessera.dir<in>, type = !mpfr}, unit], pure = false}
  tessera.define private @mpfr.pow(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> i32 attributes {argModes = [{dir = #tessera.dir<inout>, type = !mpfr}, {dir = #tessera.dir<in>, type = !mpfr}, {dir = #tessera.dir<in>, type = !mpfr}, unit], pure = false}
  tessera.define private @mpfr.sqrt(!llvm.ptr, !llvm.ptr, i32) -> i32 attributes {argModes = [{dir = #tessera.dir<inout>, type = !mpfr}, {dir = #tessera.dir<in>, type = !mpfr}, unit], pure = false}

  // The literal passed to the call takes the type of the parameter it is
  // passed to, and the one compared with the result takes the result's type.
  // CHECK-LABEL: llvm.func @call_compare
  llvm.func @call_compare(%rop: !mpfr, %x: !mpfr, %y: !mpfr, %rnd: i32) -> i32 {
    // CHECK: %[[H:.*]] = llvm.mlir.constant(5.000000e-01 : f64) : f64
    // CHECK: %[[C:.*]] = tessera.call @mpfr.cmp_d(%arg2, %[[H]]) : ({{.*}}, f64) -> i32
    // CHECK: %[[Z:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[P:.*]] = llvm.icmp "eq" %[[C]], %[[Z]] : i32
    // CHECK: llvm.cond_br %[[P]], ^[[THEN:.*]], ^[[ELSE:.*]]
    // CHECK: ^[[THEN]]:
    // CHECK: tessera.call @mpfr.sqrt
    // CHECK: ^[[ELSE]]:
    // CHECK: tessera.call @mpfr.pow
    %0:2 = tessera.guard "mpfr.cmp_d(y, 0.5) == 0" args(%rop, %x, %y, %rnd) {argNames = ["rop", "x", "y", "rnd"]} : (!mpfr, !mpfr, !mpfr, i32) -> (!mpfr, i32) {
      %1:2 = tessera.call @mpfr.sqrt(%rop, %x, %rnd) : (!mpfr, !mpfr, i32) -> (!mpfr, i32)
      tessera.yield %1#0, %1#1 : !mpfr, i32
    } else {
      %2:2 = tessera.call @mpfr.pow(%rop, %x, %y, %rnd) : (!mpfr, !mpfr, !mpfr, i32) -> (!mpfr, i32)
      tessera.yield %2#0, %2#1 : !mpfr, i32
    }
    llvm.return %0#1 : i32
  }

  // A '&&' whose right side makes a call short-circuits, as in C: mpfr_cmp_d,
  // which raises MPFR's erange flag on a NaN, is only given y once y is known
  // not to be NaN, and the sign of x is only asked once y is known to be 0.5.
  // CHECK-LABEL: llvm.func @short_circuit
  llvm.func @short_circuit(%rop: !mpfr, %x: !mpfr, %y: !mpfr, %rnd: i32) -> i32 {
    // CHECK: %[[N:.*]] = tessera.call @mpfr.nan_p(%arg2)
    // CHECK: %[[NOTNAN:.*]] = llvm.icmp "eq" %[[N]], %{{.*}} : i32
    // CHECK: %[[F1:.*]] = llvm.mlir.constant(false) : i1
    // CHECK: llvm.cond_br %[[NOTNAN]], ^[[CMP:.*]], ^[[J1:.*]](%[[F1]] : i1)
    // CHECK: ^[[CMP]]:
    // CHECK: %[[C:.*]] = tessera.call @mpfr.cmp_d(%arg2
    // CHECK: %[[HALF:.*]] = llvm.icmp "eq" %[[C]], %{{.*}} : i32
    // CHECK: llvm.br ^[[J1]](%[[HALF]] : i1)
    // CHECK: ^[[J1]](%[[A:.*]]: i1):
    // CHECK: %[[F2:.*]] = llvm.mlir.constant(false) : i1
    // CHECK: llvm.cond_br %[[A]], ^[[SGN:.*]], ^[[J2:.*]](%[[F2]] : i1)
    // CHECK: ^[[SGN]]:
    // CHECK: %[[S:.*]] = tessera.call @mpfr.signbit(%arg1)
    // CHECK: %[[POS:.*]] = llvm.icmp "eq" %[[S]], %{{.*}} : i32
    // CHECK: llvm.br ^[[J2]](%[[POS]] : i1)
    // CHECK: ^[[J2]](%[[B:.*]]: i1):
    // CHECK: llvm.cond_br %[[B]], ^[[THEN:.*]], ^[[ELSE:.*]]
    // CHECK: ^[[THEN]]:
    // CHECK: tessera.call @mpfr.sqrt
    // CHECK: ^[[ELSE]]:
    // CHECK: tessera.call @mpfr.pow
    %0:2 = tessera.guard "mpfr.nan_p(y) == 0 && mpfr.cmp_d(y, 0.5) == 0 && mpfr.signbit(x) == 0" args(%rop, %x, %y, %rnd) {argNames = ["rop", "x", "y", "rnd"]} : (!mpfr, !mpfr, !mpfr, i32) -> (!mpfr, i32) {
      %1:2 = tessera.call @mpfr.sqrt(%rop, %x, %rnd) : (!mpfr, !mpfr, i32) -> (!mpfr, i32)
      tessera.yield %1#0, %1#1 : !mpfr, i32
    } else {
      %2:2 = tessera.call @mpfr.pow(%rop, %x, %y, %rnd) : (!mpfr, !mpfr, !mpfr, i32) -> (!mpfr, i32)
      tessera.yield %2#0, %2#1 : !mpfr, i32
    }
    llvm.return %0#1 : i32
  }

  // For '||' the left side deciding means it is true.
  // CHECK-LABEL: llvm.func @short_circuit_or
  llvm.func @short_circuit_or(%rop: !mpfr, %x: !mpfr, %y: !mpfr, %rnd: i32) -> i32 {
    // CHECK: %[[L:.*]] = llvm.icmp "sgt" %arg3, %{{.*}} : i32
    // CHECK: %[[T:.*]] = llvm.mlir.constant(true) : i1
    // CHECK: llvm.cond_br %[[L]], ^[[J:.*]](%[[T]] : i1), ^[[R:.*]]
    // CHECK: ^[[R]]:
    // CHECK: tessera.call @mpfr.nan_p(%arg2)
    // CHECK: llvm.br ^[[J]](
    // CHECK: ^[[J]](%[[V:.*]]: i1):
    // CHECK: llvm.cond_br %[[V]]
    %0:2 = tessera.guard "rnd > 3 || mpfr.nan_p(y) != 0" args(%rop, %x, %y, %rnd) {argNames = ["rop", "x", "y", "rnd"]} : (!mpfr, !mpfr, !mpfr, i32) -> (!mpfr, i32) {
      %1:2 = tessera.call @mpfr.sqrt(%rop, %x, %rnd) : (!mpfr, !mpfr, i32) -> (!mpfr, i32)
      tessera.yield %1#0, %1#1 : !mpfr, i32
    } else {
      %2:2 = tessera.call @mpfr.pow(%rop, %x, %y, %rnd) : (!mpfr, !mpfr, !mpfr, i32) -> (!mpfr, i32)
      tessera.yield %2#0, %2#1 : !mpfr, i32
    }
    llvm.return %0#1 : i32
  }

  // CHECK-NOT: tessera.guard
}

// -----

// A call the condition cannot make keeps the original computation, with a
// warning saying why, and leaves no part of the check behind.

!mpfr = !llvm.struct<(i64, i32, i64, ptr)>

module {
  tessera.define private @mpfr.nan_p(!llvm.ptr) -> i32 attributes {argModes = [{dir = #tessera.dir<in>, type = !mpfr}], pure = true}
  tessera.define private @mpfr.cmp_d(!llvm.ptr, f64) -> i32 attributes {argModes = [{dir = #tessera.dir<in>, type = !mpfr}, unit], pure = false}
  tessera.define private @mpfr.pow(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> i32 attributes {argModes = [{dir = #tessera.dir<inout>, type = !mpfr}, {dir = #tessera.dir<in>, type = !mpfr}, {dir = #tessera.dir<in>, type = !mpfr}, unit], pure = false}
  tessera.define private @mpfr.sqrt(!llvm.ptr, !llvm.ptr, i32) -> i32 attributes {argModes = [{dir = #tessera.dir<inout>, type = !mpfr}, {dir = #tessera.dir<in>, type = !mpfr}, unit], pure = false}

  // CHECK-LABEL: llvm.func @not_annotated
  llvm.func @not_annotated(%rop: !mpfr, %x: !mpfr, %y: !mpfr, %rnd: i32) -> i32 {
    // CHECK-NOT: llvm.cond_br
    // CHECK: tessera.call @mpfr.pow
    // CHECK-NEXT: llvm.return
    // expected-warning @+1 {{optimization rule not applied here, the original call is kept: condition calls 'mpfr.integer_p', which has no tessera.define; is it annotated tessera_op?}}
    %0:2 = tessera.guard "mpfr.integer_p(y) == 0" args(%rop, %x, %y, %rnd) {argNames = ["rop", "x", "y", "rnd"]} : (!mpfr, !mpfr, !mpfr, i32) -> (!mpfr, i32) {
      %1:2 = tessera.call @mpfr.sqrt(%rop, %x, %rnd) : (!mpfr, !mpfr, i32) -> (!mpfr, i32)
      tessera.yield %1#0, %1#1 : !mpfr, i32
    } else {
      %2:2 = tessera.call @mpfr.pow(%rop, %x, %y, %rnd) : (!mpfr, !mpfr, !mpfr, i32) -> (!mpfr, i32)
      tessera.yield %2#0, %2#1 : !mpfr, i32
    }
    llvm.return %0#1 : i32
  }

  // The call runs whichever way the guard goes, so it must not write.
  // CHECK-LABEL: llvm.func @writes
  llvm.func @writes(%rop: !mpfr, %x: !mpfr, %y: !mpfr, %rnd: i32) -> i32 {
    // CHECK-NOT: llvm.cond_br
    // CHECK: tessera.call @mpfr.pow
    // CHECK-NEXT: llvm.return
    // expected-warning @+1 {{optimization rule not applied here, the original call is kept: condition calls 'mpfr.sqrt', which writes one of its arguments}}
    %0:2 = tessera.guard "mpfr.sqrt(rop, x, rnd) == 0" args(%rop, %x, %y, %rnd) {argNames = ["rop", "x", "y", "rnd"]} : (!mpfr, !mpfr, !mpfr, i32) -> (!mpfr, i32) {
      %1:2 = tessera.call @mpfr.sqrt(%rop, %x, %rnd) : (!mpfr, !mpfr, i32) -> (!mpfr, i32)
      tessera.yield %1#0, %1#1 : !mpfr, i32
    } else {
      %2:2 = tessera.call @mpfr.pow(%rop, %x, %y, %rnd) : (!mpfr, !mpfr, !mpfr, i32) -> (!mpfr, i32)
      tessera.yield %2#0, %2#1 : !mpfr, i32
    }
    llvm.return %0#1 : i32
  }

  // CHECK-LABEL: llvm.func @arity
  llvm.func @arity(%rop: !mpfr, %x: !mpfr, %y: !mpfr, %rnd: i32) -> i32 {
    // CHECK-NOT: llvm.cond_br
    // CHECK: tessera.call @mpfr.pow
    // CHECK-NEXT: llvm.return
    // expected-warning @+1 {{optimization rule not applied here, the original call is kept: 'mpfr.cmp_d' takes 2 argument(s), but the condition passes 1}}
    %0:2 = tessera.guard "mpfr.cmp_d(y) == 0" args(%rop, %x, %y, %rnd) {argNames = ["rop", "x", "y", "rnd"]} : (!mpfr, !mpfr, !mpfr, i32) -> (!mpfr, i32) {
      %1:2 = tessera.call @mpfr.sqrt(%rop, %x, %rnd) : (!mpfr, !mpfr, i32) -> (!mpfr, i32)
      tessera.yield %1#0, %1#1 : !mpfr, i32
    } else {
      %2:2 = tessera.call @mpfr.pow(%rop, %x, %y, %rnd) : (!mpfr, !mpfr, !mpfr, i32) -> (!mpfr, i32)
      tessera.yield %2#0, %2#1 : !mpfr, i32
    }
    llvm.return %0#1 : i32
  }

  // Giving up on the right side of a short-circuit, after its blocks and the
  // call on the left have been built, still leaves nothing behind.
  // CHECK-LABEL: llvm.func @gives_up_part_way
  llvm.func @gives_up_part_way(%rop: !mpfr, %x: !mpfr, %y: !mpfr, %rnd: i32) -> i32 {
    // CHECK-NOT: tessera.call @mpfr.nan_p
    // CHECK-NOT: llvm.cond_br
    // CHECK: tessera.call @mpfr.pow
    // CHECK-NEXT: llvm.return
    // expected-warning @+1 {{optimization rule not applied here, the original call is kept: argument 1 of 'mpfr.cmp_d' in a condition is 'i32', which it does not take}}
    %0:2 = tessera.guard "mpfr.nan_p(y) == 0 && mpfr.cmp_d(y, rnd) == 0" args(%rop, %x, %y, %rnd) {argNames = ["rop", "x", "y", "rnd"]} : (!mpfr, !mpfr, !mpfr, i32) -> (!mpfr, i32) {
      %1:2 = tessera.call @mpfr.sqrt(%rop, %x, %rnd) : (!mpfr, !mpfr, i32) -> (!mpfr, i32)
      tessera.yield %1#0, %1#1 : !mpfr, i32
    } else {
      %2:2 = tessera.call @mpfr.pow(%rop, %x, %y, %rnd) : (!mpfr, !mpfr, !mpfr, i32) -> (!mpfr, i32)
      tessera.yield %2#0, %2#1 : !mpfr, i32
    }
    llvm.return %0#1 : i32
  }
}
