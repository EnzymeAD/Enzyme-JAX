// RUN: enzymexlamlir-opt %s -tessera-propagate-properties -split-input-file | FileCheck %s

// tessera-propagate-properties turns declarations into facts: on each
// tessera.call argument it records what is known of the value passed there.
// A rule's condition reads these facts and nothing else.

// What a definition always makes true of an output becomes a fact wherever
// that output is passed on, and a preserves rule carries it through a call
// whose inputs all have it.
!mat = !llvm.struct<"Mat3", (array<9 x f64>)>
module {
  tessera.define private @lib.mass(!llvm.ptr {llvm.sret = !mat, tessera.guarantees = ["SPD"]}, i64) attributes {argModes = [unit], pure = true}
  tessera.define private @lib.stiffness(!llvm.ptr {llvm.sret = !mat}, i64) attributes {argModes = [unit], pure = true}
  tessera.define private @lib.add(!llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}, {dir = #tessera.dir<in>, type = !mat}, {dir = #tessera.dir<out>, type = !mat}], pure = true, tessera.preserves = [{inputs = [0, 1], property = "SPD"}]}
  tessera.define private @lib.use(!llvm.ptr) -> f64 attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}], pure = true}

  // CHECK-LABEL: llvm.func @facts
  llvm.func @facts(%n: i64) -> f64 {
    // CHECK: %[[A:.*]] = tessera.call @lib.mass
    // CHECK: %[[B:.*]] = tessera.call @lib.mass
    // CHECK: %[[K:.*]] = tessera.call @lib.stiffness
    %a = tessera.call @lib.mass(%n) : (i64) -> !mat
    %b = tessera.call @lib.mass(%n) : (i64) -> !mat
    %k = tessera.call @lib.stiffness(%n) : (i64) -> !mat

    // Both terms are SPD, and so is the sum.
    // CHECK: %[[S:.*]] = tessera.call @lib.add(%[[A]], %[[B]]) <arg_attrs = [{tessera.property = ["SPD"]}, {tessera.property = ["SPD"]}]>
    // CHECK: tessera.call @lib.use(%[[S]]) <arg_attrs = [{tessera.property = ["SPD"]}]>
    %s = tessera.call @lib.add(%a, %b) : (!mat, !mat) -> !mat
    %u1 = tessera.call @lib.use(%s) : (!mat) -> f64

    // One term is not known to be SPD, so the sum is not either.
    // CHECK: %[[T:.*]] = tessera.call @lib.add(%[[A]], %[[K]]) <arg_attrs = [{tessera.property = ["SPD"]}, {}]>
    // CHECK: tessera.call @lib.use(%[[T]]) :
    %t = tessera.call @lib.add(%a, %k) : (!mat, !mat) -> !mat
    %u2 = tessera.call @lib.use(%t) : (!mat) -> f64

    %r = llvm.fadd %u1, %u2 : f64
    llvm.return %r : f64
  }
}

// -----

// What a definition assumes of a parameter is a fact inside it: of the
// parameter itself, and of the matrix loaded from it.
!mat = !llvm.struct<"Mat3", (array<9 x f64>)>
module {
  tessera.define private @lib.use(!llvm.ptr) -> f64 attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}], pure = true}
  tessera.define private @petsc.solve(!llvm.ptr) -> i32 attributes {argModes = [unit], pure = false}

  // CHECK-LABEL: llvm.func @assumed
  // CHECK: tessera.call @lib.use(%{{.*}}) <arg_attrs = [{tessera.property = ["SPD"]}]>
  llvm.func @assumed(%M: !llvm.ptr {tessera.assumes = ["SPD"]}) -> f64 {
    %m = llvm.load %M : !llvm.ptr -> !mat
    %r = tessera.call @lib.use(%m) : (!mat) -> f64
    llvm.return %r : f64
  }

  // A handle is passed as it is, so the fact is of the handle.
  // CHECK-LABEL: llvm.func @assumed_handle
  // CHECK: tessera.call @petsc.solve(%arg0) <arg_attrs = [{tessera.property = ["SPD"]}]>
  llvm.func @assumed_handle(%A: !llvm.ptr {tessera.assumes = ["SPD"]}) -> i32 {
    %r = tessera.call @petsc.solve(%A) : (!llvm.ptr) -> i32
    llvm.return %r : i32
  }
}

// -----

// A guarantee on a val=inout argument is of the value the function leaves
// there, which comes back as the call's leading result. The value passed in
// gains nothing.
!mat = !llvm.struct<"Mat3", (array<9 x f64>)>
module {
  tessera.define private @lib.make_spd(!llvm.ptr {tessera.guarantees = ["SPD"]}) attributes {argModes = [{dir = #tessera.dir<inout>, type = !mat}], pure = true}
  tessera.define private @lib.use(!llvm.ptr) -> f64 attributes {argModes = [{dir = #tessera.dir<in>, type = !mat}], pure = true}

  // CHECK-LABEL: llvm.func @guaranteed_inout
  llvm.func @guaranteed_inout(%m: !mat) -> f64 {
    // CHECK: %[[S:.*]] = tessera.call @lib.make_spd(%arg0) :
    // CHECK: tessera.call @lib.use(%arg0) :
    // CHECK: tessera.call @lib.use(%[[S]]) <arg_attrs = [{tessera.property = ["SPD"]}]>
    %s = tessera.call @lib.make_spd(%m) : (!mat) -> !mat
    %before = tessera.call @lib.use(%m) : (!mat) -> f64
    %after = tessera.call @lib.use(%s) : (!mat) -> f64
    %r = llvm.fadd %before, %after : f64
    llvm.return %r : f64
  }
}
