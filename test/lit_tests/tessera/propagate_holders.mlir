// RUN: enzymexlamlir-opt %s -tessera-propagate-properties -split-input-file | FileCheck %s

// Holders. A call may store a pointer to an object it is given, as
// KSPSetOperators(ksp, A, P) stores A inside ksp. A later call given ksp, but
// not A, can then change A. So a call given a holder -- a pointer an earlier
// call, given the object or another holder, may have stored it in -- counts
// as reaching the object.
//
// Any call, annotated or not, may store any pointer it is given, unless the
// parameter is marked `tessera.noescape` (C's noescape) or `llvm.nocapture`.
// Leaving the mark out only loses facts. A call that leaves a holder
// unchanged (`tessera.unchanged`) changes nothing it holds either.
//
// Each function below states SPD on the matrix in `%slot`, then reaches the
// rule's call some way. Where the property is lost, the CHECK-NOT shows it.

module {
  tessera.define private @petsc.ksp_set_operators(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [unit, unit, unit], pure = false}
  llvm.mlir.global external @comm_world() {addr_space = 0 : i32} : !llvm.ptr
  llvm.func @mat_create(!llvm.ptr) -> i32
  llvm.func @fill_coo(!llvm.ptr {tessera.establishes = ["SPD"]}, !llvm.ptr) -> i32
  // Unannotated: may store anything it is given.
  llvm.func @register(!llvm.ptr, !llvm.ptr) -> i32
  llvm.func @touch(!llvm.ptr) -> i32
  llvm.func @wrap(!llvm.ptr) -> !llvm.ptr
  // Leaves the object alone, but may store a pointer to it.
  llvm.func @watch(!llvm.ptr, !llvm.ptr {tessera.unchanged}) -> i32
  // Leaves it alone, and holds on to no pointer to it.
  llvm.func @look(!llvm.ptr, !llvm.ptr {tessera.unchanged, tessera.noescape}) -> i32
  llvm.func @inspect(!llvm.ptr {tessera.unchanged}) -> i32
  llvm.func @register_nocapture(!llvm.ptr, !llvm.ptr {llvm.nocapture}) -> i32

  // Stored before the matrix is SPD, and changed through the holder after.
  // CHECK-LABEL: llvm.func @stored_before_fact
  // CHECK: tessera.call @petsc.ksp_set_operators
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @stored_before_fact(%ksp: !llvm.ptr, %x: !llvm.ptr, %reg: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @register(%reg, %a0) : (!llvm.ptr, !llvm.ptr) -> i32
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e2 = llvm.call @fill_coo(%a1, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %e3 = llvm.call @touch(%reg) : (!llvm.ptr) -> i32
    %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e4 = tessera.call @petsc.ksp_set_operators(%ksp, %a2, %a2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e4 : i32
  }

  // Stored after, by a call that leaves the matrix unchanged: unchanged says
  // the call itself leaves it alone, not that nothing can change it later.
  // CHECK-LABEL: llvm.func @stored_by_unchanging_call
  // CHECK: tessera.call @petsc.ksp_set_operators
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @stored_by_unchanging_call(%ksp: !llvm.ptr, %x: !llvm.ptr, %reg: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%a0, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e2 = llvm.call @watch(%reg, %a1) : (!llvm.ptr, !llvm.ptr) -> i32
    %e3 = llvm.call @touch(%reg) : (!llvm.ptr) -> i32
    %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e4 = tessera.call @petsc.ksp_set_operators(%ksp, %a2, %a2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e4 : i32
  }

  // A call given the matrix noescape stores no pointer to it, so %reg holds
  // nothing.
  // CHECK-LABEL: llvm.func @noescape
  // CHECK: tessera.call @petsc.ksp_set_operators({{.*}}) <arg_attrs = [{}, {tessera.property = ["SPD"]}, {tessera.property = ["SPD"]}]>
  llvm.func @noescape(%ksp: !llvm.ptr, %x: !llvm.ptr, %reg: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%a0, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e2 = llvm.call @look(%reg, %a1) : (!llvm.ptr, !llvm.ptr) -> i32
    %e3 = llvm.call @touch(%reg) : (!llvm.ptr) -> i32
    %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e4 = tessera.call @petsc.ksp_set_operators(%ksp, %a2, %a2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e4 : i32
  }

  // A call that leaves the holder unchanged leaves what it holds alone.
  // CHECK-LABEL: llvm.func @holder_unchanged
  // CHECK: tessera.call @petsc.ksp_set_operators({{.*}}) <arg_attrs = [{}, {tessera.property = ["SPD"]}, {tessera.property = ["SPD"]}]>
  llvm.func @holder_unchanged(%ksp: !llvm.ptr, %x: !llvm.ptr, %reg: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @register(%reg, %a0) : (!llvm.ptr, !llvm.ptr) -> i32
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e2 = llvm.call @fill_coo(%a1, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %e3 = llvm.call @inspect(%reg) : (!llvm.ptr) -> i32
    %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e4 = tessera.call @petsc.ksp_set_operators(%ksp, %a2, %a2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e4 : i32
  }

  // A holder of a holder counts too.
  // CHECK-LABEL: llvm.func @holder_of_holder
  // CHECK: tessera.call @petsc.ksp_set_operators
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @holder_of_holder(%ksp: !llvm.ptr, %x: !llvm.ptr, %reg: !llvm.ptr, %outer: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @register(%reg, %a0) : (!llvm.ptr, !llvm.ptr) -> i32
    %e2 = llvm.call @register(%outer, %reg) : (!llvm.ptr, !llvm.ptr) -> i32
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e3 = llvm.call @fill_coo(%a1, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %e4 = llvm.call @touch(%outer) : (!llvm.ptr) -> i32
    %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e5 = tessera.call @petsc.ksp_set_operators(%ksp, %a2, %a2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e5 : i32
  }

  // So does what a call returns.
  // CHECK-LABEL: llvm.func @returned_holder
  // CHECK: tessera.call @petsc.ksp_set_operators
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @returned_holder(%ksp: !llvm.ptr, %x: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %h = llvm.call @wrap(%a0) : (!llvm.ptr) -> !llvm.ptr
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%a1, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %e2 = llvm.call @touch(%h) : (!llvm.ptr) -> i32
    %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e3 = tessera.call @petsc.ksp_set_operators(%ksp, %a2, %a2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e3 : i32
  }

  // Nor does one LLVM marks nocapture.
  // CHECK-LABEL: llvm.func @nocapture
  // CHECK: tessera.call @petsc.ksp_set_operators({{.*}}) <arg_attrs = [{}, {tessera.property = ["SPD"]}, {tessera.property = ["SPD"]}]>
  llvm.func @nocapture(%ksp: !llvm.ptr, %x: !llvm.ptr, %reg: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @register_nocapture(%reg, %a0) : (!llvm.ptr, !llvm.ptr) -> i32
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e2 = llvm.call @fill_coo(%a1, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %e3 = llvm.call @touch(%reg) : (!llvm.ptr) -> i32
    %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e4 = tessera.call @petsc.ksp_set_operators(%ksp, %a2, %a2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e4 : i32
  }

  // A stack variable with no pointer in its type holds nothing, as
  // MatSetValues(A, 4, idx, ...) leaves `PetscInt idx[4]`.
  // CHECK-LABEL: llvm.func @scalar_array_not_holder
  // CHECK: tessera.call @petsc.ksp_set_operators({{.*}}) <arg_attrs = [{}, {tessera.property = ["SPD"]}, {tessera.property = ["SPD"]}]>
  llvm.func @scalar_array_not_holder(%ksp: !llvm.ptr, %x: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %idx = llvm.alloca %c1 x !llvm.array<4 x i32> : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @register(%idx, %a0) : (!llvm.ptr, !llvm.ptr) -> i32
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e2 = llvm.call @fill_coo(%a1, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %e3 = llvm.call @touch(%idx) : (!llvm.ptr) -> i32
    %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e4 = tessera.call @petsc.ksp_set_operators(%ksp, %a2, %a2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e4 : i32
  }

  // Nor, by the encapsulation assumption, does a pointer loaded from a
  // global, as MatCreate(PETSC_COMM_WORLD, &A) passes the communicator.
  // CHECK-LABEL: llvm.func @global_not_holder
  // CHECK: tessera.call @petsc.ksp_set_operators({{.*}}) <arg_attrs = [{}, {tessera.property = ["SPD"]}, {tessera.property = ["SPD"]}]>
  llvm.func @global_not_holder(%ksp: !llvm.ptr, %x: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %g = llvm.mlir.addressof @comm_world : !llvm.ptr
    %comm0 = llvm.load %g : !llvm.ptr -> !llvm.ptr
    %e0 = llvm.call @register(%comm0, %slot) : (!llvm.ptr, !llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%a0, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %comm1 = llvm.load %g : !llvm.ptr -> !llvm.ptr
    %e2 = llvm.call @touch(%comm1) : (!llvm.ptr) -> i32
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e3 = tessera.call @petsc.ksp_set_operators(%ksp, %a1, %a1) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e3 : i32
  }

  // A call that stores the matrix after the rule's call doesn't count.
  // CHECK-LABEL: llvm.func @stored_after_use
  // CHECK: tessera.call @petsc.ksp_set_operators({{.*}}) <arg_attrs = [{}, {tessera.property = ["SPD"]}, {tessera.property = ["SPD"]}]>
  llvm.func @stored_after_use(%ksp: !llvm.ptr, %x: !llvm.ptr, %reg: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%a0, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %e2 = llvm.call @touch(%reg) : (!llvm.ptr) -> i32
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e3 = tessera.call @petsc.ksp_set_operators(%ksp, %a1, %a1) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e4 = llvm.call @register(%reg, %a2) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e3 : i32
  }

  // Nor does one on the other branch of an if.
  // CHECK-LABEL: llvm.func @stored_on_other_branch
  // CHECK: tessera.call @petsc.ksp_set_operators({{.*}}) <arg_attrs = [{}, {tessera.property = ["SPD"]}, {tessera.property = ["SPD"]}]>
  llvm.func @stored_on_other_branch(%ksp: !llvm.ptr, %x: !llvm.ptr, %reg: !llvm.ptr, %flag: i1) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%a0, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %r = scf.if %flag -> (i32) {
      %e2 = llvm.call @touch(%reg) : (!llvm.ptr) -> i32
      %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
      %e3 = tessera.call @petsc.ksp_set_operators(%ksp, %a1, %a1) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
      scf.yield %e3 : i32
    } else {
      %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
      %e4 = llvm.call @register(%reg, %a2) : (!llvm.ptr, !llvm.ptr) -> i32
      scf.yield %e4 : i32
    }
    llvm.return %r : i32
  }

  // Writing through a pointer loaded from a holder can change the matrix.
  // CHECK-LABEL: llvm.func @written_through_holder
  // CHECK: tessera.call @petsc.ksp_set_operators
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @written_through_holder(%ksp: !llvm.ptr, %x: !llvm.ptr, %reg: !llvm.ptr, %v: f64) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @register(%reg, %a0) : (!llvm.ptr, !llvm.ptr) -> i32
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e2 = llvm.call @fill_coo(%a1, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %p = llvm.load %reg : !llvm.ptr -> !llvm.ptr
    llvm.store %v, %p : f64, !llvm.ptr
    %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e3 = tessera.call @petsc.ksp_set_operators(%ksp, %a2, %a2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e3 : i32
  }

  // Writing into the holder itself only replaces the pointer it holds.
  // CHECK-LABEL: llvm.func @written_into_holder
  // CHECK: tessera.call @petsc.ksp_set_operators({{.*}}) <arg_attrs = [{}, {tessera.property = ["SPD"]}, {tessera.property = ["SPD"]}]>
  llvm.func @written_into_holder(%ksp: !llvm.ptr, %x: !llvm.ptr, %reg: !llvm.ptr, %other: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @register(%reg, %a0) : (!llvm.ptr, !llvm.ptr) -> i32
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e2 = llvm.call @fill_coo(%a1, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.store %other, %reg : !llvm.ptr, !llvm.ptr
    %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e3 = tessera.call @petsc.ksp_set_operators(%ksp, %a2, %a2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e3 : i32
  }
}

// -----

// A pointer a call returned. What the callee guarantees of it is about the
// object it points to, so it holds until something changes the object, like a
// handle's from the call that established it.
module {
  tessera.define private @lapack.gesv(!llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [unit, unit], pure = false}
  tessera.define private @lapack.cond(!llvm.ptr {tessera.unchanged}) -> f64 attributes {argModes = [unit], pure = false}
  llvm.func @gram(i64) -> (!llvm.ptr {tessera.guarantees = ["SPD"]})
  llvm.func @scramble(!llvm.ptr)
  llvm.func @norm(!llvm.ptr {tessera.unchanged}) -> f64
  llvm.func @free(!llvm.ptr)
  // Returns an SPD object it may have taken from inside %x, as a getter does.
  llvm.func @gram_of(!llvm.ptr) -> (!llvm.ptr {tessera.guarantees = ["SPD"]})
  // Returns one that nothing of %x's is part of.
  llvm.func @gram_of_noescape(!llvm.ptr {tessera.unchanged, tessera.noescape}) -> (!llvm.ptr {tessera.guarantees = ["SPD"]})

  // CHECK-LABEL: llvm.func @untouched
  // CHECK: tessera.call @lapack.gesv({{.*}}) <arg_attrs = [{tessera.property = ["SPD"]}, {}]>
  llvm.func @untouched(%n: i64, %b: !llvm.ptr) -> i32 {
    %g = llvm.call @gram(%n) : (i64) -> !llvm.ptr
    %e = tessera.call @lapack.gesv(%g, %b) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.call @free(%g) : (!llvm.ptr) -> ()
    llvm.return %e : i32
  }

  // A call that leaves it unchanged in between.
  // CHECK-LABEL: llvm.func @unchanged_between
  // CHECK: tessera.call @lapack.gesv({{.*}}) <arg_attrs = [{tessera.property = ["SPD"]}, {}]>
  llvm.func @unchanged_between(%n: i64, %b: !llvm.ptr) -> i32 {
    %g = llvm.call @gram(%n) : (i64) -> !llvm.ptr
    %s = llvm.call @norm(%g) : (!llvm.ptr) -> f64
    %e = tessera.call @lapack.gesv(%g, %b) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e : i32
  }

  // A store into it, `G[0] = -1`.
  // CHECK-LABEL: llvm.func @stored_into
  // CHECK: tessera.call @lapack.gesv
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @stored_into(%n: i64, %b: !llvm.ptr, %v: f64) -> i32 {
    %g = llvm.call @gram(%n) : (i64) -> !llvm.ptr
    llvm.store %v, %g : f64, !llvm.ptr
    %e = tessera.call @lapack.gesv(%g, %b) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e : i32
  }

  // Or further in, `G[1] = -1`.
  // CHECK-LABEL: llvm.func @stored_inside
  // CHECK: tessera.call @lapack.gesv
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @stored_inside(%n: i64, %b: !llvm.ptr, %v: f64) -> i32 {
    %g = llvm.call @gram(%n) : (i64) -> !llvm.ptr
    %p = llvm.getelementptr %g[1] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %v, %p : f64, !llvm.ptr
    %e = tessera.call @lapack.gesv(%g, %b) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e : i32
  }

  // A call that might change it.
  // CHECK-LABEL: llvm.func @unannotated_call
  // CHECK: tessera.call @lapack.gesv
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @unannotated_call(%n: i64, %b: !llvm.ptr) -> i32 {
    %g = llvm.call @gram(%n) : (i64) -> !llvm.ptr
    llvm.call @scramble(%g) : (!llvm.ptr) -> ()
    %e = tessera.call @lapack.gesv(%g, %b) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e : i32
  }

  // Stored somewhere, it could reach a call the walk can't see.
  // CHECK-LABEL: llvm.func @escapes
  // CHECK: tessera.call @lapack.gesv
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @escapes(%n: i64, %b: !llvm.ptr, %out: !llvm.ptr) -> i32 {
    %g = llvm.call @gram(%n) : (i64) -> !llvm.ptr
    llvm.store %g, %out : !llvm.ptr, !llvm.ptr
    %e = tessera.call @lapack.gesv(%g, %b) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e : i32
  }

  // Returned by a call given %x, it may be something %x holds, so a call
  // that changes %x may change it.
  // CHECK-LABEL: llvm.func @returned_from_argument
  // CHECK: tessera.call @lapack.gesv
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @returned_from_argument(%x: !llvm.ptr, %b: !llvm.ptr) -> i32 {
    %g = llvm.call @gram_of(%x) : (!llvm.ptr) -> !llvm.ptr
    llvm.call @scramble(%x) : (!llvm.ptr) -> ()
    %e = tessera.call @lapack.gesv(%g, %b) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e : i32
  }

  // Or a pointer into it: a store straight into the array the call was given
  // can change what it returned, even with no pointer in the array's type.
  // CHECK-LABEL: llvm.func @returned_into_argument
  // CHECK: tessera.call @lapack.gesv
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @returned_into_argument(%b: !llvm.ptr, %v: f64) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %m = llvm.alloca %c1 x !llvm.array<16 x f64> : (i32) -> !llvm.ptr
    %g = llvm.call @gram_of(%m) : (!llvm.ptr) -> !llvm.ptr
    %p = llvm.getelementptr %m[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x f64>
    llvm.store %v, %p : f64, !llvm.ptr
    %e = tessera.call @lapack.gesv(%g, %b) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e : i32
  }

  // Unless the call was given %x noescape.
  // CHECK-LABEL: llvm.func @returned_from_noescape_argument
  // CHECK: tessera.call @lapack.gesv({{.*}}) <arg_attrs = [{tessera.property = ["SPD"]}, {}]>
  llvm.func @returned_from_noescape_argument(%x: !llvm.ptr, %b: !llvm.ptr) -> i32 {
    %g = llvm.call @gram_of_noescape(%x) : (!llvm.ptr) -> !llvm.ptr
    llvm.call @scramble(%x) : (!llvm.ptr) -> ()
    %e = tessera.call @lapack.gesv(%g, %b) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e : i32
  }

  // A new object each iteration. The solve may store G in %b, but the G it
  // was given in the iteration before is another object, so a call given %b
  // before this iteration's solve can't change this one.
  // CHECK-LABEL: llvm.func @new_each_iteration
  // CHECK: tessera.call @lapack.gesv({{.*}}) <arg_attrs = [{tessera.property = ["SPD"]}, {}]>
  llvm.func @new_each_iteration(%n: i64, %b: !llvm.ptr) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %g = llvm.call @gram(%n) : (i64) -> !llvm.ptr
      llvm.call @scramble(%b) : (!llvm.ptr) -> ()
      %e = tessera.call @lapack.gesv(%g, %b) : (!llvm.ptr, !llvm.ptr) -> i32
      llvm.call @free(%g) : (!llvm.ptr) -> ()
    }
    llvm.return
  }

  // One object for every iteration: the solve may have stored it in %b in the
  // iteration before, and the call given %b may change it through that.
  // CHECK-LABEL: llvm.func @same_each_iteration
  // CHECK: tessera.call @lapack.gesv
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @same_each_iteration(%n: i64, %b: !llvm.ptr) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %g = llvm.call @gram(%n) : (i64) -> !llvm.ptr
    scf.for %i = %c0 to %c4 step %c1 {
      llvm.call @scramble(%b) : (!llvm.ptr) -> ()
      %e = tessera.call @lapack.gesv(%g, %b) : (!llvm.ptr, !llvm.ptr) -> i32
    }
    llvm.return
  }

  // True at every call given it, the fact is recorded on the value itself,
  // with no copy: a rule naming it twice still sees one value.
  // CHECK-LABEL: llvm.func @every_use
  // CHECK: %[[G:.*]] = llvm.call @gram
  // CHECK-NEXT: tessera.call @lapack.cond(%[[G]]) <arg_attrs = [{tessera.property = ["SPD"]}]>
  // CHECK-NEXT: tessera.call @lapack.cond(%[[G]]) <arg_attrs = [{tessera.property = ["SPD"]}]>
  llvm.func @every_use(%n: i64) -> f64 {
    %g = llvm.call @gram(%n) : (i64) -> !llvm.ptr
    %c0 = tessera.call @lapack.cond(%g) : (!llvm.ptr) -> f64
    %c1 = tessera.call @lapack.cond(%g) : (!llvm.ptr) -> f64
    %s = llvm.fadd %c0, %c1 : f64
    llvm.return %s : f64
  }

  // True at the first only, it goes on a copy for that call.
  // CHECK-LABEL: llvm.func @first_use_only
  // CHECK: %[[G:.*]] = llvm.call @gram
  // CHECK-NEXT: %[[F:.*]] = llvm.freeze %[[G]]
  // CHECK-NEXT: tessera.call @lapack.cond(%[[F]]) <arg_attrs = [{tessera.property = ["SPD"]}]>
  // CHECK-NEXT: llvm.call @scramble(%[[G]])
  // CHECK-NEXT: tessera.call @lapack.cond(%[[G]]) :
  llvm.func @first_use_only(%n: i64) -> f64 {
    %g = llvm.call @gram(%n) : (i64) -> !llvm.ptr
    %c0 = tessera.call @lapack.cond(%g) : (!llvm.ptr) -> f64
    llvm.call @scramble(%g) : (!llvm.ptr) -> ()
    %c1 = tessera.call @lapack.cond(%g) : (!llvm.ptr) -> f64
    %s = llvm.fadd %c0, %c1 : f64
    llvm.return %s : f64
  }
}

// -----

// Stores the function makes. Storing a pointer out of the matrix, such as its
// data, somewhere makes that memory a holder, as a call storing it would. A
// value it's passed on as, such as a select's result or a loop's carried
// value, reaches the matrix too. Putting one where the search can't follow
// -- a global, an integer -- makes the matrix shared, as a parameter's
// object is: any call that may write memory outside the function ends its
// facts.

module {
  tessera.define private @petsc.ksp_set_operators(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [unit, unit, unit], pure = false}
  llvm.mlir.global external @g() {addr_space = 0 : i32} : !llvm.ptr
  llvm.func @mat_create(!llvm.ptr) -> i32
  llvm.func @fill_coo(!llvm.ptr {tessera.establishes = ["SPD"], tessera.noescape}, !llvm.ptr) -> i32
  llvm.func @touch(!llvm.ptr) -> i32

  // Through a local variable the data was stored in.
  // CHECK-LABEL: llvm.func @stored_in_local
  // CHECK: tessera.call @petsc.ksp_set_operators
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @stored_in_local(%ksp: !llvm.ptr, %x: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %box = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%a0, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %data = llvm.load %a1 : !llvm.ptr -> !llvm.ptr
    llvm.store %data, %box : !llvm.ptr, !llvm.ptr
    %b = llvm.load %box : !llvm.ptr -> !llvm.ptr
    %e2 = llvm.call @touch(%b) : (!llvm.ptr) -> i32
    %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e3 = tessera.call @petsc.ksp_set_operators(%ksp, %a2, %a2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e3 : i32
  }

  // Through a copy of that variable.
  // CHECK-LABEL: llvm.func @copied_by_memcpy
  // CHECK: tessera.call @petsc.ksp_set_operators
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @copied_by_memcpy(%ksp: !llvm.ptr, %x: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i64) : i64
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %box = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %copy = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%a0, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %data = llvm.load %a1 : !llvm.ptr -> !llvm.ptr
    llvm.store %data, %box : !llvm.ptr, !llvm.ptr
    "llvm.intr.memcpy"(%copy, %box, %c8) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %b = llvm.load %copy : !llvm.ptr -> !llvm.ptr
    %e2 = llvm.call @touch(%b) : (!llvm.ptr) -> i32
    %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e3 = tessera.call @petsc.ksp_set_operators(%ksp, %a2, %a2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e3 : i32
  }

  // A call given neither the matrix nor a holder of it can't change it.
  // CHECK-LABEL: llvm.func @stored_in_local_other_call
  // CHECK: tessera.call @petsc.ksp_set_operators({{.*}}) <arg_attrs = [{}, {tessera.property = ["SPD"]}, {tessera.property = ["SPD"]}]>
  llvm.func @stored_in_local_other_call(%ksp: !llvm.ptr, %x: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %box = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%a0, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %data = llvm.load %a1 : !llvm.ptr -> !llvm.ptr
    llvm.store %data, %box : !llvm.ptr, !llvm.ptr
    %e2 = llvm.call @touch(%x) : (!llvm.ptr) -> i32
    %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e3 = tessera.call @petsc.ksp_set_operators(%ksp, %a2, %a2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e3 : i32
  }

  // Stored in a global, the data can be reached by a call given neither.
  // CHECK-LABEL: llvm.func @stored_in_global
  // CHECK: tessera.call @petsc.ksp_set_operators
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @stored_in_global(%ksp: !llvm.ptr, %x: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %g = llvm.mlir.addressof @g : !llvm.ptr
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%a0, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %data = llvm.load %a1 : !llvm.ptr -> !llvm.ptr
    llvm.store %data, %g : !llvm.ptr, !llvm.ptr
    %e2 = llvm.call @touch(%x) : (!llvm.ptr) -> i32
    %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e3 = tessera.call @petsc.ksp_set_operators(%ksp, %a2, %a2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e3 : i32
  }

  // Passed on as a select's result.
  // CHECK-LABEL: llvm.func @through_select
  // CHECK: tessera.call @petsc.ksp_set_operators
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @through_select(%ksp: !llvm.ptr, %x: !llvm.ptr, %c: i1) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%a0, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %data = llvm.load %a1 : !llvm.ptr -> !llvm.ptr
    %s = llvm.select %c, %data, %x : i1, !llvm.ptr
    %e2 = llvm.call @touch(%s) : (!llvm.ptr) -> i32
    %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e3 = tessera.call @petsc.ksp_set_operators(%ksp, %a2, %a2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e3 : i32
  }

  // Carried by a loop, and given as its result.
  // CHECK-LABEL: llvm.func @carried_by_loop
  // CHECK: tessera.call @petsc.ksp_set_operators
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @carried_by_loop(%ksp: !llvm.ptr, %x: !llvm.ptr, %count: i64) -> i32 {
    %n = arith.index_cast %count : i64 to index
    %c0 = arith.constant 0 : index
    %c1i = arith.constant 1 : index
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%a0, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %data = llvm.load %a1 : !llvm.ptr -> !llvm.ptr
    %last = scf.for %i = %c0 to %n step %c1i iter_args(%p = %data) -> (!llvm.ptr) {
      scf.yield %p : !llvm.ptr
    }
    %e2 = llvm.call @touch(%last) : (!llvm.ptr) -> i32
    %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e3 = tessera.call @petsc.ksp_set_operators(%ksp, %a2, %a2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e3 : i32
  }

  // Turned into an integer, which the search doesn't follow.
  // CHECK-LABEL: llvm.func @as_integer
  // CHECK: tessera.call @petsc.ksp_set_operators
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @as_integer(%ksp: !llvm.ptr, %x: !llvm.ptr) -> i64 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%a0, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %data = llvm.load %a1 : !llvm.ptr -> !llvm.ptr
    %i = llvm.ptrtoint %data : !llvm.ptr to i64
    %e2 = llvm.call @touch(%x) : (!llvm.ptr) -> i32
    %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e3 = tessera.call @petsc.ksp_set_operators(%ksp, %a2, %a2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %i : i64
  }

  // Passed through an scf.if, which the search follows: a call given
  // neither the matrix nor what the if returns can't change it.
  // CHECK-LABEL: llvm.func @through_if_other_call
  // CHECK: tessera.call @petsc.ksp_set_operators({{.*}}) <arg_attrs = [{}, {tessera.property = ["SPD"]}, {tessera.property = ["SPD"]}]>
  llvm.func @through_if_other_call(%ksp: !llvm.ptr, %x: !llvm.ptr, %c: i1) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%a0, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %data = llvm.load %a1 : !llvm.ptr -> !llvm.ptr
    %p = scf.if %c -> (!llvm.ptr) {
      scf.yield %data : !llvm.ptr
    } else {
      scf.yield %x : !llvm.ptr
    }
    %e2 = llvm.call @touch(%x) : (!llvm.ptr) -> i32
    %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e3 = tessera.call @petsc.ksp_set_operators(%ksp, %a2, %a2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e3 : i32
  }
}

// -----

// A struct holding pointers, returned by value, is a handle too: what its
// callee guarantees is of what the pointers point to, which may change.

module {
  tessera.define private @lib.solve(!llvm.struct<(ptr, i64)>) -> f64 attributes {argModes = [unit], pure = false}
  llvm.func @make_view(i64) -> (!llvm.struct<(ptr, i64)> {tessera.guarantees = ["SPD"]})
  llvm.func @scale_view(!llvm.struct<(ptr, i64)>)
  llvm.func @print_view(!llvm.struct<(ptr, i64)> {tessera.unchanged, tessera.noescape})

  // CHECK-LABEL: llvm.func @view_unchanged
  // CHECK: tessera.call @lib.solve(%{{.*}}) <arg_attrs = [{tessera.property = ["SPD"]}]>
  llvm.func @view_unchanged(%n: i64) -> f64 {
    %v = llvm.call @make_view(%n) : (i64) -> !llvm.struct<(ptr, i64)>
    llvm.call @print_view(%v) : (!llvm.struct<(ptr, i64)>) -> ()
    %r = tessera.call @lib.solve(%v) : (!llvm.struct<(ptr, i64)>) -> f64
    llvm.return %r : f64
  }

  // CHECK-LABEL: llvm.func @view_changed
  // CHECK: tessera.call @lib.solve
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @view_changed(%n: i64) -> f64 {
    %v = llvm.call @make_view(%n) : (i64) -> !llvm.struct<(ptr, i64)>
    llvm.call @scale_view(%v) : (!llvm.struct<(ptr, i64)>) -> ()
    %r = tessera.call @lib.solve(%v) : (!llvm.struct<(ptr, i64)>) -> f64
    llvm.return %r : f64
  }
}
