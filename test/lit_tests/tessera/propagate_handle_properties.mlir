// RUN: enzymexlamlir-opt %s -tessera-propagate-properties -split-input-file | FileCheck %s

// A handle, like a PETSc Mat, is a pointer to an object a library changes in
// place. `tessera.establishes` on a parameter says the object is SPD once the
// call returns; `tessera.readonly` says a call leaves it alone. At a
// tessera.call given the handle, the walk back from the call finds the
// establishing call, and records the property there if nothing in between
// could have changed the object, or put another one in the variable.
//
// The IR is shaped as the raise pipeline leaves a C program: `Mat A` lives in
// a stack slot, since `MatCreate(comm, &A)` takes its address, and each use
// of A is a fresh load from it. PetscCall's error checks nest everything after
// a call inside an `scf.if`.

module {
  tessera.define private @petsc.ksp_set_operators(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [unit, unit, unit], pure = false}
  llvm.func @mat_create(!llvm.ptr) -> i32
  llvm.func @mat_destroy(!llvm.ptr) -> i32
  llvm.func @fill_coo(!llvm.ptr {tessera.establishes = ["SPD"]}, !llvm.ptr) -> i32
  llvm.func @mat_mult(!llvm.ptr {tessera.readonly}, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @mat_scale(!llvm.ptr, f64) -> i32

  // Found through separate loads of one slot, a readonly call, and two
  // levels of error checks. Both operands that are the handle get it.
  // CHECK-LABEL: llvm.func @found
  // CHECK: tessera.call @petsc.ksp_set_operators(%{{.*}}, %[[A:[0-9]+]], %[[A]]) <arg_attrs = [{}, {tessera.property = ["SPD"]}, {tessera.property = ["SPD"]}]>
  llvm.func @found(%ksp: !llvm.ptr, %x: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %m0 = "enzymexla.pointer2memref"(%slot) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %a0 = affine.load %m0[0] : memref<?x!llvm.ptr>
    %e1 = llvm.call @fill_coo(%a0, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %f1 = llvm.icmp "ne" %e1, %c0 : i32
    %r1 = scf.if %f1 -> (i32) {
      scf.yield %e1 : i32
    } else {
      %m1 = "enzymexla.pointer2memref"(%slot) : (!llvm.ptr) -> memref<?x!llvm.ptr>
      %a1 = affine.load %m1[0] : memref<?x!llvm.ptr>
      %e2 = llvm.call @mat_mult(%a1, %x, %x) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
      %f2 = llvm.icmp "ne" %e2, %c0 : i32
      %r2 = scf.if %f2 -> (i32) {
        scf.yield %e2 : i32
      } else {
        %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
        %e3 = tessera.call @petsc.ksp_set_operators(%ksp, %a2, %a2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
        scf.yield %e3 : i32
      }
      scf.yield %r2 : i32
    }
    llvm.return %r1 : i32
  }

  // A call that may change the object ends the walk.
  // CHECK-LABEL: llvm.func @changed
  // CHECK: tessera.call @petsc.ksp_set_operators
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @changed(%ksp: !llvm.ptr, %x: !llvm.ptr, %s: f64) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%a0, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e2 = llvm.call @mat_scale(%a1, %s) : (!llvm.ptr, f64) -> i32
    %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e3 = tessera.call @petsc.ksp_set_operators(%ksp, %a2, %a2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e3 : i32
  }

  // So does a call given the slot itself, which can put another object in it.
  // CHECK-LABEL: llvm.func @slot_passed
  // CHECK: tessera.call @petsc.ksp_set_operators
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @slot_passed(%ksp: !llvm.ptr, %x: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%a0, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %e2 = llvm.call @mat_destroy(%slot) : (!llvm.ptr) -> i32
    %e3 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e4 = tessera.call @petsc.ksp_set_operators(%ksp, %a2, %a2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e4 : i32
  }

  // And a store into the slot.
  // CHECK-LABEL: llvm.func @slot_stored
  // CHECK: tessera.call @petsc.ksp_set_operators
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @slot_stored(%ksp: !llvm.ptr, %x: !llvm.ptr, %other: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%a0, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %m = "enzymexla.pointer2memref"(%slot) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    affine.store %other, %m[0] : memref<?x!llvm.ptr>
    %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e2 = tessera.call @petsc.ksp_set_operators(%ksp, %a2, %a2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e2 : i32
  }

  // Before the establishing call the object is not SPD yet.
  // CHECK-LABEL: llvm.func @before
  // CHECK: tessera.call @petsc.ksp_set_operators
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @before(%ksp: !llvm.ptr, %x: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = tessera.call @petsc.ksp_set_operators(%ksp, %a0, %a0) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e2 = llvm.call @fill_coo(%a1, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e2 : i32
  }

  // An establishing call that only may have run does not count.
  // CHECK-LABEL: llvm.func @maybe
  // CHECK: tessera.call @petsc.ksp_set_operators
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @maybe(%ksp: !llvm.ptr, %x: !llvm.ptr, %flag: i1) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    scf.if %flag {
      %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
      %e1 = llvm.call @fill_coo(%a0, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    }
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e2 = tessera.call @petsc.ksp_set_operators(%ksp, %a1, %a1) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e2 : i32
  }
}

// -----

// Loops. One that only reads the object is passed over; one that may change
// it ends the walk, and so does leaving a loop whose body changes the object
// after the call, since that runs before the call on the next iteration.
module {
  tessera.define private @petsc.ksp_set_operators(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [unit, unit, unit], pure = false}
  llvm.func @mat_create(!llvm.ptr) -> i32
  llvm.func @fill_coo(!llvm.ptr {tessera.establishes = ["SPD"]}, !llvm.ptr) -> i32
  llvm.func @mat_mult(!llvm.ptr {tessera.readonly}, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @mat_scale(!llvm.ptr, f64) -> i32

  // CHECK-LABEL: llvm.func @readonly_loop
  // CHECK: tessera.call @petsc.ksp_set_operators({{.*}}) <arg_attrs = [{}, {tessera.property = ["SPD"]}, {tessera.property = ["SPD"]}]>
  llvm.func @readonly_loop(%ksp: !llvm.ptr, %x: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %lb = arith.constant 0 : index
    %ub = arith.constant 100 : index
    %step = arith.constant 1 : index
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%a0, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    scf.for %i = %lb to %ub step %step {
      %ai = llvm.load %slot : !llvm.ptr -> !llvm.ptr
      %ei = llvm.call @mat_mult(%ai, %x, %x) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    }
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e2 = tessera.call @petsc.ksp_set_operators(%ksp, %a1, %a1) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e2 : i32
  }

  // CHECK-LABEL: llvm.func @changing_loop
  // CHECK: tessera.call @petsc.ksp_set_operators
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @changing_loop(%ksp: !llvm.ptr, %x: !llvm.ptr, %s: f64) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %lb = arith.constant 0 : index
    %ub = arith.constant 100 : index
    %step = arith.constant 1 : index
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%a0, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    scf.for %i = %lb to %ub step %step {
      %ai = llvm.load %slot : !llvm.ptr -> !llvm.ptr
      %ei = llvm.call @mat_scale(%ai, %s) : (!llvm.ptr, f64) -> i32
    }
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e2 = tessera.call @petsc.ksp_set_operators(%ksp, %a1, %a1) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e2 : i32
  }

  // CHECK-LABEL: llvm.func @inside_changing_loop
  // CHECK: tessera.call @petsc.ksp_set_operators
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @inside_changing_loop(%ksp: !llvm.ptr, %x: !llvm.ptr, %s: f64) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %lb = arith.constant 0 : index
    %ub = arith.constant 100 : index
    %step = arith.constant 1 : index
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%a0, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    scf.for %i = %lb to %ub step %step {
      %ai = llvm.load %slot : !llvm.ptr -> !llvm.ptr
      %ei = tessera.call @petsc.ksp_set_operators(%ksp, %ai, %ai) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
      %aj = llvm.load %slot : !llvm.ptr -> !llvm.ptr
      %ej = llvm.call @mat_scale(%aj, %s) : (!llvm.ptr, f64) -> i32
    }
    llvm.return %e1 : i32
  }
}

// -----

// Where the walk cannot tell the handle is the same, it records nothing.
module {
  tessera.define private @petsc.ksp_set_operators(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [unit, unit, unit], pure = false}
  llvm.func @mat_create(!llvm.ptr) -> i32
  llvm.func @fill_coo(!llvm.ptr {tessera.establishes = ["SPD"]}, !llvm.ptr) -> i32

  // The handle is copied into another variable, which a call could be given
  // without it looking like this handle.
  // CHECK-LABEL: llvm.func @copied
  // CHECK: tessera.call @petsc.ksp_set_operators
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @copied(%ksp: !llvm.ptr, %x: !llvm.ptr, %copy: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    llvm.store %a0, %copy : !llvm.ptr, !llvm.ptr
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%a1, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e2 = tessera.call @petsc.ksp_set_operators(%ksp, %a2, %a2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e2 : i32
  }

  // One load given to two calls: a fact recorded on the second would be read
  // as true of the value at the first, before it was established.
  // CHECK-LABEL: llvm.func @shared_load
  // CHECK: tessera.call @petsc.ksp_set_operators
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @shared_load(%ksp: !llvm.ptr, %x: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%a, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %e2 = tessera.call @petsc.ksp_set_operators(%ksp, %a, %a) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e2 : i32
  }
}
