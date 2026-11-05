// RUN: enzymexlamlir-opt %s -tessera-propagate-properties | FileCheck %s

// Calls that may change the object a handle names end its facts, however the
// handle reaches them.

module {
  tessera.define private @petsc.ksp_set_operators(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [unit, unit, unit], pure = false}
  llvm.func @mat_create(!llvm.ptr) -> i32
  llvm.func @mat_destroy(!llvm.ptr) -> i32
  llvm.func @fill_coo(!llvm.ptr {tessera.establishes = ["SPD"]}, !llvm.ptr) -> i32
  llvm.func @petsc_printf(!llvm.ptr, ...) -> i32
  llvm.func @mat_scale_as1(!llvm.ptr<1>, f64) -> i32
  // Keeps A's object SPD, and writes another matrix to its second argument.
  llvm.func @mat_split(!llvm.ptr, !llvm.ptr) attributes {tessera.preserves = [{inputs = [0], property = "SPD"}]}

  // The handle given as a variadic argument.
  // CHECK-LABEL: llvm.func @variadic_argument
  // CHECK: tessera.call @petsc.ksp_set_operators
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @variadic_argument(%ksp: !llvm.ptr, %x: !llvm.ptr, %fmt: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%a0, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e2 = llvm.call @petsc_printf(%fmt, %a1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e3 = tessera.call @petsc.ksp_set_operators(%ksp, %a2, %a2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e3 : i32
  }

  // The handle given through an address space cast.
  // CHECK-LABEL: llvm.func @addrspacecast_argument
  // CHECK: tessera.call @petsc.ksp_set_operators
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @addrspacecast_argument(%ksp: !llvm.ptr, %x: !llvm.ptr, %s: f64) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%a1, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %c = llvm.addrspacecast %a2 : !llvm.ptr to !llvm.ptr<1>
    %e2 = llvm.call @mat_scale_as1(%c, %s) : (!llvm.ptr<1>, f64) -> i32
    %a3 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e3 = tessera.call @petsc.ksp_set_operators(%ksp, %a3, %a3) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e3 : i32
  }

  // A preserving call also given &A may store another matrix into A: what it
  // preserves is of the old one.
  // CHECK-LABEL: llvm.func @preserves_but_stores_handle
  // CHECK: tessera.call @petsc.ksp_set_operators
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @preserves_but_stores_handle(%ksp: !llvm.ptr, %x: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%a0, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    llvm.call @mat_split(%a1, %slot) : (!llvm.ptr, !llvm.ptr) -> ()
    %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e3 = tessera.call @petsc.ksp_set_operators(%ksp, %a2, %a2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e3 : i32
  }

  // The same call given another variable keeps the property.
  // CHECK-LABEL: llvm.func @preserves_keeps
  // CHECK: tessera.call @petsc.ksp_set_operators({{.*}}) <arg_attrs = [{}, {tessera.property = ["SPD"]}, {tessera.property = ["SPD"]}]>
  llvm.func @preserves_keeps(%ksp: !llvm.ptr, %x: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %other = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%a0, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    llvm.call @mat_split(%a1, %other) : (!llvm.ptr, !llvm.ptr) -> ()
    %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e3 = tessera.call @petsc.ksp_set_operators(%ksp, %a2, %a2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e3 : i32
  }

  // Loaded before another matrix is stored into A, the handle fill_coo is
  // given names the old one: what it establishes isn't of A's matrix.
  // CHECK-LABEL: llvm.func @stale_load
  // CHECK: tessera.call @petsc.ksp_set_operators
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @stale_load(%ksp: !llvm.ptr, %x: !llvm.ptr, %other: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %old = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    llvm.store %other, %slot : !llvm.ptr, !llvm.ptr
    %e1 = llvm.call @fill_coo(%old, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %a = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e2 = tessera.call @petsc.ksp_set_operators(%ksp, %a, %a) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e2 : i32
  }

  // Loaded after the store, it names the matrix A holds from then on.
  // CHECK-LABEL: llvm.func @load_after_store
  // CHECK: tessera.call @petsc.ksp_set_operators({{.*}}) <arg_attrs = [{}, {tessera.property = ["SPD"]}, {tessera.property = ["SPD"]}]>
  llvm.func @load_after_store(%ksp: !llvm.ptr, %x: !llvm.ptr, %other: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    llvm.store %other, %slot : !llvm.ptr, !llvm.ptr
    %new = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%new, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %a = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e2 = tessera.call @petsc.ksp_set_operators(%ksp, %a, %a) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e2 : i32
  }

  // A store into A after the use, in the same `if`, as `MatDestroy(&A)` is
  // there, doesn't make the handle stale where it's used.
  // CHECK-LABEL: llvm.func @destroyed_after_use
  // CHECK: tessera.call @petsc.ksp_set_operators({{.*}}) <arg_attrs = [{}, {tessera.property = ["SPD"]}, {tessera.property = ["SPD"]}]>
  llvm.func @destroyed_after_use(%ksp: !llvm.ptr, %x: !llvm.ptr, %c: i1) -> i32 {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%a, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %r = scf.if %c -> (i32) {
      %e2 = tessera.call @petsc.ksp_set_operators(%ksp, %a, %a) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
      %e3 = llvm.call @mat_destroy(%slot) : (!llvm.ptr) -> i32
      scf.yield %e2 : i32
    } else {
      scf.yield %c0 : i32
    }
    llvm.return %r : i32
  }

  // In a loop, a store after the use runs before it on the next iteration.
  // CHECK-LABEL: llvm.func @destroyed_after_use_in_loop
  // CHECK: tessera.call @petsc.ksp_set_operators
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @destroyed_after_use_in_loop(%ksp: !llvm.ptr, %x: !llvm.ptr, %count: i64) -> i32 {
    %n = arith.index_cast %count : i64 to index
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%a, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    scf.for %i = %i0 to %n step %i1 {
      %e2 = tessera.call @petsc.ksp_set_operators(%ksp, %a, %a) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
      %e3 = llvm.call @mat_destroy(%slot) : (!llvm.ptr) -> i32
    }
    llvm.return %e1 : i32
  }
}
