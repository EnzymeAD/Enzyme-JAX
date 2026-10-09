// RUN: enzymexlamlir-opt %s -lift-tessera-annotations -llvm-to-tessera -tessera-propagate-properties | FileCheck %s --check-prefix=FACT
// RUN: enzymexlamlir-opt %s -lift-tessera-annotations -llvm-to-tessera -tessera-propagate-properties -tessera-to-llvm | FileCheck %s --check-prefix=LOWER

// A fact stated on a statement, for a handle built in place rather than by a
// function:
//
//   PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
//   __attribute__((tessera_guarantees("SPD(A)")));
//
// reaches the module as a call, after the statement, to a marker the plugin
// generated: a weak function that does nothing, annotated with the guarantee
// on its parameter and with tessera_fact. The call establishes the fact as
// any call guaranteeing it of a handle does, and once the facts have been
// used, tessera-to-llvm removes the calls and the marker. (llvm-to-tessera
// has dropped the annotation table by then, which named the marker too.)

module {
  llvm.mlir.global private unnamed_addr constant @".str.spd_arg0"("tessera_guarantees=SPD:arg0\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.assumes_arg0"("tessera_assumes=SPD:arg0\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.fact"("tessera_fact\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.file"("ex2.c\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<4 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %zero = llvm.mlir.zero : !llvm.ptr
    %line = llvm.mlir.constant(1 : i32) : i32
    %file = llvm.mlir.addressof @".str.file" : !llvm.ptr
    %spd_arg0 = llvm.mlir.addressof @".str.spd_arg0" : !llvm.ptr
    %fact = llvm.mlir.addressof @".str.fact" : !llvm.ptr
    %assumes_arg0 = llvm.mlir.addressof @".str.assumes_arg0" : !llvm.ptr
    %f_marker = llvm.mlir.addressof @__tessera_fact_0 : !llvm.ptr
    %f_assume = llvm.mlir.addressof @__tessera_fact_1 : !llvm.ptr
    %undef = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0a = llvm.insertvalue %f_marker, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0b = llvm.insertvalue %spd_arg0, %e0a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0c = llvm.insertvalue %file, %e0b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0d = llvm.insertvalue %line, %e0c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0 = llvm.insertvalue %zero, %e0d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1a = llvm.insertvalue %f_marker, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1b = llvm.insertvalue %fact, %e1a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1c = llvm.insertvalue %file, %e1b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1d = llvm.insertvalue %line, %e1c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1 = llvm.insertvalue %zero, %e1d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2a = llvm.insertvalue %f_assume, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2b = llvm.insertvalue %assumes_arg0, %e2a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2c = llvm.insertvalue %file, %e2b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2d = llvm.insertvalue %line, %e2c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2 = llvm.insertvalue %zero, %e2d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3a = llvm.insertvalue %f_assume, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3b = llvm.insertvalue %fact, %e3a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3c = llvm.insertvalue %file, %e3b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3d = llvm.insertvalue %line, %e3c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3 = llvm.insertvalue %zero, %e3d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %arr0 = llvm.mlir.undef : !llvm.array<4 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %arr1 = llvm.insertvalue %e0, %arr0[0] : !llvm.array<4 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %arr2 = llvm.insertvalue %e1, %arr1[1] : !llvm.array<4 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %arr3 = llvm.insertvalue %e2, %arr2[2] : !llvm.array<4 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %arr4 = llvm.insertvalue %e3, %arr3[3] : !llvm.array<4 x struct<(ptr, ptr, ptr, i32, ptr)>>
    llvm.return %arr4 : !llvm.array<4 x struct<(ptr, ptr, ptr, i32, ptr)>>
  }

  tessera.define private @petsc.ksp_set_operators(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [unit, unit, unit], pure = false, tessera.original_name = "KSPSetOperators"}
  llvm.func @mat_create(!llvm.ptr) -> i32
  llvm.func @mat_assembly_end(!llvm.ptr, i32) -> i32
  llvm.func @mat_permute(!llvm.ptr, !llvm.ptr) -> i32
  llvm.func @mat_destroy(!llvm.ptr) -> i32
  llvm.func @mat_scale(!llvm.ptr, f64) -> i32

  // The marker is lifted as FillCOO's guarantee is, and marked. It is given
  // the variable's address: the fact is of whatever handle A holds there.
  // FACT: llvm.func weak @__tessera_fact_0(%{{.*}}: !llvm.ptr {tessera.establishes = ["SPD"]}) attributes {{{.*}}tessera.fact_marker{{.*}}}
  llvm.func weak @__tessera_fact_0(%a: !llvm.ptr) attributes {no_inline} {
    llvm.return
  }

  // The call after assembly establishes SPD of the object A names, so
  // KSPSetOperators sees it; then the call and the marker are gone, and the
  // program is as it was without the fact.
  // FACT-LABEL: llvm.func @main
  // FACT: llvm.call @__tessera_fact_0
  // FACT: tessera.call @petsc.ksp_set_operators(%{{.*}}, %[[A:[0-9]+]], %[[A]]) <arg_attrs = [{}, {tessera.property = ["SPD"]}, {tessera.property = ["SPD"]}]>
  // LOWER-NOT: __tessera_fact_0
  // LOWER-LABEL: llvm.func @main
  // LOWER: llvm.call @mat_assembly_end
  // LOWER-NOT: __tessera_fact_0
  // LOWER: llvm.call @KSPSetOperators
  // LOWER-NOT: __tessera_fact_0
  llvm.func @main(%ksp: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @mat_assembly_end(%a0, %c1) : (!llvm.ptr, i32) -> i32
    llvm.call @__tessera_fact_0(%slot) : (!llvm.ptr) -> ()
    %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e2 = tessera.call @petsc.ksp_set_operators(%ksp, %a2, %a2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e2 : i32
  }

  // A may be replaced in a branch before the fact, as ex18's -permute does.
  // The fact is stated after the branch, of whichever matrix A then holds.
  // FACT-LABEL: llvm.func @replaced_in_branch
  // FACT: tessera.call @petsc.ksp_set_operators(%{{.*}}, %[[A:[0-9]+]], %[[A]]) <arg_attrs = [{}, {tessera.property = ["SPD"]}, {tessera.property = ["SPD"]}]>
  llvm.func @replaced_in_branch(%ksp: !llvm.ptr, %permute: i1) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %perm = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    scf.if %permute {
      %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
      %e1 = llvm.call @mat_permute(%a0, %perm) : (!llvm.ptr, !llvm.ptr) -> i32
      %e2 = llvm.call @mat_destroy(%slot) : (!llvm.ptr) -> i32
      %p = llvm.load %perm : !llvm.ptr -> !llvm.ptr
      llvm.store %p, %slot : !llvm.ptr, !llvm.ptr
    }
    llvm.call @__tessera_fact_0(%slot) : (!llvm.ptr) -> ()
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e3 = tessera.call @petsc.ksp_set_operators(%ksp, %a1, %a1) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e3 : i32
  }

  // A fact holds only until something may change the object: here a call
  // given A that is not readonly.
  // FACT-LABEL: llvm.func @changed_after
  // FACT: tessera.call @petsc.ksp_set_operators
  // FACT-NOT: tessera.property
  // FACT: llvm.return
  llvm.func @changed_after(%ksp: !llvm.ptr, %s: f64) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    llvm.call @__tessera_fact_0(%slot) : (!llvm.ptr) -> ()
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @mat_scale(%a0, %s) : (!llvm.ptr, f64) -> i32
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e2 = tessera.call @petsc.ksp_set_operators(%ksp, %a1, %a1) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e2 : i32
  }

  // Or until a call given the variable's address, which may put another
  // matrix in it. Only a marker is taken to leave the variable alone.
  // FACT-LABEL: llvm.func @replaced_after
  // FACT: tessera.call @petsc.ksp_set_operators
  // FACT-NOT: tessera.property
  // FACT: llvm.return
  llvm.func @replaced_after(%ksp: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    llvm.call @__tessera_fact_0(%slot) : (!llvm.ptr) -> ()
    %e1 = llvm.call @mat_destroy(%slot) : (!llvm.ptr) -> i32
    %e2 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e3 = tessera.call @petsc.ksp_set_operators(%ksp, %a0, %a0) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e3 : i32
  }

  // The same statement with tessera_assumes:
  //
  //   __attribute__((tessera_assumes("SPD(A)")));
  //
  // gets a marker of its own, annotated with the assumption on its parameter
  // rather than a guarantee. It is lifted as any assumption is, to a
  // tessera.property on the parameter; since the marker does nothing, that
  // holds after the call too, of whatever handle A holds there. The marker is
  // removed the same way.
  // FACT: llvm.func weak @__tessera_fact_1(%{{.*}}: !llvm.ptr {tessera.property = ["SPD"]}) attributes {{{.*}}tessera.fact_marker{{.*}}}
  // LOWER-NOT: __tessera_fact_1
  llvm.func weak @__tessera_fact_1(%a: !llvm.ptr) attributes {no_inline} {
    llvm.return
  }

  // FACT-LABEL: llvm.func @assumed
  // FACT: llvm.call @__tessera_fact_1
  // FACT: tessera.call @petsc.ksp_set_operators(%{{.*}}, %[[A:[0-9]+]], %[[A]]) <arg_attrs = [{}, {tessera.property = ["SPD"]}, {tessera.property = ["SPD"]}]>
  // LOWER-LABEL: llvm.func @assumed
  // LOWER: llvm.call @mat_assembly_end
  // LOWER-NOT: __tessera_fact_1
  // LOWER: llvm.call @KSPSetOperators
  // LOWER-NOT: __tessera_fact_1
  llvm.func @assumed(%ksp: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @mat_assembly_end(%a0, %c1) : (!llvm.ptr, i32) -> i32
    llvm.call @__tessera_fact_1(%slot) : (!llvm.ptr) -> ()
    %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e2 = tessera.call @petsc.ksp_set_operators(%ksp, %a2, %a2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e2 : i32
  }

  // An assumed fact, too, holds only until something may change the object.
  // FACT-LABEL: llvm.func @assumed_changed_after
  // FACT: tessera.call @petsc.ksp_set_operators
  // FACT-NOT: tessera.property
  // FACT: llvm.return
  llvm.func @assumed_changed_after(%ksp: !llvm.ptr, %s: f64) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    llvm.call @__tessera_fact_1(%slot) : (!llvm.ptr) -> ()
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @mat_scale(%a0, %s) : (!llvm.ptr, f64) -> i32
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e2 = tessera.call @petsc.ksp_set_operators(%ksp, %a1, %a1) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %e2 : i32
  }
}
