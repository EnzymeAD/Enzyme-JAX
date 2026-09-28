// RUN: enzymexlamlir-opt %s -lift-tessera-annotations -split-input-file -verify-diagnostics | FileCheck %s

// The plugin's [[tessera::guarantees]], [[tessera::assumes]] and
// [[tessera::preserves]] reach the module as annotation strings, with
// parameters already turned into argument positions that leave out any sret.
//
// What is always true goes on the definition, as a `tessera.property` on the
// output guaranteed or the parameter assumed. A preserves is only true given
// its inputs, so it stays a rule on the function, with its inputs renumbered
// to the function's own parameters. clang can list an annotation once per
// redeclaration, so a repeat is dropped.

module {
  llvm.mlir.global private unnamed_addr constant @".str.spd_ret"("tessera_guarantees=SPD:return\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.sym_arg1"("tessera_guarantees=symmetric:arg1\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.op_assemble"("tessera_op=lib.assemble(n, out:val=out)\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.op_add"("tessera_op=lib.add(a:val=in, b:val=in, out:val=out)\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.keep_01"("tessera_preserves=SPD:0,1\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.keep_0"("tessera_preserves=SPD:0\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.assume_0"("tessera_assumes=SPD:arg0\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.file"("file.cpp\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<9 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %zero = llvm.mlir.zero : !llvm.ptr
    %line = llvm.mlir.constant(1 : i32) : i32
    %file = llvm.mlir.addressof @".str.file" : !llvm.ptr
    %spd_ret = llvm.mlir.addressof @".str.spd_ret" : !llvm.ptr
    %sym_arg1 = llvm.mlir.addressof @".str.sym_arg1" : !llvm.ptr
    %op_assemble = llvm.mlir.addressof @".str.op_assemble" : !llvm.ptr
    %op_add = llvm.mlir.addressof @".str.op_add" : !llvm.ptr
    %keep_01 = llvm.mlir.addressof @".str.keep_01" : !llvm.ptr
    %keep_0 = llvm.mlir.addressof @".str.keep_0" : !llvm.ptr
    %assume_0 = llvm.mlir.addressof @".str.assume_0" : !llvm.ptr
    %f_mass = llvm.mlir.addressof @mass : !llvm.ptr
    %f_assemble = llvm.mlir.addressof @assemble : !llvm.ptr
    %f_add = llvm.mlir.addressof @add : !llvm.ptr
    %f_scale = llvm.mlir.addressof @scale : !llvm.ptr
    %f_solve = llvm.mlir.addressof @solve : !llvm.ptr
    %undef = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0a = llvm.insertvalue %f_mass, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0b = llvm.insertvalue %spd_ret, %e0a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0c = llvm.insertvalue %file, %e0b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0d = llvm.insertvalue %line, %e0c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0 = llvm.insertvalue %zero, %e0d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1a = llvm.insertvalue %f_mass, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1b = llvm.insertvalue %spd_ret, %e1a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1c = llvm.insertvalue %file, %e1b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1d = llvm.insertvalue %line, %e1c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1 = llvm.insertvalue %zero, %e1d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2a = llvm.insertvalue %f_assemble, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2b = llvm.insertvalue %op_assemble, %e2a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2c = llvm.insertvalue %file, %e2b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2d = llvm.insertvalue %line, %e2c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2 = llvm.insertvalue %zero, %e2d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3a = llvm.insertvalue %f_assemble, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3b = llvm.insertvalue %sym_arg1, %e3a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3c = llvm.insertvalue %file, %e3b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3d = llvm.insertvalue %line, %e3c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3 = llvm.insertvalue %zero, %e3d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e4a = llvm.insertvalue %f_assemble, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e4b = llvm.insertvalue %spd_ret, %e4a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e4c = llvm.insertvalue %file, %e4b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e4d = llvm.insertvalue %line, %e4c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e4 = llvm.insertvalue %zero, %e4d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e5a = llvm.insertvalue %f_add, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e5b = llvm.insertvalue %op_add, %e5a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e5c = llvm.insertvalue %file, %e5b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e5d = llvm.insertvalue %line, %e5c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e5 = llvm.insertvalue %zero, %e5d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e6a = llvm.insertvalue %f_add, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e6b = llvm.insertvalue %keep_01, %e6a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e6c = llvm.insertvalue %file, %e6b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e6d = llvm.insertvalue %line, %e6c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e6 = llvm.insertvalue %zero, %e6d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e7a = llvm.insertvalue %f_scale, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e7b = llvm.insertvalue %keep_0, %e7a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e7c = llvm.insertvalue %file, %e7b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e7d = llvm.insertvalue %line, %e7c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e7 = llvm.insertvalue %zero, %e7d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e8a = llvm.insertvalue %f_solve, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e8b = llvm.insertvalue %assume_0, %e8a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e8c = llvm.insertvalue %file, %e8b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e8d = llvm.insertvalue %line, %e8c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e8 = llvm.insertvalue %zero, %e8d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t = llvm.mlir.undef : !llvm.array<9 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %t0 = llvm.insertvalue %e0, %t[0] : !llvm.array<9 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %t1 = llvm.insertvalue %e1, %t0[1] : !llvm.array<9 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %t2 = llvm.insertvalue %e2, %t1[2] : !llvm.array<9 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %t3 = llvm.insertvalue %e3, %t2[3] : !llvm.array<9 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %t4 = llvm.insertvalue %e4, %t3[4] : !llvm.array<9 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %t5 = llvm.insertvalue %e5, %t4[5] : !llvm.array<9 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %t6 = llvm.insertvalue %e6, %t5[6] : !llvm.array<9 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %t7 = llvm.insertvalue %e7, %t6[7] : !llvm.array<9 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %t8 = llvm.insertvalue %e8, %t7[8] : !llvm.array<9 x struct<(ptr, ptr, ptr, i32, ptr)>>
    llvm.return %t8 : !llvm.array<9 x struct<(ptr, ptr, ptr, i32, ptr)>>
  }

  // A matrix returned by value comes back through an sret parameter, so that
  // is where the guarantee on the return value goes.
  // CHECK: llvm.func @mass(%arg0: !llvm.ptr {llvm.sret = !llvm.struct<"Mat3", (array<9 x f64>)>, tessera.property = ["SPD"]}, %arg1: i64)
  llvm.func @mass(%ret: !llvm.ptr {llvm.sret = !llvm.struct<"Mat3", (array<9 x f64>)>}, %n: i64) attributes {no_inline} {
    llvm.return
  }

  // A written argument and a returned value each get their own.
  // CHECK: llvm.func @assemble(%arg0: i64, %arg1: !llvm.ptr {tessera.property = ["symmetric"]}) -> (i64 {tessera.property = ["SPD"]})
  llvm.func @assemble(%n: i64, %out: !llvm.ptr) -> i64 attributes {no_inline} {
    llvm.return %n : i64
  }

  // CHECK: llvm.func @add(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr)
  // CHECK-SAME: tessera.preserves = [{inputs = [0, 1], property = "SPD"}]
  llvm.func @add(%a: !llvm.ptr, %b: !llvm.ptr, %out: !llvm.ptr) attributes {no_inline} {
    llvm.return
  }

  // Behind an sret, the plugin's position 0 is parameter 1.
  // CHECK: llvm.func @scale(
  // CHECK-SAME: tessera.preserves = [{inputs = [1], property = "SPD"}]
  llvm.func @scale(%ret: !llvm.ptr {llvm.sret = !llvm.struct<"Mat3", (array<9 x f64>)>}, %a: !llvm.ptr, %s: f64) attributes {no_inline} {
    llvm.return
  }

  // An assumption is about the parameter on entry.
  // CHECK: llvm.func @solve(%arg0: !llvm.ptr {tessera.property = ["SPD"]}, %arg1: !llvm.ptr)
  llvm.func @solve(%A: !llvm.ptr, %b: !llvm.ptr) -> f64 attributes {no_inline} {
    %c = llvm.mlir.constant(0.0 : f64) : f64
    llvm.return %c : f64
  }
}

// -----

module {
  llvm.mlir.global private unnamed_addr constant @".str.sym_arg0"("tessera_guarantees=symmetric:arg0\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.op_reads"("tessera_op=lib.reads(A:val=in)\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.file"("file.cpp\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %zero = llvm.mlir.zero : !llvm.ptr
    %line = llvm.mlir.constant(1 : i32) : i32
    %file = llvm.mlir.addressof @".str.file" : !llvm.ptr
    %sym_arg0 = llvm.mlir.addressof @".str.sym_arg0" : !llvm.ptr
    %op_reads = llvm.mlir.addressof @".str.op_reads" : !llvm.ptr
    %f_reads_only = llvm.mlir.addressof @reads_only : !llvm.ptr
    %undef = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0a = llvm.insertvalue %f_reads_only, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0b = llvm.insertvalue %sym_arg0, %e0a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0c = llvm.insertvalue %file, %e0b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0d = llvm.insertvalue %line, %e0c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0 = llvm.insertvalue %zero, %e0d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1a = llvm.insertvalue %f_reads_only, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1b = llvm.insertvalue %op_reads, %e1a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1c = llvm.insertvalue %file, %e1b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1d = llvm.insertvalue %line, %e1c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1 = llvm.insertvalue %zero, %e1d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t = llvm.mlir.undef : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %t0 = llvm.insertvalue %e0, %t[0] : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %t1 = llvm.insertvalue %e1, %t0[1] : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>>
    llvm.return %t1 : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>>
  }

  // A property on a parameter of a definition is read as holding on entry
  // unless the function is known to write it, so a guarantee on a matrix the
  // tessera_op only reads would be misread as an assumption. It is left out
  // with a warning, and the compile goes on without it.
  // CHECK-LABEL: llvm.func @reads_only
  // CHECK-NOT: tessera.property
  // CHECK-NOT: tessera.establishes
  // expected-warning @below {{ignoring tessera guarantees annotation 'symmetric:arg0': argument 0 is val=in, so the function only reads the value behind it}}
  llvm.func @reads_only(%A: !llvm.ptr) attributes {no_inline} {
    llvm.return
  }
}

// -----

// A pointer a function takes as it is, with no tessera_op lifting the value
// behind it, is a handle to an object the function may change in place, like
// a PETSc Mat. A guarantee on it is about that object from the call on, so it
// becomes `tessera.establishes`, never a `tessera.property` that would be read
// as true of the pointer. A readonly says a function leaves the object alone,
// and works on a declaration too.
module {
  llvm.mlir.global private unnamed_addr constant @".str.spd_arg0"("tessera_guarantees=SPD:arg0\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.spd_arg1"("tessera_guarantees=SPD:arg1\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.ro_arg0"("tessera_readonly=arg0\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.ro_arg1"("tessera_readonly=arg1\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.file"("file.c\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<4 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %zero = llvm.mlir.zero : !llvm.ptr
    %line = llvm.mlir.constant(1 : i32) : i32
    %file = llvm.mlir.addressof @".str.file" : !llvm.ptr
    %spd_arg0 = llvm.mlir.addressof @".str.spd_arg0" : !llvm.ptr
    %spd_arg1 = llvm.mlir.addressof @".str.spd_arg1" : !llvm.ptr
    %ro_arg0 = llvm.mlir.addressof @".str.ro_arg0" : !llvm.ptr
    %ro_arg1 = llvm.mlir.addressof @".str.ro_arg1" : !llvm.ptr
    %f_fill = llvm.mlir.addressof @fill_coo : !llvm.ptr
    %f_mult = llvm.mlir.addressof @mat_mult : !llvm.ptr
    %f_shift = llvm.mlir.addressof @mat_shift : !llvm.ptr
    %undef = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0a = llvm.insertvalue %f_fill, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0b = llvm.insertvalue %spd_arg0, %e0a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0c = llvm.insertvalue %file, %e0b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0d = llvm.insertvalue %line, %e0c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0 = llvm.insertvalue %zero, %e0d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1a = llvm.insertvalue %f_mult, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1b = llvm.insertvalue %ro_arg0, %e1a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1c = llvm.insertvalue %file, %e1b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1d = llvm.insertvalue %line, %e1c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1 = llvm.insertvalue %zero, %e1d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2a = llvm.insertvalue %f_shift, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2b = llvm.insertvalue %spd_arg1, %e2a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2c = llvm.insertvalue %file, %e2b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2d = llvm.insertvalue %line, %e2c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2 = llvm.insertvalue %zero, %e2d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3a = llvm.insertvalue %f_shift, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3b = llvm.insertvalue %ro_arg1, %e3a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3c = llvm.insertvalue %file, %e3b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3d = llvm.insertvalue %line, %e3c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3 = llvm.insertvalue %zero, %e3d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t = llvm.mlir.undef : !llvm.array<4 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %t0 = llvm.insertvalue %e0, %t[0] : !llvm.array<4 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %t1 = llvm.insertvalue %e1, %t0[1] : !llvm.array<4 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %t2 = llvm.insertvalue %e2, %t1[2] : !llvm.array<4 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %t3 = llvm.insertvalue %e3, %t2[3] : !llvm.array<4 x struct<(ptr, ptr, ptr, i32, ptr)>>
    llvm.return %t3 : !llvm.array<4 x struct<(ptr, ptr, ptr, i32, ptr)>>
  }

  // CHECK: llvm.func @fill_coo(%arg0: !llvm.ptr {tessera.establishes = ["SPD"]}, %arg1: !llvm.ptr) -> i32
  llvm.func @fill_coo(%A: !llvm.ptr, %ctx: !llvm.ptr) -> i32 attributes {no_inline} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %c0 : i32
  }

  // CHECK: llvm.func @mat_mult(!llvm.ptr {tessera.readonly}, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @mat_mult(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32

  // Neither can say anything about a value passed by value.
  // CHECK: llvm.func @mat_shift(!llvm.ptr, f64) -> i32
  // expected-warning @below {{ignoring tessera guarantees annotation 'SPD:arg1': argument 1 is not a pointer, so the function cannot give it a property}}
  // expected-warning @below {{ignoring tessera readonly annotation 'arg1': argument 1 is not a pointer, so there is no object to leave alone}}
  llvm.func @mat_shift(!llvm.ptr, f64) -> i32
}
