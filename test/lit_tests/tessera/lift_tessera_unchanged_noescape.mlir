// RUN: enzymexlamlir-opt %s -lift-tessera-annotations -split-input-file -verify-diagnostics | FileCheck %s

// tessera_unchanged says a call changes no property of an object it is given,
// though it may write it: MatSetOption sets a flag on the matrix, not its
// entries. tessera_keeps and tessera_readonly, older spellings, are read the
// same way. tessera_noescape, which the frontend adds for a parameter marked
// `__attribute__((noescape))`, says the call holds on to no pointer to the
// object once it returns, as MatMult doesn't. Each is a unit attribute on the
// parameter, and one that can't be placed is dropped with a warning, which
// only loses facts.
module {
  llvm.mlir.global private unnamed_addr constant @".str.unchanged"("tessera_unchanged=A\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.noescape"("tessera_noescape=A,x\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.bad"("tessera_noescape=s\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.params_option"("tessera_params=A,op,flg\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.params_mult"("tessera_params=A,x,y\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.params_scale"("tessera_params=A,s\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.file"("file.c\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<6 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %zero = llvm.mlir.zero : !llvm.ptr
    %line = llvm.mlir.constant(1 : i32) : i32
    %file = llvm.mlir.addressof @".str.file" : !llvm.ptr
    %unchanged = llvm.mlir.addressof @".str.unchanged" : !llvm.ptr
    %noescape = llvm.mlir.addressof @".str.noescape" : !llvm.ptr
    %bad = llvm.mlir.addressof @".str.bad" : !llvm.ptr
    %p_option = llvm.mlir.addressof @".str.params_option" : !llvm.ptr
    %p_mult = llvm.mlir.addressof @".str.params_mult" : !llvm.ptr
    %p_scale = llvm.mlir.addressof @".str.params_scale" : !llvm.ptr
    %f_option = llvm.mlir.addressof @mat_set_option : !llvm.ptr
    %f_mult = llvm.mlir.addressof @mat_mult : !llvm.ptr
    %f_scale = llvm.mlir.addressof @mat_scale : !llvm.ptr
    %undef = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0a = llvm.insertvalue %f_option, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0b = llvm.insertvalue %unchanged, %e0a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0c = llvm.insertvalue %file, %e0b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0d = llvm.insertvalue %line, %e0c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0 = llvm.insertvalue %zero, %e0d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1a = llvm.insertvalue %f_option, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1b = llvm.insertvalue %p_option, %e1a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1c = llvm.insertvalue %file, %e1b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1d = llvm.insertvalue %line, %e1c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1 = llvm.insertvalue %zero, %e1d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2a = llvm.insertvalue %f_mult, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2b = llvm.insertvalue %noescape, %e2a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2c = llvm.insertvalue %file, %e2b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2d = llvm.insertvalue %line, %e2c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2 = llvm.insertvalue %zero, %e2d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3a = llvm.insertvalue %f_mult, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3b = llvm.insertvalue %p_mult, %e3a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3c = llvm.insertvalue %file, %e3b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3d = llvm.insertvalue %line, %e3c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3 = llvm.insertvalue %zero, %e3d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e4a = llvm.insertvalue %f_scale, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e4b = llvm.insertvalue %bad, %e4a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e4c = llvm.insertvalue %file, %e4b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e4d = llvm.insertvalue %line, %e4c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e4 = llvm.insertvalue %zero, %e4d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e5a = llvm.insertvalue %f_scale, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e5b = llvm.insertvalue %p_scale, %e5a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e5c = llvm.insertvalue %file, %e5b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e5d = llvm.insertvalue %line, %e5c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e5 = llvm.insertvalue %zero, %e5d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t = llvm.mlir.undef : !llvm.array<6 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %t0 = llvm.insertvalue %e0, %t[0] : !llvm.array<6 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %t1 = llvm.insertvalue %e1, %t0[1] : !llvm.array<6 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %t2 = llvm.insertvalue %e2, %t1[2] : !llvm.array<6 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %t3 = llvm.insertvalue %e3, %t2[3] : !llvm.array<6 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %t4 = llvm.insertvalue %e4, %t3[4] : !llvm.array<6 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %t5 = llvm.insertvalue %e5, %t4[5] : !llvm.array<6 x struct<(ptr, ptr, ptr, i32, ptr)>>
    llvm.return %t5 : !llvm.array<6 x struct<(ptr, ptr, ptr, i32, ptr)>>
  }

  // CHECK: llvm.func @mat_set_option(!llvm.ptr {tessera.unchanged}, i32, i32) -> i32
  llvm.func @mat_set_option(!llvm.ptr, i32, i32) -> i32

  // CHECK: llvm.func @mat_mult(!llvm.ptr {tessera.noescape}, !llvm.ptr {tessera.noescape}, !llvm.ptr) -> i32
  llvm.func @mat_mult(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32

  // CHECK: llvm.func @mat_scale(!llvm.ptr, f64) -> i32
  // CHECK-NOT: attributes
  // expected-warning @below {{file.c:1: ignoring tessera noescape 's': 's' is not a pointer, so there is no object to hold on to}}
  llvm.func @mat_scale(!llvm.ptr, f64) -> i32
}

// -----

// C's noescape can't be written on `this`, so `tessera_noescape("this")` on a
// member function says it instead, as on MFEM's BilinearForm(fes), which keeps
// the space but nothing keeps the form. `this` is the first argument.
module {
  llvm.mlir.global private unnamed_addr constant @".str.noescape"("tessera_noescape=this\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.params"("tessera_params=this,f\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.file"("file.cpp\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %zero = llvm.mlir.zero : !llvm.ptr
    %line = llvm.mlir.constant(1 : i32) : i32
    %file = llvm.mlir.addressof @".str.file" : !llvm.ptr
    %noescape = llvm.mlir.addressof @".str.noescape" : !llvm.ptr
    %params = llvm.mlir.addressof @".str.params" : !llvm.ptr
    %f = llvm.mlir.addressof @_ZN4mfem12BilinearFormC1EPNS_18FiniteElementSpaceE : !llvm.ptr
    %undef = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0a = llvm.insertvalue %f, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0b = llvm.insertvalue %noescape, %e0a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0c = llvm.insertvalue %file, %e0b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0d = llvm.insertvalue %line, %e0c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0 = llvm.insertvalue %zero, %e0d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1a = llvm.insertvalue %f, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1b = llvm.insertvalue %params, %e1a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1c = llvm.insertvalue %file, %e1b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1d = llvm.insertvalue %line, %e1c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1 = llvm.insertvalue %zero, %e1d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t = llvm.mlir.undef : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %t0 = llvm.insertvalue %e0, %t[0] : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %t1 = llvm.insertvalue %e1, %t0[1] : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>>
    llvm.return %t1 : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>>
  }

  // CHECK: llvm.func @_ZN4mfem12BilinearFormC1EPNS_18FiniteElementSpaceE(!llvm.ptr {tessera.noescape}, !llvm.ptr)
  llvm.func @_ZN4mfem12BilinearFormC1EPNS_18FiniteElementSpaceE(!llvm.ptr, !llvm.ptr)
}
