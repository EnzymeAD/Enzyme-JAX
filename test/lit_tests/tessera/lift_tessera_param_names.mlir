// RUN: enzymexlamlir-opt %s -lift-tessera-annotations -split-input-file -verify-diagnostics | FileCheck %s

// The plugin passes a tessera attribute's strings on as they were written,
// with the names of the function's argument positions, `this` first for a
// member function, as tessera_params. Everything that reads them is here.

// A fact, a readonly or a preserves can name a parameter as the source does.
// One attribute can state several facts; each becomes its own property.
module {
  llvm.mlir.global private unnamed_addr constant @".str.0"("tessera_op=mfem.mult(this:val=in, x, y:val=out)\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.1"("tessera_params=this,x,y\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.2"("tessera_guarantees=SPD(y), symmetric(y)\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.3"("tessera_readonly=x\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.4"("tessera_preserves=SPD,this\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.5"("tessera_params=n\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.6"("tessera_guarantees=SPD(return)\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.file"("file.cpp\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<7 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %zero = llvm.mlir.zero : !llvm.ptr
    %file = llvm.mlir.addressof @".str.file" : !llvm.ptr
    %undef = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t0 = llvm.mlir.undef : !llvm.array<7 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f0 = llvm.mlir.addressof @_ZNK4mfem3Mat4MultERKNS_3VecERS1_ : !llvm.ptr
    %s0 = llvm.mlir.addressof @".str.0" : !llvm.ptr
    %l0 = llvm.mlir.constant(12 : i32) : i32
    %e0a = llvm.insertvalue %f0, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0b = llvm.insertvalue %s0, %e0a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0c = llvm.insertvalue %file, %e0b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0d = llvm.insertvalue %l0, %e0c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0 = llvm.insertvalue %zero, %e0d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t1 = llvm.insertvalue %e0, %t0[0] : !llvm.array<7 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f1 = llvm.mlir.addressof @_ZNK4mfem3Mat4MultERKNS_3VecERS1_ : !llvm.ptr
    %s1 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %l1 = llvm.mlir.constant(12 : i32) : i32
    %e1a = llvm.insertvalue %f1, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1b = llvm.insertvalue %s1, %e1a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1c = llvm.insertvalue %file, %e1b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1d = llvm.insertvalue %l1, %e1c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1 = llvm.insertvalue %zero, %e1d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t2 = llvm.insertvalue %e1, %t1[1] : !llvm.array<7 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f2 = llvm.mlir.addressof @_ZNK4mfem3Mat4MultERKNS_3VecERS1_ : !llvm.ptr
    %s2 = llvm.mlir.addressof @".str.2" : !llvm.ptr
    %l2 = llvm.mlir.constant(12 : i32) : i32
    %e2a = llvm.insertvalue %f2, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2b = llvm.insertvalue %s2, %e2a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2c = llvm.insertvalue %file, %e2b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2d = llvm.insertvalue %l2, %e2c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2 = llvm.insertvalue %zero, %e2d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t3 = llvm.insertvalue %e2, %t2[2] : !llvm.array<7 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f3 = llvm.mlir.addressof @_ZNK4mfem3Mat4MultERKNS_3VecERS1_ : !llvm.ptr
    %s3 = llvm.mlir.addressof @".str.3" : !llvm.ptr
    %l3 = llvm.mlir.constant(12 : i32) : i32
    %e3a = llvm.insertvalue %f3, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3b = llvm.insertvalue %s3, %e3a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3c = llvm.insertvalue %file, %e3b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3d = llvm.insertvalue %l3, %e3c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3 = llvm.insertvalue %zero, %e3d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t4 = llvm.insertvalue %e3, %t3[3] : !llvm.array<7 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f4 = llvm.mlir.addressof @_ZNK4mfem3Mat4MultERKNS_3VecERS1_ : !llvm.ptr
    %s4 = llvm.mlir.addressof @".str.4" : !llvm.ptr
    %l4 = llvm.mlir.constant(12 : i32) : i32
    %e4a = llvm.insertvalue %f4, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e4b = llvm.insertvalue %s4, %e4a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e4c = llvm.insertvalue %file, %e4b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e4d = llvm.insertvalue %l4, %e4c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e4 = llvm.insertvalue %zero, %e4d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t5 = llvm.insertvalue %e4, %t4[4] : !llvm.array<7 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f5 = llvm.mlir.addressof @_Z4massl : !llvm.ptr
    %s5 = llvm.mlir.addressof @".str.5" : !llvm.ptr
    %l5 = llvm.mlir.constant(20 : i32) : i32
    %e5a = llvm.insertvalue %f5, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e5b = llvm.insertvalue %s5, %e5a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e5c = llvm.insertvalue %file, %e5b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e5d = llvm.insertvalue %l5, %e5c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e5 = llvm.insertvalue %zero, %e5d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t6 = llvm.insertvalue %e5, %t5[5] : !llvm.array<7 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f6 = llvm.mlir.addressof @_Z4massl : !llvm.ptr
    %s6 = llvm.mlir.addressof @".str.6" : !llvm.ptr
    %l6 = llvm.mlir.constant(20 : i32) : i32
    %e6a = llvm.insertvalue %f6, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e6b = llvm.insertvalue %s6, %e6a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e6c = llvm.insertvalue %file, %e6b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e6d = llvm.insertvalue %l6, %e6c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e6 = llvm.insertvalue %zero, %e6d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t7 = llvm.insertvalue %e6, %t6[6] : !llvm.array<7 x struct<(ptr, ptr, ptr, i32, ptr)>>
    llvm.return %t7 : !llvm.array<7 x struct<(ptr, ptr, ptr, i32, ptr)>>
  }

  // CHECK: llvm.func @_ZNK4mfem3Mat4MultERKNS_3VecERS1_(%arg0: !llvm.ptr, %arg1: !llvm.ptr {tessera.readonly}, %arg2: !llvm.ptr {tessera.guarantees = ["SPD", "symmetric"]})
  // CHECK-SAME: tessera.preserves = [{inputs = [0], property = "SPD"}]
  llvm.func @_ZNK4mfem3Mat4MultERKNS_3VecERS1_(%this: !llvm.ptr, %x: !llvm.ptr, %y: !llvm.ptr) attributes {no_inline} {
    llvm.return
  }

  // A matrix returned by value comes back through an sret parameter, which
  // the names leave out as positions do.
  // CHECK: llvm.func @_Z4massl(%arg0: !llvm.ptr {llvm.sret = !llvm.struct<"Mat3", (array<9 x f64>)>, tessera.guarantees = ["SPD"]}, %arg1: i64)
  llvm.func @_Z4massl(%ret: !llvm.ptr {llvm.sret = !llvm.struct<"Mat3", (array<9 x f64>)>}, %n: i64) attributes {no_inline} {
    llvm.return
  }
}

// -----

// What the source got wrong is reported here, with where it was written, and
// only that piece is left out.
module {
  llvm.mlir.global private unnamed_addr constant @".str.0"("tessera_params=A,B\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.1"("tessera_guarantees=SPD(A),B,SPD(C),symmetric(B)\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.2"("tessera_readonly=A,B\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.3"("tessera_assumes=SPD\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.4"("tessera_preserves=SPD,B,Z\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.5"("tessera_params=B\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.6"("tessera_guarantees=SPD\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.file"("file.cpp\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<7 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %zero = llvm.mlir.zero : !llvm.ptr
    %file = llvm.mlir.addressof @".str.file" : !llvm.ptr
    %undef = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t0 = llvm.mlir.undef : !llvm.array<7 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f0 = llvm.mlir.addressof @shift : !llvm.ptr
    %s0 = llvm.mlir.addressof @".str.0" : !llvm.ptr
    %l0 = llvm.mlir.constant(30 : i32) : i32
    %e0a = llvm.insertvalue %f0, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0b = llvm.insertvalue %s0, %e0a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0c = llvm.insertvalue %file, %e0b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0d = llvm.insertvalue %l0, %e0c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0 = llvm.insertvalue %zero, %e0d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t1 = llvm.insertvalue %e0, %t0[0] : !llvm.array<7 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f1 = llvm.mlir.addressof @shift : !llvm.ptr
    %s1 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %l1 = llvm.mlir.constant(30 : i32) : i32
    %e1a = llvm.insertvalue %f1, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1b = llvm.insertvalue %s1, %e1a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1c = llvm.insertvalue %file, %e1b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1d = llvm.insertvalue %l1, %e1c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1 = llvm.insertvalue %zero, %e1d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t2 = llvm.insertvalue %e1, %t1[1] : !llvm.array<7 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f2 = llvm.mlir.addressof @shift : !llvm.ptr
    %s2 = llvm.mlir.addressof @".str.2" : !llvm.ptr
    %l2 = llvm.mlir.constant(30 : i32) : i32
    %e2a = llvm.insertvalue %f2, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2b = llvm.insertvalue %s2, %e2a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2c = llvm.insertvalue %file, %e2b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2d = llvm.insertvalue %l2, %e2c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2 = llvm.insertvalue %zero, %e2d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t3 = llvm.insertvalue %e2, %t2[2] : !llvm.array<7 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f3 = llvm.mlir.addressof @shift : !llvm.ptr
    %s3 = llvm.mlir.addressof @".str.3" : !llvm.ptr
    %l3 = llvm.mlir.constant(30 : i32) : i32
    %e3a = llvm.insertvalue %f3, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3b = llvm.insertvalue %s3, %e3a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3c = llvm.insertvalue %file, %e3b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3d = llvm.insertvalue %l3, %e3c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3 = llvm.insertvalue %zero, %e3d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t4 = llvm.insertvalue %e3, %t3[3] : !llvm.array<7 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f4 = llvm.mlir.addressof @shift : !llvm.ptr
    %s4 = llvm.mlir.addressof @".str.4" : !llvm.ptr
    %l4 = llvm.mlir.constant(30 : i32) : i32
    %e4a = llvm.insertvalue %f4, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e4b = llvm.insertvalue %s4, %e4a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e4c = llvm.insertvalue %file, %e4b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e4d = llvm.insertvalue %l4, %e4c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e4 = llvm.insertvalue %zero, %e4d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t5 = llvm.insertvalue %e4, %t4[4] : !llvm.array<7 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f5 = llvm.mlir.addressof @fill : !llvm.ptr
    %s5 = llvm.mlir.addressof @".str.5" : !llvm.ptr
    %l5 = llvm.mlir.constant(40 : i32) : i32
    %e5a = llvm.insertvalue %f5, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e5b = llvm.insertvalue %s5, %e5a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e5c = llvm.insertvalue %file, %e5b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e5d = llvm.insertvalue %l5, %e5c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e5 = llvm.insertvalue %zero, %e5d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t6 = llvm.insertvalue %e5, %t5[5] : !llvm.array<7 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f6 = llvm.mlir.addressof @fill : !llvm.ptr
    %s6 = llvm.mlir.addressof @".str.6" : !llvm.ptr
    %l6 = llvm.mlir.constant(40 : i32) : i32
    %e6a = llvm.insertvalue %f6, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e6b = llvm.insertvalue %s6, %e6a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e6c = llvm.insertvalue %file, %e6b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e6d = llvm.insertvalue %l6, %e6c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e6 = llvm.insertvalue %zero, %e6d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t7 = llvm.insertvalue %e6, %t6[6] : !llvm.array<7 x struct<(ptr, ptr, ptr, i32, ptr)>>
    llvm.return %t7 : !llvm.array<7 x struct<(ptr, ptr, ptr, i32, ptr)>>
  }

  // A struct passed by value arrives as a pointer to the caller's copy, so
  // neither a guarantee nor a readonly can say anything about it.
  // CHECK: llvm.func @shift(%arg0: !llvm.ptr {llvm.byval = !llvm.struct<(f64)>}, %arg1: !llvm.ptr {tessera.establishes = ["symmetric"], tessera.readonly}) -> i32 attributes {no_inline} {
  // expected-warning @below {{file.cpp:30: ignoring tessera guarantees 'SPD(A)': 'A' is passed by value, so the function cannot give it a property}}
  // expected-warning @below {{file.cpp:30: ignoring tessera guarantees 'B': 'B' names a parameter, not a property; state a property of it as "SPD(B)"}}
  // expected-warning @below {{file.cpp:30: ignoring tessera guarantees 'SPD(C)': 'C' is not a parameter of the function}}
  // expected-warning @below {{file.cpp:30: ignoring tessera readonly 'A': 'A' is passed by value, so there is no object to leave alone}}
  // expected-warning @below {{file.cpp:30: ignoring tessera assumes 'SPD': an assumption needs the parameter it applies to; write "SPD(M)"}}
  // expected-warning @below {{file.cpp:30: ignoring tessera preserves 'SPD,B,Z': 'Z' is not a parameter of the function}}
  llvm.func @shift(%A: !llvm.ptr {llvm.byval = !llvm.struct<(f64)>}, %B: !llvm.ptr) -> i32 attributes {no_inline} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %c0 : i32
  }

  // CHECK: llvm.func @fill(%arg0: !llvm.ptr) attributes {no_inline} {
  // expected-warning @below {{file.cpp:40: ignoring tessera guarantees 'SPD': the function returns nothing; name the parameter the fact applies to, as "SPD(M)"}}
  llvm.func @fill(%B: !llvm.ptr) attributes {no_inline} {
    llvm.return
  }
}

// -----

// A fact stated on a statement arrives on a marker whose parameters are the
// variables the plugin found named in it, under their own names.
module {
  llvm.mlir.global private unnamed_addr constant @".str.0"("tessera_fact\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.1"("tessera_params=A\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.2"("tessera_guarantees=SPD(A),SPD(B),SPD\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.3"("tessera_assumes=symmetric(A)\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.file"("ex2.c\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<4 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %zero = llvm.mlir.zero : !llvm.ptr
    %file = llvm.mlir.addressof @".str.file" : !llvm.ptr
    %undef = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t0 = llvm.mlir.undef : !llvm.array<4 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f0 = llvm.mlir.addressof @__tessera_fact_0 : !llvm.ptr
    %s0 = llvm.mlir.addressof @".str.0" : !llvm.ptr
    %l0 = llvm.mlir.constant(50 : i32) : i32
    %e0a = llvm.insertvalue %f0, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0b = llvm.insertvalue %s0, %e0a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0c = llvm.insertvalue %file, %e0b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0d = llvm.insertvalue %l0, %e0c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0 = llvm.insertvalue %zero, %e0d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t1 = llvm.insertvalue %e0, %t0[0] : !llvm.array<4 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f1 = llvm.mlir.addressof @__tessera_fact_0 : !llvm.ptr
    %s1 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %l1 = llvm.mlir.constant(50 : i32) : i32
    %e1a = llvm.insertvalue %f1, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1b = llvm.insertvalue %s1, %e1a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1c = llvm.insertvalue %file, %e1b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1d = llvm.insertvalue %l1, %e1c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1 = llvm.insertvalue %zero, %e1d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t2 = llvm.insertvalue %e1, %t1[1] : !llvm.array<4 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f2 = llvm.mlir.addressof @__tessera_fact_0 : !llvm.ptr
    %s2 = llvm.mlir.addressof @".str.2" : !llvm.ptr
    %l2 = llvm.mlir.constant(50 : i32) : i32
    %e2a = llvm.insertvalue %f2, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2b = llvm.insertvalue %s2, %e2a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2c = llvm.insertvalue %file, %e2b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2d = llvm.insertvalue %l2, %e2c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2 = llvm.insertvalue %zero, %e2d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t3 = llvm.insertvalue %e2, %t2[2] : !llvm.array<4 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f3 = llvm.mlir.addressof @__tessera_fact_0 : !llvm.ptr
    %s3 = llvm.mlir.addressof @".str.3" : !llvm.ptr
    %l3 = llvm.mlir.constant(50 : i32) : i32
    %e3a = llvm.insertvalue %f3, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3b = llvm.insertvalue %s3, %e3a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3c = llvm.insertvalue %file, %e3b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3d = llvm.insertvalue %l3, %e3c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3 = llvm.insertvalue %zero, %e3d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t4 = llvm.insertvalue %e3, %t3[3] : !llvm.array<4 x struct<(ptr, ptr, ptr, i32, ptr)>>
    llvm.return %t4 : !llvm.array<4 x struct<(ptr, ptr, ptr, i32, ptr)>>
  }

  // CHECK: llvm.func weak @__tessera_fact_0(%arg0: !llvm.ptr {tessera.assumes = ["symmetric"], tessera.establishes = ["SPD"]}) attributes {{{.*}}tessera.fact_marker{{.*}}} {
  // expected-warning @below {{ex2.c:50: ignoring tessera guarantees 'SPD(B)' stated on a statement: 'B' is not a pointer or object variable in scope there}}
  // expected-warning @below {{ex2.c:50: ignoring tessera guarantees 'SPD' stated on a statement: a fact stated on a statement names a variable, as "SPD(A)" does}}
  llvm.func weak @__tessera_fact_0(%a: !llvm.ptr) attributes {no_inline} {
    llvm.return
  }
}

// -----

// The type of the value behind each argument a tessera op may lift comes in a
// global the plugin made for it. It goes on the function as tessera.arg_types,
// and the global, which was kept only by llvm.compiler.used, goes.
module {
  llvm.mlir.global internal constant @_ZL20__tessera_arg_type_0() {addr_space = 0 : i32} : !llvm.struct<"Mat", (array<4 x f64>)> {
    %0 = llvm.mlir.zero : !llvm.struct<"Mat", (array<4 x f64>)>
    llvm.return %0 : !llvm.struct<"Mat", (array<4 x f64>)>
  }
  llvm.mlir.global internal constant @__tessera_arg_type_1(0.0 : f64) {addr_space = 0 : i32} : f64
  llvm.mlir.global internal @kept(0 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global private unnamed_addr constant @".str.0"("pure_tessera_op=eigen.inv(x:val=in)\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.1"("tessera_arg_types=_ZL20__tessera_arg_type_0\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.2"("tessera_op=lib.scale(n, p:val=in, q)\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.3"("tessera_arg_types=,__tessera_arg_type_1,__tessera_arg_type_9\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.file"("file.cpp\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<4 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %zero = llvm.mlir.zero : !llvm.ptr
    %file = llvm.mlir.addressof @".str.file" : !llvm.ptr
    %undef = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t0 = llvm.mlir.undef : !llvm.array<4 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f0 = llvm.mlir.addressof @_Z3inv3Mat : !llvm.ptr
    %s0 = llvm.mlir.addressof @".str.0" : !llvm.ptr
    %l0 = llvm.mlir.constant(60 : i32) : i32
    %e0a = llvm.insertvalue %f0, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0b = llvm.insertvalue %s0, %e0a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0c = llvm.insertvalue %file, %e0b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0d = llvm.insertvalue %l0, %e0c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0 = llvm.insertvalue %zero, %e0d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t1 = llvm.insertvalue %e0, %t0[0] : !llvm.array<4 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f1 = llvm.mlir.addressof @_Z3inv3Mat : !llvm.ptr
    %s1 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %l1 = llvm.mlir.constant(60 : i32) : i32
    %e1a = llvm.insertvalue %f1, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1b = llvm.insertvalue %s1, %e1a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1c = llvm.insertvalue %file, %e1b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1d = llvm.insertvalue %l1, %e1c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1 = llvm.insertvalue %zero, %e1d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t2 = llvm.insertvalue %e1, %t1[1] : !llvm.array<4 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f2 = llvm.mlir.addressof @scale : !llvm.ptr
    %s2 = llvm.mlir.addressof @".str.2" : !llvm.ptr
    %l2 = llvm.mlir.constant(70 : i32) : i32
    %e2a = llvm.insertvalue %f2, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2b = llvm.insertvalue %s2, %e2a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2c = llvm.insertvalue %file, %e2b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2d = llvm.insertvalue %l2, %e2c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2 = llvm.insertvalue %zero, %e2d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t3 = llvm.insertvalue %e2, %t2[2] : !llvm.array<4 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f3 = llvm.mlir.addressof @scale : !llvm.ptr
    %s3 = llvm.mlir.addressof @".str.3" : !llvm.ptr
    %l3 = llvm.mlir.constant(70 : i32) : i32
    %e3a = llvm.insertvalue %f3, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3b = llvm.insertvalue %s3, %e3a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3c = llvm.insertvalue %file, %e3b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3d = llvm.insertvalue %l3, %e3c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3 = llvm.insertvalue %zero, %e3d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t4 = llvm.insertvalue %e3, %t3[3] : !llvm.array<4 x struct<(ptr, ptr, ptr, i32, ptr)>>
    llvm.return %t4 : !llvm.array<4 x struct<(ptr, ptr, ptr, i32, ptr)>>
  }
  llvm.mlir.global appending @llvm.compiler.used() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<2 x ptr> {
    %0 = llvm.mlir.addressof @_ZL20__tessera_arg_type_0 : !llvm.ptr
    %1 = llvm.mlir.addressof @kept : !llvm.ptr
    %2 = llvm.mlir.undef : !llvm.array<2 x ptr>
    %3 = llvm.insertvalue %0, %2[0] : !llvm.array<2 x ptr>
    %4 = llvm.insertvalue %1, %3[1] : !llvm.array<2 x ptr>
    llvm.return %4 : !llvm.array<2 x ptr>
  }

  // CHECK-NOT: llvm.mlir.global internal constant @{{.*}}__tessera_arg_type
  // CHECK: llvm.mlir.global appending @llvm.compiler.used() {{.*}} : !llvm.array<1 x ptr> {
  // CHECK-NEXT: llvm.mlir.undef : !llvm.array<1 x ptr>
  // CHECK-NEXT: llvm.mlir.addressof @kept
  // CHECK-NEXT: llvm.insertvalue
  // CHECK-NEXT: llvm.return

  // CHECK: llvm.func @_Z3inv3Mat({{.*}}) attributes {pure_tessera_op = "eigen.inv(x:val=in)", tessera.arg_types = [!llvm.struct<"Mat", (array<4 x f64>)>]}
  llvm.func @_Z3inv3Mat(%ret: !llvm.ptr {llvm.sret = !llvm.struct<"Mat", (array<4 x f64>)>}, %x: !llvm.ptr) {
    llvm.return
  }

  // A position given no type, or a global that is not there, gets none.
  // CHECK: llvm.func @scale({{.*}}) attributes {tessera.arg_types = [unit, f64, unit], tessera_op = "lib.scale(n, p:val=in, q)"}
  llvm.func @scale(%n: i64, %p: !llvm.ptr, %q: !llvm.ptr) {
    llvm.return
  }
}

// -----

// A rule may use the C names of the program it is in, which the plugin gives
// the values of: a constant's value, a string macro's string, or a global's
// symbol. A name the left-hand side binds, or that is part of a call, keeps
// its own meaning.
module {
  llvm.mlir.global internal @__tessera_optimize_rule_0(0 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global internal @__tessera_names(0 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global private unnamed_addr constant @".str.0"("tessera_optimize=if n > LIMIT, lib.f(n, x) -> lib.g(x, MODE, mfem::IntRules, ::mfem::IntRules); lib.f(n, x)\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.1"("tessera_name=LIMIT=64\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.2"("tessera_name=MODE='cg'\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.3"("tessera_name=mfem::IntRules=@_ZN4mfem8IntRulesE\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.4"("tessera_name=::mfem::IntRules=@_ZN4mfem8IntRulesE\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.5"("tessera_name=x=99\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.6"("tessera_name=f=7\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.file"("file.cpp\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<7 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %zero = llvm.mlir.zero : !llvm.ptr
    %file = llvm.mlir.addressof @".str.file" : !llvm.ptr
    %undef = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t0 = llvm.mlir.undef : !llvm.array<7 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f0 = llvm.mlir.addressof @__tessera_optimize_rule_0 : !llvm.ptr
    %s0 = llvm.mlir.addressof @".str.0" : !llvm.ptr
    %l0 = llvm.mlir.constant(0 : i32) : i32
    %e0a = llvm.insertvalue %f0, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0b = llvm.insertvalue %s0, %e0a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0c = llvm.insertvalue %file, %e0b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0d = llvm.insertvalue %l0, %e0c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0 = llvm.insertvalue %zero, %e0d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t1 = llvm.insertvalue %e0, %t0[0] : !llvm.array<7 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f1 = llvm.mlir.addressof @__tessera_names : !llvm.ptr
    %s1 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %l1 = llvm.mlir.constant(0 : i32) : i32
    %e1a = llvm.insertvalue %f1, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1b = llvm.insertvalue %s1, %e1a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1c = llvm.insertvalue %file, %e1b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1d = llvm.insertvalue %l1, %e1c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1 = llvm.insertvalue %zero, %e1d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t2 = llvm.insertvalue %e1, %t1[1] : !llvm.array<7 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f2 = llvm.mlir.addressof @__tessera_names : !llvm.ptr
    %s2 = llvm.mlir.addressof @".str.2" : !llvm.ptr
    %l2 = llvm.mlir.constant(0 : i32) : i32
    %e2a = llvm.insertvalue %f2, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2b = llvm.insertvalue %s2, %e2a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2c = llvm.insertvalue %file, %e2b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2d = llvm.insertvalue %l2, %e2c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2 = llvm.insertvalue %zero, %e2d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t3 = llvm.insertvalue %e2, %t2[2] : !llvm.array<7 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f3 = llvm.mlir.addressof @__tessera_names : !llvm.ptr
    %s3 = llvm.mlir.addressof @".str.3" : !llvm.ptr
    %l3 = llvm.mlir.constant(0 : i32) : i32
    %e3a = llvm.insertvalue %f3, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3b = llvm.insertvalue %s3, %e3a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3c = llvm.insertvalue %file, %e3b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3d = llvm.insertvalue %l3, %e3c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3 = llvm.insertvalue %zero, %e3d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t4 = llvm.insertvalue %e3, %t3[3] : !llvm.array<7 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f4 = llvm.mlir.addressof @__tessera_names : !llvm.ptr
    %s4 = llvm.mlir.addressof @".str.4" : !llvm.ptr
    %l4 = llvm.mlir.constant(0 : i32) : i32
    %e4a = llvm.insertvalue %f4, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e4b = llvm.insertvalue %s4, %e4a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e4c = llvm.insertvalue %file, %e4b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e4d = llvm.insertvalue %l4, %e4c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e4 = llvm.insertvalue %zero, %e4d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t5 = llvm.insertvalue %e4, %t4[4] : !llvm.array<7 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f5 = llvm.mlir.addressof @__tessera_names : !llvm.ptr
    %s5 = llvm.mlir.addressof @".str.5" : !llvm.ptr
    %l5 = llvm.mlir.constant(0 : i32) : i32
    %e5a = llvm.insertvalue %f5, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e5b = llvm.insertvalue %s5, %e5a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e5c = llvm.insertvalue %file, %e5b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e5d = llvm.insertvalue %l5, %e5c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e5 = llvm.insertvalue %zero, %e5d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t6 = llvm.insertvalue %e5, %t5[5] : !llvm.array<7 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f6 = llvm.mlir.addressof @__tessera_names : !llvm.ptr
    %s6 = llvm.mlir.addressof @".str.6" : !llvm.ptr
    %l6 = llvm.mlir.constant(0 : i32) : i32
    %e6a = llvm.insertvalue %f6, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e6b = llvm.insertvalue %s6, %e6a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e6c = llvm.insertvalue %file, %e6b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e6d = llvm.insertvalue %l6, %e6c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e6 = llvm.insertvalue %zero, %e6d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t7 = llvm.insertvalue %e6, %t6[6] : !llvm.array<7 x struct<(ptr, ptr, ptr, i32, ptr)>>
    llvm.return %t7 : !llvm.array<7 x struct<(ptr, ptr, ptr, i32, ptr)>>
  }

  // CHECK: tessera.optimization "if n > 64, lib.f(n, x) -> lib.g(x, 'cg', @_ZN4mfem8IntRulesE, @_ZN4mfem8IntRulesE); lib.f(n, x)"
}
