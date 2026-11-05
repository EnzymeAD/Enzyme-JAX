// RUN: enzymexlamlir-opt %s -lift-tessera-annotations -verify-diagnostics | FileCheck %s

// A guarantee with a condition becomes tessera.conditional_establishes: the
// argument, the property, the condition, and the argument or C value each
// name in it stands for. It must be about a handle parameter, name only
// parameters and C names, and be a guarantee.

module {
  llvm.mlir.global private unnamed_addr constant @".str.0"("tessera_params=A,B,C,trans\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.1"("tessera_guarantees=SPD(C) if FullColumnRank(A) && A == B && trans == Trans\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.2"("tessera_params=A,C\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.3"("tessera_guarantees=SPD(C) if FullColumnRank(Z)\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.4"("tessera_params=A\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.5"("tessera_assumes=SPD(A) if FullColumnRank(A)\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.name"("tessera_name=Trans=1\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.file"("file.c\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<6 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %zero = llvm.mlir.zero : !llvm.ptr
    %file = llvm.mlir.addressof @".str.file" : !llvm.ptr
    %line = llvm.mlir.constant(7 : i32) : i32
    %undef = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t0 = llvm.mlir.undef : !llvm.array<6 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f0 = llvm.mlir.addressof @gram : !llvm.ptr
    %s0 = llvm.mlir.addressof @".str.0" : !llvm.ptr
    %e0a = llvm.insertvalue %f0, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0b = llvm.insertvalue %s0, %e0a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0c = llvm.insertvalue %file, %e0b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0d = llvm.insertvalue %line, %e0c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0 = llvm.insertvalue %zero, %e0d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t1 = llvm.insertvalue %e0, %t0[0] : !llvm.array<6 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f1 = llvm.mlir.addressof @gram : !llvm.ptr
    %s1 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %e1a = llvm.insertvalue %f1, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1b = llvm.insertvalue %s1, %e1a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1c = llvm.insertvalue %file, %e1b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1d = llvm.insertvalue %line, %e1c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1 = llvm.insertvalue %zero, %e1d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t2 = llvm.insertvalue %e1, %t1[1] : !llvm.array<6 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f2 = llvm.mlir.addressof @bad_name : !llvm.ptr
    %s2 = llvm.mlir.addressof @".str.2" : !llvm.ptr
    %e2a = llvm.insertvalue %f2, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2b = llvm.insertvalue %s2, %e2a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2c = llvm.insertvalue %file, %e2b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2d = llvm.insertvalue %line, %e2c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2 = llvm.insertvalue %zero, %e2d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t3 = llvm.insertvalue %e2, %t2[2] : !llvm.array<6 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f3 = llvm.mlir.addressof @bad_name : !llvm.ptr
    %s3 = llvm.mlir.addressof @".str.3" : !llvm.ptr
    %e3a = llvm.insertvalue %f3, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3b = llvm.insertvalue %s3, %e3a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3c = llvm.insertvalue %file, %e3b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3d = llvm.insertvalue %line, %e3c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3 = llvm.insertvalue %zero, %e3d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t4 = llvm.insertvalue %e3, %t3[3] : !llvm.array<6 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f4 = llvm.mlir.addressof @on_assumes : !llvm.ptr
    %s4 = llvm.mlir.addressof @".str.4" : !llvm.ptr
    %e4a = llvm.insertvalue %f4, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e4b = llvm.insertvalue %s4, %e4a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e4c = llvm.insertvalue %file, %e4b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e4d = llvm.insertvalue %line, %e4c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e4 = llvm.insertvalue %zero, %e4d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t5 = llvm.insertvalue %e4, %t4[4] : !llvm.array<6 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f5 = llvm.mlir.addressof @on_assumes : !llvm.ptr
    %s5 = llvm.mlir.addressof @".str.5" : !llvm.ptr
    %e5a = llvm.insertvalue %f5, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e5b = llvm.insertvalue %s5, %e5a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e5c = llvm.insertvalue %file, %e5b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e5d = llvm.insertvalue %line, %e5c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e5 = llvm.insertvalue %zero, %e5d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t6 = llvm.insertvalue %e5, %t5[5] : !llvm.array<6 x struct<(ptr, ptr, ptr, i32, ptr)>>
    llvm.return %t6 : !llvm.array<6 x struct<(ptr, ptr, ptr, i32, ptr)>>
  }

  // CHECK-LABEL: llvm.func @gram
  // CHECK-SAME: tessera.conditional_establishes = [{arg = 2 : i64, condition = "FullColumnRank(A) && A == B && trans == Trans", constants = {Trans = 1 : i64}, names = {A = 0 : i64, B = 1 : i64, trans = 3 : i64}, property = "SPD"}]
  llvm.func @gram(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32)

  // expected-warning @+3 {{ignoring tessera guarantees 'SPD(C) if FullColumnRank(Z)': the condition names 'Z', which is not a parameter of the function}}
  // CHECK-LABEL: llvm.func @bad_name
  // CHECK-NOT: conditional_establishes
  llvm.func @bad_name(!llvm.ptr, !llvm.ptr)

  // expected-warning @+2 {{ignoring tessera assumes 'SPD(A) if FullColumnRank(A)': only a guarantee may have a condition}}
  // CHECK-LABEL: llvm.func @on_assumes
  llvm.func @on_assumes(!llvm.ptr)
}
