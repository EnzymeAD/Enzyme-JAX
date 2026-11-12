// RUN: enzymexlamlir-opt %s -lift-tessera-annotations -verify-diagnostics | FileCheck %s

// A guarantee may hold only under a condition. One about a handle
// parameter's object becomes tessera.conditional_establishes; one about a
// value the function returns or writes becomes tessera.conditional_guarantees,
// keyed by the result or the parameter (sret included) it is on.
//
// A preserves is short for the guarantee whose condition is itself:
// tessera_preserves("sorted(this)") is tessera_guarantees("sorted(this) if
// sorted(this)"), so the two spellings lift to the same attribute. A condition
// of the preserves' own is added to that one.

module {
  llvm.mlir.global private unnamed_addr constant @".str.0"("tessera_params=this,pred\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.1"("tessera_preserves=sorted(this)\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.2"("tessera_guarantees=sorted(this) if sorted(this)\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.3"("tessera_params=this,x\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.4"("tessera_preserves=sorted(this), unique(this)\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.5"("tessera_op=lib.normalize(v:val=inout, n)\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.6"("tessera_params=v,n\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.7"("tessera_preserves=sorted(v) if n > 0\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.8"("tessera_op=lib.sum(a:val=in, b:val=in)\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.9"("tessera_params=a,b\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.10"("tessera_guarantees=SPD if SPD(a) && SPD(b)\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.11"("tessera_params=A\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.12"("tessera_guarantees=positive if SPD(A)\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.13"("tessera_op=lib.mistakes(A, b:val=in, out:val=out)\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.14"("tessera_params=A,b,out\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.15"("tessera_preserves=SPD,A\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.16"("tessera_preserves=sorted\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.17"("tessera_preserves=sorted(out)\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.18"("tessera_preserves=sorted(b)\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.file"("file.cpp\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<20 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %zero = llvm.mlir.zero : !llvm.ptr
    %file = llvm.mlir.addressof @".str.file" : !llvm.ptr
    %line = llvm.mlir.constant(5 : i32) : i32
    %undef = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t0 = llvm.mlir.undef : !llvm.array<20 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f0 = llvm.mlir.addressof @remove_if : !llvm.ptr
    %s0 = llvm.mlir.addressof @".str.0" : !llvm.ptr
    %e0a = llvm.insertvalue %f0, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0b = llvm.insertvalue %s0, %e0a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0c = llvm.insertvalue %file, %e0b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0d = llvm.insertvalue %line, %e0c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0 = llvm.insertvalue %zero, %e0d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t1 = llvm.insertvalue %e0, %t0[0] : !llvm.array<20 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f1 = llvm.mlir.addressof @remove_if : !llvm.ptr
    %s1 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %e1a = llvm.insertvalue %f1, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1b = llvm.insertvalue %s1, %e1a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1c = llvm.insertvalue %file, %e1b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1d = llvm.insertvalue %line, %e1c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1 = llvm.insertvalue %zero, %e1d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t2 = llvm.insertvalue %e1, %t1[1] : !llvm.array<20 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f2 = llvm.mlir.addressof @remove_if_spelled_out : !llvm.ptr
    %s2 = llvm.mlir.addressof @".str.0" : !llvm.ptr
    %e2a = llvm.insertvalue %f2, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2b = llvm.insertvalue %s2, %e2a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2c = llvm.insertvalue %file, %e2b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2d = llvm.insertvalue %line, %e2c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2 = llvm.insertvalue %zero, %e2d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t3 = llvm.insertvalue %e2, %t2[2] : !llvm.array<20 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f3 = llvm.mlir.addressof @remove_if_spelled_out : !llvm.ptr
    %s3 = llvm.mlir.addressof @".str.2" : !llvm.ptr
    %e3a = llvm.insertvalue %f3, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3b = llvm.insertvalue %s3, %e3a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3c = llvm.insertvalue %file, %e3b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3d = llvm.insertvalue %line, %e3c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e3 = llvm.insertvalue %zero, %e3d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t4 = llvm.insertvalue %e3, %t3[3] : !llvm.array<20 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f4 = llvm.mlir.addressof @push_sorted : !llvm.ptr
    %s4 = llvm.mlir.addressof @".str.3" : !llvm.ptr
    %e4a = llvm.insertvalue %f4, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e4b = llvm.insertvalue %s4, %e4a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e4c = llvm.insertvalue %file, %e4b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e4d = llvm.insertvalue %line, %e4c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e4 = llvm.insertvalue %zero, %e4d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t5 = llvm.insertvalue %e4, %t4[4] : !llvm.array<20 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f5 = llvm.mlir.addressof @push_sorted : !llvm.ptr
    %s5 = llvm.mlir.addressof @".str.4" : !llvm.ptr
    %e5a = llvm.insertvalue %f5, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e5b = llvm.insertvalue %s5, %e5a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e5c = llvm.insertvalue %file, %e5b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e5d = llvm.insertvalue %line, %e5c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e5 = llvm.insertvalue %zero, %e5d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t6 = llvm.insertvalue %e5, %t5[5] : !llvm.array<20 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f6 = llvm.mlir.addressof @normalize : !llvm.ptr
    %s6 = llvm.mlir.addressof @".str.5" : !llvm.ptr
    %e6a = llvm.insertvalue %f6, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e6b = llvm.insertvalue %s6, %e6a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e6c = llvm.insertvalue %file, %e6b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e6d = llvm.insertvalue %line, %e6c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e6 = llvm.insertvalue %zero, %e6d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t7 = llvm.insertvalue %e6, %t6[6] : !llvm.array<20 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f7 = llvm.mlir.addressof @normalize : !llvm.ptr
    %s7 = llvm.mlir.addressof @".str.6" : !llvm.ptr
    %e7a = llvm.insertvalue %f7, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e7b = llvm.insertvalue %s7, %e7a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e7c = llvm.insertvalue %file, %e7b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e7d = llvm.insertvalue %line, %e7c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e7 = llvm.insertvalue %zero, %e7d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t8 = llvm.insertvalue %e7, %t7[7] : !llvm.array<20 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f8 = llvm.mlir.addressof @normalize : !llvm.ptr
    %s8 = llvm.mlir.addressof @".str.7" : !llvm.ptr
    %e8a = llvm.insertvalue %f8, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e8b = llvm.insertvalue %s8, %e8a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e8c = llvm.insertvalue %file, %e8b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e8d = llvm.insertvalue %line, %e8c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e8 = llvm.insertvalue %zero, %e8d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t9 = llvm.insertvalue %e8, %t8[8] : !llvm.array<20 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f9 = llvm.mlir.addressof @sum : !llvm.ptr
    %s9 = llvm.mlir.addressof @".str.8" : !llvm.ptr
    %e9a = llvm.insertvalue %f9, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e9b = llvm.insertvalue %s9, %e9a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e9c = llvm.insertvalue %file, %e9b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e9d = llvm.insertvalue %line, %e9c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e9 = llvm.insertvalue %zero, %e9d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t10 = llvm.insertvalue %e9, %t9[9] : !llvm.array<20 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f10 = llvm.mlir.addressof @sum : !llvm.ptr
    %s10 = llvm.mlir.addressof @".str.9" : !llvm.ptr
    %e10a = llvm.insertvalue %f10, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e10b = llvm.insertvalue %s10, %e10a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e10c = llvm.insertvalue %file, %e10b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e10d = llvm.insertvalue %line, %e10c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e10 = llvm.insertvalue %zero, %e10d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t11 = llvm.insertvalue %e10, %t10[10] : !llvm.array<20 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f11 = llvm.mlir.addressof @sum : !llvm.ptr
    %s11 = llvm.mlir.addressof @".str.10" : !llvm.ptr
    %e11a = llvm.insertvalue %f11, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e11b = llvm.insertvalue %s11, %e11a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e11c = llvm.insertvalue %file, %e11b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e11d = llvm.insertvalue %line, %e11c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e11 = llvm.insertvalue %zero, %e11d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t12 = llvm.insertvalue %e11, %t11[11] : !llvm.array<20 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f12 = llvm.mlir.addressof @trace : !llvm.ptr
    %s12 = llvm.mlir.addressof @".str.11" : !llvm.ptr
    %e12a = llvm.insertvalue %f12, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e12b = llvm.insertvalue %s12, %e12a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e12c = llvm.insertvalue %file, %e12b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e12d = llvm.insertvalue %line, %e12c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e12 = llvm.insertvalue %zero, %e12d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t13 = llvm.insertvalue %e12, %t12[12] : !llvm.array<20 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f13 = llvm.mlir.addressof @trace : !llvm.ptr
    %s13 = llvm.mlir.addressof @".str.12" : !llvm.ptr
    %e13a = llvm.insertvalue %f13, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e13b = llvm.insertvalue %s13, %e13a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e13c = llvm.insertvalue %file, %e13b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e13d = llvm.insertvalue %line, %e13c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e13 = llvm.insertvalue %zero, %e13d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t14 = llvm.insertvalue %e13, %t13[13] : !llvm.array<20 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f14 = llvm.mlir.addressof @mistakes : !llvm.ptr
    %s14 = llvm.mlir.addressof @".str.13" : !llvm.ptr
    %e14a = llvm.insertvalue %f14, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e14b = llvm.insertvalue %s14, %e14a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e14c = llvm.insertvalue %file, %e14b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e14d = llvm.insertvalue %line, %e14c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e14 = llvm.insertvalue %zero, %e14d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t15 = llvm.insertvalue %e14, %t14[14] : !llvm.array<20 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f15 = llvm.mlir.addressof @mistakes : !llvm.ptr
    %s15 = llvm.mlir.addressof @".str.14" : !llvm.ptr
    %e15a = llvm.insertvalue %f15, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e15b = llvm.insertvalue %s15, %e15a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e15c = llvm.insertvalue %file, %e15b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e15d = llvm.insertvalue %line, %e15c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e15 = llvm.insertvalue %zero, %e15d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t16 = llvm.insertvalue %e15, %t15[15] : !llvm.array<20 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f16 = llvm.mlir.addressof @mistakes : !llvm.ptr
    %s16 = llvm.mlir.addressof @".str.15" : !llvm.ptr
    %e16a = llvm.insertvalue %f16, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e16b = llvm.insertvalue %s16, %e16a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e16c = llvm.insertvalue %file, %e16b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e16d = llvm.insertvalue %line, %e16c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e16 = llvm.insertvalue %zero, %e16d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t17 = llvm.insertvalue %e16, %t16[16] : !llvm.array<20 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f17 = llvm.mlir.addressof @mistakes : !llvm.ptr
    %s17 = llvm.mlir.addressof @".str.16" : !llvm.ptr
    %e17a = llvm.insertvalue %f17, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e17b = llvm.insertvalue %s17, %e17a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e17c = llvm.insertvalue %file, %e17b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e17d = llvm.insertvalue %line, %e17c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e17 = llvm.insertvalue %zero, %e17d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t18 = llvm.insertvalue %e17, %t17[17] : !llvm.array<20 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f18 = llvm.mlir.addressof @mistakes : !llvm.ptr
    %s18 = llvm.mlir.addressof @".str.17" : !llvm.ptr
    %e18a = llvm.insertvalue %f18, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e18b = llvm.insertvalue %s18, %e18a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e18c = llvm.insertvalue %file, %e18b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e18d = llvm.insertvalue %line, %e18c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e18 = llvm.insertvalue %zero, %e18d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t19 = llvm.insertvalue %e18, %t18[18] : !llvm.array<20 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f19 = llvm.mlir.addressof @mistakes : !llvm.ptr
    %s19 = llvm.mlir.addressof @".str.18" : !llvm.ptr
    %e19a = llvm.insertvalue %f19, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e19b = llvm.insertvalue %s19, %e19a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e19c = llvm.insertvalue %file, %e19b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e19d = llvm.insertvalue %line, %e19c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e19 = llvm.insertvalue %zero, %e19d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t20 = llvm.insertvalue %e19, %t19[19] : !llvm.array<20 x struct<(ptr, ptr, ptr, i32, ptr)>>
    llvm.return %t20 : !llvm.array<20 x struct<(ptr, ptr, ptr, i32, ptr)>>
  }

  // CHECK-LABEL: llvm.func @remove_if(
  // CHECK-SAME: tessera.conditional_establishes = [{arg = 0 : i64, condition = "sorted(this)", constants = {}, names = {this = 0 : i64}, property = "sorted"}]
  llvm.func @remove_if(!llvm.ptr, !llvm.ptr)

  // CHECK-LABEL: llvm.func @remove_if_spelled_out(
  // CHECK-SAME: tessera.conditional_establishes = [{arg = 0 : i64, condition = "sorted(this)", constants = {}, names = {this = 0 : i64}, property = "sorted"}]
  llvm.func @remove_if_spelled_out(!llvm.ptr, !llvm.ptr)

  // A preserves on a function that returns a value is still about the
  // argument it names, and one attribute can preserve several properties.
  // CHECK-LABEL: llvm.func @push_sorted(
  // CHECK-SAME: tessera.conditional_establishes = [{arg = 0 : i64, condition = "sorted(this)", constants = {}, names = {this = 0 : i64}, property = "sorted"}, {arg = 0 : i64, condition = "unique(this)", constants = {}, names = {this = 0 : i64}, property = "unique"}]
  llvm.func @push_sorted(!llvm.ptr, i32) -> i32

  // A val=inout argument keeps the property in the value the function writes
  // back, here only when n is positive.
  // CHECK-LABEL: llvm.func @normalize(
  // CHECK-SAME: tessera.conditional_guarantees = [{arg = 0 : i64, condition = "sorted(v) && (n > 0)", constants = {}, names = {n = 1 : i64, v = 0 : i64}, property = "sorted"}]
  llvm.func @normalize(!llvm.ptr, i32)

  // A sum returned by value, through an sret: what the older
  // tessera_preserves("SPD", "a", "b") said.
  // CHECK-LABEL: llvm.func @sum(
  // CHECK-SAME: tessera.conditional_guarantees = [{arg = 0 : i64, condition = "SPD(a) && SPD(b)", constants = {}, names = {a = 1 : i64, b = 2 : i64}, property = "SPD"}]
  llvm.func @sum(!llvm.ptr {llvm.sret = !llvm.struct<"Mat3", (array<9 x f64>)>}, !llvm.ptr, !llvm.ptr)

  // CHECK-LABEL: llvm.func @trace(
  // CHECK-SAME: tessera.conditional_guarantees = [{condition = "SPD(A)", constants = {}, names = {A = 0 : i64}, property = "positive", result = 0 : i64}]
  llvm.func @trace(!llvm.ptr) -> f64

  // CHECK-LABEL: llvm.func @mistakes(
  // CHECK-NOT: conditional
  // expected-warning @below {{file.cpp:5: ignoring tessera preserves 'SPD,A': a preserves now names each argument that keeps a property, as "sorted(this)"}}
  // expected-warning @below {{file.cpp:5: ignoring tessera preserves 'sorted': name the argument that keeps the property, as "sorted(this)"}}
  // expected-warning @below {{file.cpp:5: ignoring tessera preserves 'sorted(out)': 'out' is val=out, so it has no value on entry to keep the property from}}
  // expected-warning @below {{file.cpp:5: ignoring tessera preserves 'sorted(b)': 'b' is val=in, so the function only reads the value behind it}}
  llvm.func @mistakes(!llvm.ptr, !llvm.ptr, !llvm.ptr)
}
