// RUN: enzymexlamlir-opt %s -lift-tessera-annotations="warn-unannotated-handles=true" -verify-diagnostics

// With warn-unannotated-handles, a tessera op's handle that is neither
// unchanged, given a property nor preserved is reported. ksp and A are
// unchanged (marked with the older spelling, tessera_readonly); P is not.

module {
  llvm.mlir.global private unnamed_addr constant @".str.0"("tessera_op=petsc.ksp_set_operators(ksp, A, P)\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.1"("tessera_params=ksp,A,P\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.2"("tessera_readonly=ksp,A\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.file"("file.c\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<3 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %zero = llvm.mlir.zero : !llvm.ptr
    %file = llvm.mlir.addressof @".str.file" : !llvm.ptr
    %line = llvm.mlir.constant(7 : i32) : i32
    %undef = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t0 = llvm.mlir.undef : !llvm.array<3 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %f = llvm.mlir.addressof @KSPSetOperators : !llvm.ptr
    %s0 = llvm.mlir.addressof @".str.0" : !llvm.ptr
    %e0a = llvm.insertvalue %f, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0b = llvm.insertvalue %s0, %e0a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0c = llvm.insertvalue %file, %e0b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0d = llvm.insertvalue %line, %e0c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e0 = llvm.insertvalue %zero, %e0d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t1 = llvm.insertvalue %e0, %t0[0] : !llvm.array<3 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %s1 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %e1a = llvm.insertvalue %f, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1b = llvm.insertvalue %s1, %e1a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1c = llvm.insertvalue %file, %e1b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1d = llvm.insertvalue %line, %e1c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e1 = llvm.insertvalue %zero, %e1d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t2 = llvm.insertvalue %e1, %t1[1] : !llvm.array<3 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %s2 = llvm.mlir.addressof @".str.2" : !llvm.ptr
    %e2a = llvm.insertvalue %f, %undef[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2b = llvm.insertvalue %s2, %e2a[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2c = llvm.insertvalue %file, %e2b[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2d = llvm.insertvalue %line, %e2c[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %e2 = llvm.insertvalue %zero, %e2d[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %t3 = llvm.insertvalue %e2, %t2[2] : !llvm.array<3 x struct<(ptr, ptr, ptr, i32, ptr)>>
    llvm.return %t3 : !llvm.array<3 x struct<(ptr, ptr, ptr, i32, ptr)>>
  }

  // expected-warning @+1 {{tessera op takes handle 'P' without unchanged, a guarantee or a preserves}}
  llvm.func @KSPSetOperators(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
}
