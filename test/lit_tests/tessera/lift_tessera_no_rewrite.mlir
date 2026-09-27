// RUN: enzymexlamlir-opt %s -lift-tessera-annotations | FileCheck %s

// `__attribute__((annotate("tessera_no_rewrite")))` marks a function whose body
// no rule may rewrite. It needs no plugin support: a plain annotation lands in
// llvm.global.annotations like any other. A function can carry it alongside
// its tessera_op annotation, so every annotation on a function is kept, not
// just the last one found.

module {
  llvm.mlir.global private unnamed_addr constant @".str"("tessera_op=lib.calc_inverse(x)\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.1"("tessera_no_rewrite\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.2"("file.cpp\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<3 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.addressof @".str.2" : !llvm.ptr
    %3 = llvm.mlir.addressof @".str" : !llvm.ptr
    %4 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %5 = llvm.mlir.addressof @calc_inverse : !llvm.ptr
    %6 = llvm.mlir.addressof @fallback : !llvm.ptr
    %7 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %8 = llvm.insertvalue %5, %7[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %9 = llvm.insertvalue %3, %8[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %10 = llvm.insertvalue %2, %9[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %11 = llvm.insertvalue %1, %10[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %12 = llvm.insertvalue %0, %11[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %13 = llvm.insertvalue %5, %7[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %14 = llvm.insertvalue %4, %13[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %15 = llvm.insertvalue %2, %14[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %16 = llvm.insertvalue %1, %15[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %17 = llvm.insertvalue %0, %16[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %18 = llvm.insertvalue %6, %7[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %19 = llvm.insertvalue %4, %18[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %20 = llvm.insertvalue %2, %19[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %21 = llvm.insertvalue %1, %20[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %22 = llvm.insertvalue %0, %21[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %23 = llvm.mlir.undef : !llvm.array<3 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %24 = llvm.insertvalue %12, %23[0] : !llvm.array<3 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %25 = llvm.insertvalue %17, %24[1] : !llvm.array<3 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %26 = llvm.insertvalue %22, %25[2] : !llvm.array<3 x struct<(ptr, ptr, ptr, i32, ptr)>>
    llvm.return %26 : !llvm.array<3 x struct<(ptr, ptr, ptr, i32, ptr)>>
  }

  // CHECK: llvm.func @calc_inverse
  // CHECK-SAME: tessera.no_rewrite
  // CHECK-SAME: tessera_op = "lib.calc_inverse(x)"
  llvm.func @calc_inverse(%arg0 : f32) -> f32 attributes {no_inline} {
    llvm.return %arg0 : f32
  }

  // CHECK: llvm.func @fallback
  // CHECK-SAME: tessera.no_rewrite
  llvm.func @fallback(%arg0 : f32) -> f32 attributes {no_inline} {
    llvm.return %arg0 : f32
  }
}
