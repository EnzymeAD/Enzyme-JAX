// RUN: enzymexlamlir-opt %s -lift-perfify-annotations | FileCheck %s
// RUN: enzymexlamlir-opt %s -lift-perfify-annotations | FileCheck %s --check-prefix=NOBB

// Two annotated functions: both perfify.conditions go in the one block of the
// (terminator-less) perfify.assumptions op.

module {
  llvm.mlir.global private unnamed_addr constant @".str"("perfify_op=le 9\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.1"("multi.cpp\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.2"("perfify_op=le 4\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %3 = llvm.mlir.addressof @".str" : !llvm.ptr
    %4 = llvm.mlir.addressof @square : !llvm.ptr
    %5 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %6 = llvm.insertvalue %4, %5[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %7 = llvm.insertvalue %3, %6[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %8 = llvm.insertvalue %2, %7[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %9 = llvm.insertvalue %1, %8[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %10 = llvm.insertvalue %0, %9[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %11 = llvm.mlir.addressof @".str.2" : !llvm.ptr
    %12 = llvm.mlir.addressof @double : !llvm.ptr
    %13 = llvm.insertvalue %12, %5[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %14 = llvm.insertvalue %11, %13[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %15 = llvm.insertvalue %2, %14[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %16 = llvm.insertvalue %1, %15[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %17 = llvm.insertvalue %0, %16[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %18 = llvm.mlir.undef : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %19 = llvm.insertvalue %10, %18[0] : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %20 = llvm.insertvalue %17, %19[1] : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>>
    llvm.return %20 : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>>
  }
  llvm.func @square(%arg0: i64) -> i64 {
    %0 = llvm.mul %arg0, %arg0 : i64
    llvm.return %0 : i64
  }
  llvm.func @double(%arg0: i64) -> i64 {
    %0 = llvm.add %arg0, %arg0 : i64
    llvm.return %0 : i64
  }
}

// CHECK: perfify.assumptions {
// CHECK-DAG: perfify.conditions @square true pre {
// CHECK-DAG: perfify.conditions @double true pre {

// NOBB-NOT: ^bb
