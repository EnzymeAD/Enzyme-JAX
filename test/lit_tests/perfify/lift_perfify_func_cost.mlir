// RUN: enzymexlamlir-opt %s -split-input-file -lift-perfify-annotations -verify-diagnostics | FileCheck %s

// A function takes one perfify annotation: a cost hypothesis or a func cost.

// a hypothesis and a func cost on one function
module {
  llvm.mlir.global private unnamed_addr constant @".str"("perfify_op=le 9\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.1"("func_cost.cpp\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.2"("perfify_func_cost=10\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %3 = llvm.mlir.addressof @".str" : !llvm.ptr
    %4 = llvm.mlir.addressof @hello : !llvm.ptr
    %5 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %6 = llvm.insertvalue %4, %5[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %7 = llvm.insertvalue %3, %6[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %8 = llvm.insertvalue %2, %7[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %9 = llvm.insertvalue %1, %8[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %10 = llvm.insertvalue %0, %9[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %11 = llvm.mlir.addressof @".str.2" : !llvm.ptr
    %12 = llvm.insertvalue %4, %5[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %13 = llvm.insertvalue %11, %12[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %14 = llvm.insertvalue %2, %13[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %15 = llvm.insertvalue %1, %14[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %16 = llvm.insertvalue %0, %15[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %17 = llvm.mlir.undef : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %18 = llvm.insertvalue %10, %17[0] : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %19 = llvm.insertvalue %16, %18[1] : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>>
    llvm.return %19 : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>>
  }
  // expected-error @+1 {{conflicting perfify annotations}}
  llvm.func @hello() {
    llvm.return
  }
}

// -----

// two different hypotheses on one function
module {
  llvm.mlir.global private unnamed_addr constant @".str"("perfify_op=le 9\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.1"("func_cost.cpp\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.2"("perfify_op=le 4\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %3 = llvm.mlir.addressof @".str" : !llvm.ptr
    %4 = llvm.mlir.addressof @hello : !llvm.ptr
    %5 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %6 = llvm.insertvalue %4, %5[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %7 = llvm.insertvalue %3, %6[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %8 = llvm.insertvalue %2, %7[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %9 = llvm.insertvalue %1, %8[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %10 = llvm.insertvalue %0, %9[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %11 = llvm.mlir.addressof @".str.2" : !llvm.ptr
    %12 = llvm.insertvalue %4, %5[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %13 = llvm.insertvalue %11, %12[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %14 = llvm.insertvalue %2, %13[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %15 = llvm.insertvalue %1, %14[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %16 = llvm.insertvalue %0, %15[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %17 = llvm.mlir.undef : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %18 = llvm.insertvalue %10, %17[0] : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %19 = llvm.insertvalue %16, %18[1] : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>>
    llvm.return %19 : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>>
  }
  // expected-error @+1 {{conflicting perfify annotations}}
  llvm.func @hello() {
    llvm.return
  }
}

// -----

// an unrelated annotation neither conflicts with nor replaces a perfify one
module {
  llvm.mlir.global private unnamed_addr constant @".str"("perfify_op=le 9\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.1"("func_cost.cpp\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.2"("foo\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %3 = llvm.mlir.addressof @".str" : !llvm.ptr
    %4 = llvm.mlir.addressof @hello : !llvm.ptr
    %5 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %6 = llvm.insertvalue %4, %5[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %7 = llvm.insertvalue %3, %6[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %8 = llvm.insertvalue %2, %7[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %9 = llvm.insertvalue %1, %8[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %10 = llvm.insertvalue %0, %9[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %11 = llvm.mlir.addressof @".str.2" : !llvm.ptr
    %12 = llvm.insertvalue %4, %5[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %13 = llvm.insertvalue %11, %12[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %14 = llvm.insertvalue %2, %13[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %15 = llvm.insertvalue %1, %14[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %16 = llvm.insertvalue %0, %15[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %17 = llvm.mlir.undef : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %18 = llvm.insertvalue %10, %17[0] : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %19 = llvm.insertvalue %16, %18[1] : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>>
    llvm.return %19 : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>>
  }
  llvm.func @hello() {
    llvm.return
  }
}

// CHECK: perfify.conditions @hello true pre {

// -----

// a constant cost
module {
  llvm.mlir.global private unnamed_addr constant @".str"("perfify_func_cost=10\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.1"("func_cost.cpp\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %3 = llvm.mlir.addressof @".str" : !llvm.ptr
    %4 = llvm.mlir.addressof @hello : !llvm.ptr
    %5 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %6 = llvm.insertvalue %4, %5[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %7 = llvm.insertvalue %3, %6[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %8 = llvm.insertvalue %2, %7[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %9 = llvm.insertvalue %1, %8[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %10 = llvm.insertvalue %0, %9[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %11 = llvm.mlir.undef : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %12 = llvm.insertvalue %10, %11[0] : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>>
    llvm.return %12 : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>>
  }
  llvm.func @hello() {
    llvm.return
  }
}

// CHECK: perfify.func_bound @hello {
// CHECK-NEXT: %[[C10:.*]] = arith.constant 10 : i64
// CHECK-NEXT: perfify.yield %[[C10]] : i64
// CHECK-NEXT: }

// -----

// a symbolic cost declares its symbols
module {
  llvm.mlir.global private unnamed_addr constant @".str"("perfify_func_cost=3X + 1\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.1"("func_cost.cpp\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %3 = llvm.mlir.addressof @".str" : !llvm.ptr
    %4 = llvm.mlir.addressof @hello : !llvm.ptr
    %5 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %6 = llvm.insertvalue %4, %5[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %7 = llvm.insertvalue %3, %6[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %8 = llvm.insertvalue %2, %7[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %9 = llvm.insertvalue %1, %8[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %10 = llvm.insertvalue %0, %9[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %11 = llvm.mlir.undef : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %12 = llvm.insertvalue %10, %11[0] : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>>
    llvm.return %12 : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>>
  }
  llvm.func @hello() {
    llvm.return
  }
}

// CHECK: perfify.define_symbol @X : i64
// CHECK: perfify.func_bound @hello {
// CHECK-NEXT: %[[C3:.*]] = arith.constant 3 : i64
// CHECK-NEXT: %[[X:.*]] = perfify.sym_cost @X : i64
// CHECK-NEXT: %[[MUL:.*]] = arith.muli %[[C3]], %[[X]] : i64
// CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : i64
// CHECK-NEXT: %[[ADD:.*]] = arith.addi %[[MUL]], %[[C1]] : i64
// CHECK-NEXT: perfify.yield %[[ADD]] : i64
// CHECK-NEXT: }

// -----

// a cost symbol can't reuse the name of another symbol
module {
  llvm.mlir.global private unnamed_addr constant @".str"("perfify_func_cost=hello\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.1"("func_cost.cpp\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %3 = llvm.mlir.addressof @".str" : !llvm.ptr
    %4 = llvm.mlir.addressof @hello : !llvm.ptr
    %5 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %6 = llvm.insertvalue %4, %5[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %7 = llvm.insertvalue %3, %6[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %8 = llvm.insertvalue %2, %7[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %9 = llvm.insertvalue %1, %8[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %10 = llvm.insertvalue %0, %9[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %11 = llvm.mlir.undef : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %12 = llvm.insertvalue %10, %11[0] : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>>
    llvm.return %12 : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>>
  }
  // expected-error @+1 {{perfify cost symbol 'hello' conflicts with an existing symbol}}
  llvm.func @hello() {
    llvm.return
  }
}

// -----

// a malformed cost is ignored
module {
  llvm.mlir.global private unnamed_addr constant @".str"("perfify_func_cost=-5\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.1"("func_cost.cpp\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %3 = llvm.mlir.addressof @".str" : !llvm.ptr
    %4 = llvm.mlir.addressof @malformed : !llvm.ptr
    %5 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %6 = llvm.insertvalue %4, %5[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %7 = llvm.insertvalue %3, %6[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %8 = llvm.insertvalue %2, %7[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %9 = llvm.insertvalue %1, %8[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %10 = llvm.insertvalue %0, %9[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %11 = llvm.mlir.undef : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %12 = llvm.insertvalue %10, %11[0] : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>>
    llvm.return %12 : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>>
  }
  // expected-warning @+1 {{ignoring malformed perfify function cost '-5'}}
  llvm.func @malformed() {
    llvm.return
  }
}

// CHECK: llvm.func @malformed
// CHECK-NOT: perfify.func_bound
