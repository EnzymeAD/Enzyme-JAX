// RUN: enzymexlamlir-opt %s -split-input-file -lift-perfify-annotations -verify-diagnostics | FileCheck %s

// "pre eq M -> post le 2M + 1" assumes a symbolic starting cost of M and
// bounds the total cost at the end: { start_cost == M } f { fn_cost <= 2M + 1 }.

module {
  llvm.mlir.global private unnamed_addr constant @".str"("perfify_op=pre eq M -> post le 2M + 1\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.1"("pre_post.cpp\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %3 = llvm.mlir.addressof @".str" : !llvm.ptr
    %4 = llvm.mlir.addressof @f : !llvm.ptr
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
  llvm.func @f(%arg0: i64) -> i64 {
    %0 = llvm.mul %arg0, %arg0 : i64
    llvm.return %0 : i64
  }
}

// CHECK: perfify.define_symbol @M : i64
// CHECK: perfify.conditions @f true pre {
// CHECK-NEXT: %[[ARG0:.*]] = perfify.arg 0
// CHECK-NEXT: perfify.estimated_cost @pre_cost {
// CHECK-NEXT: %[[M:.*]] = perfify.sym_cost @M : i64
// CHECK-NEXT: perfify.yield %[[M]] : i64
// CHECK-NEXT: }
// CHECK-NEXT: %[[PRE:.*]] = perfify.cmp eq, %[[ARG0]], @pre_cost
// CHECK: } post {
// CHECK: perfify.estimated_cost @post_cost {
// CHECK: perfify.sym_cost @M : i64
// CHECK: perfify.cmp le, %{{.*}}, @post_cost

// -----

// A "pre" precondition must be followed by a "post" label.

module {
  llvm.mlir.global private unnamed_addr constant @".str"("perfify_op=pre eq M -> le 2M + 1\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.1"("pre_post.cpp\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %3 = llvm.mlir.addressof @".str" : !llvm.ptr
    %4 = llvm.mlir.addressof @missing_post : !llvm.ptr
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
  // expected-warning @below {{ignoring malformed perfify cost hypothesis}}
  llvm.func @missing_post(%arg0: i64) -> i64 {
    %0 = llvm.mul %arg0, %arg0 : i64
    llvm.return %0 : i64
  }
}

// CHECK-LABEL: llvm.func @missing_post
// CHECK-NOT: perfify.conditions

// -----

// A precondition must be labeled "pre": the old "arg<N>" form is rejected.

module {
  llvm.mlir.global private unnamed_addr constant @".str"("perfify_op=arg0 eq 5 -> post le 9\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.1"("pre_post.cpp\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %3 = llvm.mlir.addressof @".str" : !llvm.ptr
    %4 = llvm.mlir.addressof @old_arg : !llvm.ptr
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
  // expected-warning @below {{ignoring malformed perfify cost hypothesis 'arg0 eq 5 -> post le 9'}}
  llvm.func @old_arg(%arg0: i64) -> i64 {
    %0 = llvm.mul %arg0, %arg0 : i64
    llvm.return %0 : i64
  }
}

// CHECK-LABEL: llvm.func @old_arg
// CHECK-NOT: perfify.conditions
