// RUN: enzymexlamlir-opt %s -lift-perfify-annotations | FileCheck %s

// Models the MLIR imported from the LLVM IR that the Reactant clang plugin
// produces for cost_hypothesis.cpp: the
// __attribute__((perfify_op("arg0 eq 0 -> le 9"))) annotation on square_mul
// becomes a "perfify_op=arg0 eq 0 -> le 9" entry in
// @llvm.global.annotations. The pass lifts it into a perfify.conditions
// Hoare triple ({ arg0 == 0 } square_mul { fn_cost <= 9 }) inside the
// module's perfify.assumptions op.

module {
  llvm.mlir.global private unnamed_addr constant @".str"("perfify_op=arg0 eq 0 -> le 9\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.1"("cost_hypothesis.cpp\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(6 : i32) : i32
    %2 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %3 = llvm.mlir.addressof @".str" : !llvm.ptr
    %4 = llvm.mlir.addressof @square_mul : !llvm.ptr
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
  llvm.func @square_mul(%arg0: i64) -> i64 {
    %0 = llvm.mul %arg0, %arg0 : i64
    %1 = llvm.mul %0, %0 : i64
    %2 = llvm.mul %1, %1 : i64
    llvm.return %2 : i64
  }
}

// CHECK: llvm.func @square_mul(%[[ARG:.*]]: i64) -> i64 attributes {perfify_cond = "arg0 eq 0 -> le 9"} {
// CHECK: perfify.assumptions {
// CHECK-NEXT: perfify.conditions @square_mul true pre {
// CHECK-NEXT: %[[ARG0:.*]] = perfify.arg 0
// CHECK-NEXT: %[[ZERO:.*]] = perfify.constant_cost {
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i64
// CHECK-NEXT: perfify.yield
// CHECK-NEXT: } : !perfify.cost
// CHECK-NEXT: %[[PRE:.*]] = perfify.cmp eq, %[[ARG0]], %[[ZERO]]
// CHECK-NEXT: perfify.assume %[[PRE]]
// CHECK-NEXT: } post {
// CHECK-NEXT: %[[FNCOST:.*]] = perfify.fn_cost : !perfify.cost
// CHECK-NEXT: %[[BOUND:.*]] = perfify.constant_cost {
// CHECK-NEXT: %[[C9:.*]] = arith.constant 9 : i64
// CHECK-NEXT: perfify.yield
// CHECK-NEXT: } : !perfify.cost
// CHECK-NEXT: %[[POST:.*]] = perfify.cmp le, %[[FNCOST]], %[[BOUND]]
// CHECK-NEXT: perfify.assume %[[POST]]
