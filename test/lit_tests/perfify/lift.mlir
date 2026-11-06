// RUN: enzymexlamlir-opt %s -split-input-file -lift-perfify-annotations -verify-diagnostics | FileCheck %s

// Models the MLIR imported from the LLVM IR that the Reactant clang plugin
// produces for cost_hypothesis.cpp: the
// __attribute__((perfify_op("pre eq 0 -> post le 9"))) annotation on square_mul
// becomes a "perfify_op=pre eq 0 -> post le 9" entry in
// @llvm.global.annotations. The pass lifts it into a perfify.conditions
// Hoare triple ({ start_cost == 0 } square_mul { fn_cost <= 9 }) inside the
// module's perfify.assumptions op.

module {
  llvm.mlir.global private unnamed_addr constant @".str"("perfify_op=pre eq 0 -> post le 9\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
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

// CHECK: llvm.func @square_mul(%[[ARG:.*]]: i64) -> i64 attributes {perfify_cond = "pre eq 0 -> post le 9"} {
// CHECK: perfify.assumptions {
// CHECK-NEXT: perfify.conditions @square_mul true pre {
// CHECK-NEXT: %[[ARG0:.*]] = perfify.arg 0
// CHECK-NEXT: perfify.estimated_cost @pre_cost {
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i64
// CHECK-NEXT: perfify.yield
// CHECK-NEXT: }
// CHECK-NEXT: %[[PRE:.*]] = perfify.cmp eq, %[[ARG0]], @pre_cost
// CHECK-NEXT: %[[PREH:.*]] = perfify.hypothesis %[[PRE]] pre
// CHECK-NEXT: perfify.assume %[[PREH]]
// CHECK-NEXT: } post {
// CHECK-NEXT: %[[FNCOST:.*]] = perfify.fn_cost : !perfify.cost
// CHECK-NEXT: perfify.estimated_cost @post_cost {
// CHECK-NEXT: %[[C9:.*]] = arith.constant 9 : i64
// CHECK-NEXT: perfify.yield
// CHECK-NEXT: }
// CHECK-NEXT: %[[POST:.*]] = perfify.cmp le, %[[FNCOST]], @post_cost
// CHECK-NEXT: %[[POSTH:.*]] = perfify.hypothesis %[[POST]] post
// CHECK-NEXT: perfify.assume %[[POSTH]]

// -----

// for loop with trip in [1, N]
module {
  llvm.mlir.global private unnamed_addr constant @".str"("perfify_loop=eq [1, N]\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.1"("loops.cpp\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(3 : i32) : i32
    %2 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %3 = llvm.mlir.addressof @".str" : !llvm.ptr
    %4 = llvm.mlir.addressof @_Z16__perfify_loop_0v : !llvm.ptr
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
  llvm.func weak @_Z16__perfify_loop_0v() attributes {no_inline} {
    llvm.return
  }
  llvm.func @for_interval(%n: i64) {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    scf.for %i = %c0 to %n step %c1 : i64 {
      llvm.call @_Z16__perfify_loop_0v() : () -> ()
    }
    llvm.return
  }
}

// CHECK: perfify.define_symbol @N : i64
// CHECK-LABEL: llvm.func @for_interval
// CHECK: scf.for
// CHECK-NEXT: %[[T:.*]] = perfify.arg 0
// CHECK-NEXT: perfify.estimated_cost @trip_cost_min {
// CHECK: perfify.estimated_cost @trip_cost_max {
// CHECK: perfify.interval @trip_range [min_cost: @trip_cost_min, max_cost: @trip_cost_max]
// CHECK-NEXT: %[[CMP:.*]] = perfify.cmp eq, %[[T]], @trip_range
// CHECK-NEXT: %{{.*}} = perfify.hypothesis %[[CMP]]{{$}}
// CHECK-NOT: llvm.call
// CHECK: }

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

// CHECK: perfify.func_bound @hello eq {
// CHECK-NEXT: perfify.estimated_cost @func_cost {
// CHECK-NEXT: %[[C10:.*]] = arith.constant 10 : i64
// CHECK-NEXT: perfify.yield %[[C10]] : i64
// CHECK-NEXT: }
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
// CHECK: perfify.func_bound @hello eq {
// CHECK-NEXT: perfify.estimated_cost @func_cost {
// CHECK-NEXT: %[[C3:.*]] = arith.constant 3 : i64
// CHECK-NEXT: %[[X:.*]] = perfify.sym_cost @X : i64
// CHECK-NEXT: %[[MUL:.*]] = arith.muli %[[C3]], %[[X]] : i64
// CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : i64
// CHECK-NEXT: %[[ADD:.*]] = arith.addi %[[MUL]], %[[C1]] : i64
// CHECK-NEXT: perfify.yield %[[ADD]] : i64
// CHECK-NEXT: }
// CHECK-NEXT: }

// -----

// an interval
module {
  llvm.mlir.global private unnamed_addr constant @".str"("perfify_func_cost=eq [N, 3N]\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
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

// CHECK: perfify.define_symbol @N : i64
// CHECK: perfify.func_bound @hello eq {
// CHECK-NEXT: perfify.estimated_cost @func_cost_min {
// CHECK-NEXT: %[[MIN:.*]] = perfify.sym_cost @N : i64
// CHECK-NEXT: perfify.yield %[[MIN]] : i64
// CHECK-NEXT: }
// CHECK-NEXT: perfify.estimated_cost @func_cost_max {
// CHECK-NEXT: %[[C3:.*]] = arith.constant 3 : i64
// CHECK-NEXT: %[[N:.*]] = perfify.sym_cost @N : i64
// CHECK-NEXT: %[[MAX:.*]] = arith.muli %[[C3]], %[[N]] : i64
// CHECK-NEXT: perfify.yield %[[MAX]] : i64
// CHECK-NEXT: }
// CHECK-NEXT: perfify.interval @func_range [min_cost: @func_cost_min, max_cost: @func_cost_max]
// CHECK-NEXT: }

// -----

// #pragma perfify_cost "arith.addi = 3X + 1" annotates a marker global.
module {
  llvm.mlir.global internal @__perfify_cost_0(0 : i32) {addr_space = 0 : i32, alignment = 4 : i64, dso_local} : i32
  llvm.mlir.global private unnamed_addr constant @".str"("perfify_cost=arith.addi = 3X + 1\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.1"("<invalid loc>\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %3 = llvm.mlir.addressof @".str" : !llvm.ptr
    %4 = llvm.mlir.addressof @__perfify_cost_0 : !llvm.ptr
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
}

// CHECK: perfify.define_symbol @X : i64
// CHECK: perfify.assumptions {
// CHECK-NEXT: perfify.cost "arith.addi" {
// CHECK-NEXT: %[[C3:.*]] = arith.constant 3 : i64
// CHECK-NEXT: %[[X:.*]] = perfify.sym_cost @X : i64
// CHECK-NEXT: %[[MUL:.*]] = arith.muli %[[C3]], %[[X]] : i64
// CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : i64
// CHECK-NEXT: %[[ADD:.*]] = arith.addi %[[MUL]], %[[C1]] : i64
// CHECK-NEXT: perfify.yield %[[ADD]] : i64
// CHECK-NEXT: }
// CHECK-NEXT: }

// -----

// #pragma perfify_cost "* = 0" sets the cost of every op without its own.
module {
  llvm.mlir.global internal @__perfify_cost_0(0 : i32) {addr_space = 0 : i32, alignment = 4 : i64, dso_local} : i32
  llvm.mlir.global private unnamed_addr constant @".str"("perfify_cost=* = 0\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.1"("<invalid loc>\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %3 = llvm.mlir.addressof @".str" : !llvm.ptr
    %4 = llvm.mlir.addressof @__perfify_cost_0 : !llvm.ptr
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
}

// CHECK: perfify.assumptions {
// CHECK-NEXT: perfify.cost "*" {
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i64
// CHECK-NEXT: perfify.yield %[[C0]] : i64
// CHECK-NEXT: }
// CHECK-NEXT: }

// -----

// #pragma perfify_func_cost "ext = 7" names @ext by its symbol.
module {
  llvm.mlir.global internal @__perfify_named_func_cost_0(0 : i32) {addr_space = 0 : i32, alignment = 4 : i64, dso_local} : i32
  llvm.mlir.global private unnamed_addr constant @".str"("perfify_named_func_cost=ext = 7\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.1"("<invalid loc>\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %3 = llvm.mlir.addressof @".str" : !llvm.ptr
    %4 = llvm.mlir.addressof @__perfify_named_func_cost_0 : !llvm.ptr
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
  llvm.func @ext(i32) -> i32
  llvm.func @f(%arg0: i32) -> i32 {
    %0 = llvm.call @ext(%arg0) : (i32) -> i32
    llvm.return %0 : i32
  }
}

// CHECK: perfify.func_bound @ext eq {
// CHECK-NEXT: perfify.estimated_cost @func_cost {
// CHECK-NEXT: %[[C7:.*]] = arith.constant 7 : i64
// CHECK-NEXT: perfify.yield %[[C7]] : i64
// CHECK-NEXT: }
// CHECK-NEXT: }
