// RUN: enzymexlamlir-opt %s -split-input-file -lift-perfify-annotations -verify-diagnostics | FileCheck %s

// Models the MLIR imported from the LLVM IR that the Reactant clang plugin
// produces for cost_hypothesis.cpp: the
// __attribute__((perfify_op("arg0 eq 0 -> le 9"))) annotation on square_mul
// becomes a "perfify_op=arg0 eq 0 -> le 9" entry in
// @llvm.global.annotations. The pass lifts it into a perfify.conditions
// Hoare triple ({ start_cost == 0 } square_mul { fn_cost <= 9 }) inside the
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

// CHECK-NOT: ^bb
// CHECK: }

// -----

// Models the plugin output for [[perfify_loop("...")]] after control flow is
// lifted to SCF: a call to an annotated marker function at the top of the loop
// body. The pass replaces the call with a trip count perfify.hypothesis.

// while loop with trip <= N
module {
  llvm.mlir.global private unnamed_addr constant @".str"("perfify_loop=le N\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
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
  llvm.func @while_le_n(%n: i64) -> i64 {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %r = scf.while (%i = %c0) : (i64) -> i64 {
      %c = llvm.icmp "slt" %i, %n : i64
      scf.condition(%c) %i : i64
    } do {
    ^bb0(%i2: i64):
      llvm.call @_Z16__perfify_loop_0v() : () -> ()
      %i3 = llvm.add %i2, %c1 : i64
      scf.yield %i3 : i64
    }
    llvm.return %r : i64
  }
}

// CHECK: perfify.define_symbol @N : i64
// CHECK-LABEL: llvm.func @while_le_n
// CHECK: } do {
// CHECK-NEXT: ^bb0(%{{.*}}: i64):
// CHECK-NEXT: %[[T:.*]] = perfify.arg 0
// CHECK-NEXT: perfify.estimated_cost @trip_cost {
// CHECK-NEXT: %[[N:.*]] = perfify.sym_cost @N : i64
// CHECK-NEXT: perfify.yield %[[N]] : i64
// CHECK-NEXT: }
// CHECK-NEXT: %[[CMP:.*]] = perfify.cmp le, %[[T]], @trip_cost
// CHECK-NEXT: %{{.*}} = perfify.hypothesis %[[CMP]]{{$}}
// CHECK-NOT: llvm.call @_Z16__perfify_loop_0v
// CHECK: scf.yield

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

// a loop constraint can't have a precondition: warn and drop the call
module {
  llvm.mlir.global private unnamed_addr constant @".str"("perfify_loop=arg0 eq 0 -> le N\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
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
  llvm.func @malformed(%n: i64) -> i64 {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %r = scf.while (%i = %c0) : (i64) -> i64 {
      %c = llvm.icmp "slt" %i, %n : i64
      scf.condition(%c) %i : i64
    } do {
    ^bb0(%i2: i64):
      // expected-warning @+1 {{ignoring malformed perfify loop trip count}}
      llvm.call @_Z16__perfify_loop_0v() : () -> ()
      %i3 = llvm.add %i2, %c1 : i64
      scf.yield %i3 : i64
    }
    llvm.return %r : i64
  }
}

// CHECK-LABEL: llvm.func @malformed
// CHECK-NOT: perfify.hypothesis
// CHECK-NOT: llvm.call
// CHECK: llvm.return

// -----

// a coefficient that overflows i64 is malformed: warn and drop the call
module {
  llvm.mlir.global private unnamed_addr constant @".str"("perfify_loop=le 9999999999*9999999999\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
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
  llvm.func @overflow(%n: i64) -> i64 {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %r = scf.while (%i = %c0) : (i64) -> i64 {
      %c = llvm.icmp "slt" %i, %n : i64
      scf.condition(%c) %i : i64
    } do {
    ^bb0(%i2: i64):
      // expected-warning @+1 {{ignoring malformed perfify loop trip count}}
      llvm.call @_Z16__perfify_loop_0v() : () -> ()
      %i3 = llvm.add %i2, %c1 : i64
      scf.yield %i3 : i64
    }
    llvm.return %r : i64
  }
}

// CHECK-LABEL: llvm.func @overflow
// CHECK-NOT: perfify.hypothesis
// CHECK-NOT: llvm.call
// CHECK: llvm.return

// -----

// the marker must be in a loop body: warn and drop the call
module {
  llvm.mlir.global private unnamed_addr constant @".str"("perfify_loop=le N\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
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
  llvm.func @no_loop() {
    // expected-warning @+1 {{ignoring perfify loop trip count outside the body of an scf.for, affine.for, scf.while or stablehlo.while}}
    llvm.call @_Z16__perfify_loop_0v() : () -> ()
    llvm.return
  }
}

// CHECK-LABEL: llvm.func @no_loop
// CHECK-NOT: perfify.hypothesis
// CHECK-NOT: llvm.call
// CHECK: llvm.return

// -----

// an affine.for body runs once per iteration
module {
  llvm.mlir.global private unnamed_addr constant @".str"("perfify_loop=le N\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
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
  func.func @affine_trip(%n: index) {
    affine.for %i = 0 to %n {
      llvm.call @_Z16__perfify_loop_0v() : () -> ()
    }
    return
  }
}

// CHECK: perfify.define_symbol @N : i64
// CHECK-LABEL: func.func @affine_trip
// CHECK: affine.for
// CHECK-NEXT: %[[T:.*]] = perfify.arg 0
// CHECK: %[[CMP:.*]] = perfify.cmp le, %[[T]], @trip_cost
// CHECK-NEXT: %{{.*}} = perfify.hypothesis %[[CMP]]{{$}}
// CHECK-NOT: llvm.call
// CHECK: }

// -----

// a stablehlo.while body runs once per iteration
module {
  llvm.mlir.global private unnamed_addr constant @".str"("perfify_loop=le N\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
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
  func.func @hlo_while_trip(%a: tensor<i64>) {
    %r = stablehlo.while(%i = %a) : tensor<i64>
    cond {
      %c = stablehlo.compare LT, %i, %i : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      llvm.call @_Z16__perfify_loop_0v() : () -> ()
      stablehlo.return %i : tensor<i64>
    }
    return
  }
}

// CHECK: perfify.define_symbol @N : i64
// CHECK-LABEL: func.func @hlo_while_trip
// CHECK: do {
// CHECK-NEXT: %[[T:.*]] = perfify.arg 0
// CHECK: %[[CMP:.*]] = perfify.cmp le, %[[T]], @trip_cost
// CHECK-NEXT: %{{.*}} = perfify.hypothesis %[[CMP]]{{$}}
// CHECK-NOT: llvm.call
// CHECK: stablehlo.return

// -----

// the condition of a stablehlo.while runs once more than its body: warn and
// drop the call
module {
  llvm.mlir.global private unnamed_addr constant @".str"("perfify_loop=le N\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
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
  func.func @hlo_while_cond(%a: tensor<i64>) {
    %r = stablehlo.while(%i = %a) : tensor<i64>
    cond {
      // expected-warning @+1 {{ignoring perfify loop trip count outside the body of an scf.for, affine.for, scf.while or stablehlo.while}}
      llvm.call @_Z16__perfify_loop_0v() : () -> ()
      %c = stablehlo.compare LT, %i, %i : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      stablehlo.return %i : tensor<i64>
    }
    return
  }
}

// CHECK-LABEL: func.func @hlo_while_cond
// CHECK-NOT: perfify.hypothesis
// CHECK-NOT: llvm.call
// CHECK: {{^ *}}return

// -----

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

// an interval open at the bottom
module {
  llvm.mlir.global private unnamed_addr constant @".str"("perfify_func_cost=eq [, 10]\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
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
// CHECK-NEXT: perfify.estimated_cost @func_cost_max {
// CHECK-NEXT: %[[C10:.*]] = arith.constant 10 : i64
// CHECK-NEXT: perfify.yield %[[C10]] : i64
// CHECK-NEXT: }
// CHECK-NEXT: perfify.interval @func_range [max_cost: @func_cost_max]
// CHECK-NEXT: }

// -----

// a predicate other than eq
module {
  llvm.mlir.global private unnamed_addr constant @".str"("perfify_func_cost=le 10\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
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

// CHECK: perfify.func_bound @hello le {
// CHECK-NEXT: perfify.estimated_cost @func_cost {
// CHECK-NEXT: %[[C10:.*]] = arith.constant 10 : i64
// CHECK-NEXT: perfify.yield %[[C10]] : i64
// CHECK-NEXT: }
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

// -----

// a given cost has no precondition
module {
  llvm.mlir.global private unnamed_addr constant @".str"("perfify_func_cost=arg0 eq 0 -> le 9\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.1"("func_cost.cpp\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %3 = llvm.mlir.addressof @".str" : !llvm.ptr
    %4 = llvm.mlir.addressof @precondition : !llvm.ptr
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
  // expected-warning @+1 {{ignoring malformed perfify function cost 'arg0 eq 0 -> le 9'}}
  llvm.func @precondition() {
    llvm.return
  }
}

// CHECK: llvm.func @precondition
// CHECK-NOT: perfify.func_bound
