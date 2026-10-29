// RUN: enzymexlamlir-opt %s -split-input-file -lift-perfify-annotations -verify-diagnostics | FileCheck %s

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
    // expected-warning @+1 {{ignoring perfify loop trip count outside an scf.for or scf.while body}}
    llvm.call @_Z16__perfify_loop_0v() : () -> ()
    llvm.return
  }
}

// CHECK-LABEL: llvm.func @no_loop
// CHECK-NOT: perfify.hypothesis
// CHECK-NOT: llvm.call
// CHECK: llvm.return
